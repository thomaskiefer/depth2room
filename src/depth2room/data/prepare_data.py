#!/usr/bin/env python3
"""Prepare CAD-Estate data for VACE depth-to-RGB training.

Renders ground-truth depth maps from room structure meshes and pairs them with
RGB frames to produce training data in the format expected by DiffSynth-Studio
VACE fine-tuning.

Frame sampling: picks a random consecutive window of (num_frames * stride)
source frames, then takes every `stride`-th frame to produce `num_frames`
output frames at approximately 16fps (source is ~30fps, stride=2 → ~15fps).

Outputs are organized into per-video subdirectories:
    {output_dir}/{video_id}/{clip_name}_depth.pt
    {output_dir}/{video_id}/{clip_name}_raw_depth.pt
    {output_dir}/{video_id}/{clip_name}_depth_meta.json
    {output_dir}/{video_id}/{clip_name}_rgb.mp4
    {output_dir}/{video_id}/{clip_name}_ref.jpg

Requires the cad_estate package. Set CAD_ESTATE_SRC to point at the
cad-estate/src directory, or install it:
    export CAD_ESTATE_SRC=/path/to/cad-estate/src

Example usage:
    python -m depth2room.data.prepare_data \
        --cad_estate_root /path/to/cad-estate \
        --num_workers 4 --num_frames 81 --stride 2 \
        --output_dir /path/to/vace_training_dataset
"""

import argparse
import asyncio
import io
import json
import logging
import os
import random
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torchvision.io
import torchvision.transforms.functional

# Target resolution for VACE training
TARGET_H, TARGET_W = 480, 832

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------
def discover_scenes(struct_all_path: str, frames_dir: str,
                    annotations_dir: str, min_frames: int) -> list[str]:
    """Find usable clip_names that have enough annotation frames."""
    with open(struct_all_path, "r") as f:
        clip_names = [line.strip() for line in f if line.strip()]
    assert len(clip_names) > 0, "struct_all.txt is empty"
    log.info("Loaded %d clip_names from struct_all.txt", len(clip_names))

    # Cross with available frame directories
    available_video_ids = set(os.listdir(frames_dir))
    log.info("Found %d video_id directories in frames/", len(available_video_ids))

    # Filter by per-clip annotation frame count
    usable = []
    skipped_no_frames_dir = 0
    skipped_no_annotation = 0
    skipped_too_few = 0
    for clip_name in clip_names:
        m = re.match(r"^(.+)_\d+$", clip_name)
        assert m is not None, f"clip_name does not match expected pattern: {clip_name}"
        video_id = m.group(1)

        if video_id not in available_video_ids:
            skipped_no_frames_dir += 1
            continue

        frames_json_path = os.path.join(annotations_dir, clip_name, "frames.json")
        if not os.path.exists(frames_json_path):
            skipped_no_annotation += 1
            continue

        with open(frames_json_path, "r") as f:
            frames_json = json.load(f)
        num_annotation_frames = len(frames_json.get("frames", []))

        if num_annotation_frames < min_frames:
            skipped_too_few += 1
            continue

        usable.append(clip_name)

    log.info("Found %d usable scenes (>=%d annotation frames). "
             "Skipped: %d no frames dir, %d no annotation, %d too few frames.",
             len(usable), min_frames,
             skipped_no_frames_dir, skipped_no_annotation, skipped_too_few)
    return sorted(usable)


# ---------------------------------------------------------------------------
# Center-crop + resize transform
# ---------------------------------------------------------------------------
def center_crop_resize(img: torch.Tensor, target_h: int,
                       target_w: int) -> torch.Tensor:
    """Center-crop and resize a [C, H, W] or [H, W] tensor to target size."""
    if img.ndim == 2:
        h, w = img.shape
        unsqueeze = True
        img = img.unsqueeze(0)
    else:
        assert img.ndim == 3
        _, h, w = img.shape
        unsqueeze = False

    scale = max(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    assert new_h >= target_h and new_w >= target_w

    img_float = img.unsqueeze(0).float()
    img_resized = torch.nn.functional.interpolate(
        img_float, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    img_resized = img_resized.squeeze(0)

    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    img_cropped = img_resized[:, top:top + target_h, left:left + target_w]

    assert img_cropped.shape[-2:] == (target_h, target_w)

    if unsqueeze:
        img_cropped = img_cropped.squeeze(0)

    return img_cropped


# ---------------------------------------------------------------------------
# Depth processing
# ---------------------------------------------------------------------------
def process_depth_frame(depth: torch.Tensor, z_far: float,
                        eps: float = 1e-8) -> tuple[torch.Tensor, float, float]:
    """Convert raw linear depth to normalized disparity in [-1, 1].

    Follows MiDaS convention: per-frame min-max normalization of disparity.

    Args:
        depth: [H, W] float32 linear depth (gl_Position.w)
        z_far: far plane value
        eps: epsilon for numerical stability

    Returns:
        Tuple of:
          - [3, H, W] float32 tensor in [-1, 1], 3-channel (grayscale replicated)
          - d_min: float, minimum disparity value for this frame
          - d_max: float, maximum disparity value for this frame
    """
    valid = (depth > 0) & (depth < z_far)
    assert valid.any(), "No valid depth pixels in frame"

    disparity = torch.zeros_like(depth)
    disparity[valid] = 1.0 / depth[valid]

    # Per-frame min-max normalize
    d_min = disparity[valid].min()
    d_max = disparity[valid].max()
    disparity_01 = (disparity - d_min) / (d_max - d_min + eps)
    disparity_01[~valid] = 0.0

    # Convert to [-1, 1]
    disparity_normalized = 2.0 * disparity_01 - 1.0
    disparity_normalized[~valid] = 0.0

    assert disparity_normalized.min() >= -1.0 - eps
    assert disparity_normalized.max() <= 1.0 + eps

    depth_rgb = disparity_normalized.unsqueeze(0).expand(3, -1, -1).contiguous()
    return depth_rgb, float(d_min), float(d_max)


# ---------------------------------------------------------------------------
# Save RGB video with torchvision
# ---------------------------------------------------------------------------
def save_rgb_video(frames: torch.Tensor, path: str, fps: int = 16):
    """Save uint8 [N, C, H, W] frames as mp4 video."""
    assert frames.ndim == 4 and frames.shape[1] == 3
    assert frames.dtype == torch.uint8
    frames_hwc = frames.permute(0, 2, 3, 1).contiguous()
    torchvision.io.write_video(path, frames_hwc, fps=fps)


# ---------------------------------------------------------------------------
# Process a single scene
# ---------------------------------------------------------------------------
def process_scene(clip_name: str, args: argparse.Namespace) -> dict | None:
    """Process one scene: load data, render depth, save outputs.

    Returns metadata dict on success, None on failure.
    """
    # Lazy imports so that forked workers re-import properly
    cad_estate_src = os.environ.get("CAD_ESTATE_SRC", args.cad_estate_src)
    if cad_estate_src not in sys.path:
        sys.path.insert(0, cad_estate_src)

    from cad_estate.room_structure import load_room_structure
    from cad_estate.frames import load_metadata, load_images, filter as filter_frames
    from cad_estate.gl.scene_renderer import render_scene

    cad_estate_root = args.cad_estate_root
    assert os.path.isdir(cad_estate_root), f"cad_estate_root not found: {cad_estate_root}"
    annotations_dir = os.path.join(cad_estate_root, "data", "annotations")
    frames_dir = os.path.join(cad_estate_root, "data", "frames")
    assert os.path.isdir(annotations_dir), f"annotations dir not found: {annotations_dir}"
    assert os.path.isdir(frames_dir), f"frames dir not found: {frames_dir}"

    num_frames = args.num_frames
    stride = args.stride
    z_far = args.z_far
    output_dir = args.output_dir

    video_id = re.match(r"^(.+)_\d+$", clip_name).group(1)
    annotation_dir = os.path.join(annotations_dir, clip_name)

    scene_output_dir = os.path.join(output_dir, video_id)

    # Resume: skip if all output files already exist
    expected_files = [
        f"{clip_name}_depth.pt",
        f"{clip_name}_raw_depth.pt",
        f"{clip_name}_depth_meta.json",
        f"{clip_name}_rgb.mp4",
        f"{clip_name}_ref.jpg",
    ]
    if all(
        os.path.exists(os.path.join(scene_output_dir, f))
        and os.path.getsize(os.path.join(scene_output_dir, f)) > 0
        for f in expected_files
    ):
        meta_path = os.path.join(scene_output_dir, f"{clip_name}_depth_meta.json")
        with open(meta_path, "r") as f:
            depth_meta = json.load(f)
        return {
            "clip_name": clip_name,
            "video_id": video_id,
            "num_frames": num_frames,
            "stride": depth_meta.get("stride", stride),
            "source_start_idx": depth_meta.get("source_start_idx"),
            "source_resolution": depth_meta.get("source_resolution"),
            "target_resolution": [TARGET_H, TARGET_W],
            "depth_path": os.path.join(video_id, f"{clip_name}_depth.pt"),
            "rgb_path": os.path.join(video_id, f"{clip_name}_rgb.mp4"),
            "ref_path": os.path.join(video_id, f"{clip_name}_ref.jpg"),
            "ref_frame_idx": depth_meta.get("ref_frame_idx"),
            "skipped": True,
        }

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load room structure
        struct_path = os.path.join(annotation_dir, "room_structure.npz")
        assert os.path.exists(struct_path), f"Missing: {struct_path}"
        with open(struct_path, "rb") as f:
            struct_bytes = f.read()
        struct_npz = np.load(io.BytesIO(struct_bytes))
        room = load_room_structure(struct_npz)

        assert room.triangles.ndim == 3
        assert room.triangles.shape[1:] == (3, 3)
        assert room.triangles.dtype == torch.float32
        assert room.triangles.shape[0] > 0

        # Load frame metadata
        frames_json_path = os.path.join(annotation_dir, "frames.json")
        assert os.path.exists(frames_json_path)
        with open(frames_json_path, "r") as f:
            frames_json = json.load(f)

        json_h, json_w = frames_json["image_size"]
        assert json_h > 0 and json_w > 0

        frames = load_metadata(frames_json, z_near=0.1, z_far=z_far)

        total_frames = len(frames.frame_timestamps)
        source_window = num_frames * stride
        assert total_frames >= source_window

        # Sample frames
        max_start = total_frames - source_window
        start_idx = random.randint(0, max_start)
        indices = torch.arange(start_idx, start_idx + source_window, stride,
                               dtype=torch.int64)
        assert indices.shape == (num_frames,)

        frames = filter_frames(frames, indices)

        # Load RGB images
        frames = asyncio.run(load_images(frames, frames_dir))
        assert frames.frame_images is not None

        src_h, src_w = frames.frame_images.shape[2], frames.frame_images.shape[3]
        if src_h != json_h or src_w != json_w:
            log.warning("Scene %s: frames.json says %dx%d but actual RGB is %dx%d",
                        clip_name, json_w, json_h, src_w, src_h)

        # Render depth for each frame
        depth_frames = []
        raw_depth_frames = []
        depth_d_mins = []
        depth_d_maxs = []
        rgb_frames = []
        for i in range(num_frames):
            view_proj = frames.camera_intrinsics[i] @ frames.camera_extrinsics[i]
            assert torch.isfinite(view_proj).all(), (
                f"Non-finite values in view_proj matrix at frame {i} for {clip_name}"
            )

            result = render_scene(
                vertex_positions=room.triangles,
                view_projection_matrix=view_proj,
                image_size=(src_h, src_w),
                output_type=torch.float32,
                cull_back_facing=False,
                return_rgb=False,
                clear_color=(0, 0, 0, 0),
            )

            assert result.shape == (src_h, src_w, 4)

            depth_raw = result[:, :, 3]

            depth_cropped = center_crop_resize(depth_raw.to(device), TARGET_H, TARGET_W)
            raw_depth_frames.append(depth_cropped.clone())

            depth_processed, d_min, d_max = process_depth_frame(depth_cropped, z_far=z_far)
            depth_frames.append(depth_processed)
            depth_d_mins.append(d_min)
            depth_d_maxs.append(d_max)

            rgb_frame = frames.frame_images[i].float().to(device)
            rgb_cropped = center_crop_resize(rgb_frame, TARGET_H, TARGET_W)
            rgb_cropped = rgb_cropped.clamp(0, 255).to(torch.uint8)
            rgb_frames.append(rgb_cropped)

        # Stack tensors and validate
        depth_tensor = torch.stack(depth_frames, dim=1).cpu()
        assert depth_tensor.shape == (3, num_frames, TARGET_H, TARGET_W)
        from depth2room.utils import validate_depth_tensor
        validate_depth_tensor(depth_tensor, label=f"output depth for {clip_name}")

        rgb_tensor = torch.stack(rgb_frames, dim=0).cpu()
        assert rgb_tensor.shape == (num_frames, 3, TARGET_H, TARGET_W)

        raw_depth_tensor = torch.stack(raw_depth_frames, dim=0).unsqueeze(0).cpu()
        assert raw_depth_tensor.shape == (1, num_frames, TARGET_H, TARGET_W)

        # Save outputs
        os.makedirs(scene_output_dir, exist_ok=True)

        depth_path = os.path.join(scene_output_dir, f"{clip_name}_depth.pt")
        torch.save(depth_tensor, depth_path)

        raw_depth_path = os.path.join(scene_output_dir, f"{clip_name}_raw_depth.pt")
        torch.save(raw_depth_tensor, raw_depth_path)

        video_path = os.path.join(scene_output_dir, f"{clip_name}_rgb.mp4")
        save_rgb_video(rgb_tensor, video_path, fps=int(30 / stride))

        # Save reference image from a random frame
        ref_frame_idx = random.randint(0, num_frames - 1)
        ref_path = os.path.join(scene_output_dir, f"{clip_name}_ref.jpg")
        ref_img = rgb_tensor[ref_frame_idx]
        ref_pil = torchvision.transforms.functional.to_pil_image(ref_img)
        ref_pil.save(ref_path, quality=95)

        depth_meta = {
            "d_min": depth_d_mins,
            "d_max": depth_d_maxs,
            "z_far": z_far,
            "stride": stride,
            "source_start_idx": int(start_idx),
            "source_resolution": [src_h, src_w],
            "ref_frame_idx": ref_frame_idx,
            "source_fps": 30,
            "effective_fps": 30 / stride,
        }
        depth_meta_path = os.path.join(scene_output_dir, f"{clip_name}_depth_meta.json")
        with open(depth_meta_path, "w") as f:
            json.dump(depth_meta, f)

        metadata = {
            "clip_name": clip_name,
            "video_id": video_id,
            "num_frames": num_frames,
            "stride": stride,
            "source_start_idx": int(start_idx),
            "source_resolution": [src_h, src_w],
            "target_resolution": [TARGET_H, TARGET_W],
            "depth_path": os.path.join(video_id, os.path.basename(depth_path)),
            "rgb_path": os.path.join(video_id, os.path.basename(video_path)),
            "ref_path": os.path.join(video_id, os.path.basename(ref_path)),
            "ref_frame_idx": ref_frame_idx,
        }

        log.info("Processed scene %s (frames %d-%d stride %d, %dx%d -> %dx%d)",
                 clip_name, start_idx, start_idx + source_window - 1, stride,
                 src_w, src_h, TARGET_W, TARGET_H)
        return metadata

    except Exception as e:
        log.error("Failed to process scene %s: %s\n%s",
                  clip_name, e, traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Worker wrapper for multiprocessing
# ---------------------------------------------------------------------------
def _worker(args_tuple):
    """Unpack arguments for process_scene (ProcessPoolExecutor compatibility)."""
    clip_name, args, worker_gpu = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_gpu)
    torch.set_num_threads(1)
    return process_scene(clip_name, args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare CAD-Estate data for VACE depth-to-RGB training"
    )
    parser.add_argument(
        "--cad_estate_root", type=str,
        default=os.environ.get("CAD_ESTATE_ROOT"),
        help="Root of the cad-estate repository clone"
    )
    parser.add_argument(
        "--cad_estate_src", type=str,
        default=None,
        help="Path to cad-estate/src (default: <cad_estate_root>/src)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write output files"
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--z_far", type=float, default=200.0)
    parser.add_argument("--min_source_frames", type=int, default=None)
    args = parser.parse_args()

    if args.cad_estate_src is None:
        args.cad_estate_src = os.path.join(args.cad_estate_root, "src")

    if args.min_source_frames is None:
        args.min_source_frames = args.num_frames * args.stride

    os.makedirs(args.output_dir, exist_ok=True)

    struct_all_path = os.path.join(args.cad_estate_root, "data", "struct_all.txt")
    frames_dir = os.path.join(args.cad_estate_root, "data", "frames")
    annotations_dir = os.path.join(args.cad_estate_root, "data", "annotations")

    scenes = discover_scenes(struct_all_path, frames_dir, annotations_dir,
                             min_frames=args.min_source_frames)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    log.info("Will process %d scenes", len(scenes))

    metadata_list = []
    failed = 0
    skipped = 0

    if args.num_workers <= 1:
        for i, clip_name in enumerate(scenes):
            log.info("[%d/%d] Processing %s", i + 1, len(scenes), clip_name)
            result = process_scene(clip_name, args)
            if result is not None:
                if result.get("skipped"):
                    skipped += 1
                metadata_list.append(result)
            else:
                failed += 1
    else:
        num_gpus = max(1, torch.cuda.device_count())
        work_items = [(clip_name, args, i % num_gpus) for i, clip_name in enumerate(scenes)]
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(_worker, item): item[0] for item in work_items}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                if result is not None:
                    if result.get("skipped"):
                        skipped += 1
                    metadata_list.append(result)
                else:
                    failed += 1
                if done_count % 50 == 0 or done_count == len(scenes):
                    log.info("Progress: %d/%d scenes (%d skipped, %d failed)",
                             done_count, len(scenes), skipped, failed)

    assert len(metadata_list) > 0, (
        f"No scenes were successfully processed out of {len(scenes)} attempted"
    )

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f, indent=2)

    log.info("Done. %d scenes total (%d new, %d skipped, %d failed).",
             len(metadata_list), len(metadata_list) - skipped, skipped, failed)
    log.info("Metadata saved to %s", metadata_path)
    if failed > 0:
        log.warning("%.1f%% of scenes failed processing (%d/%d).",
                     100 * failed / len(scenes), failed, len(scenes))


if __name__ == "__main__":
    main()
