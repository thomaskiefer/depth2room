#!/usr/bin/env python3
"""
Run VACE depth-conditioned video generation from raw Blender EXR depth files.

Treats all arcs in a scene as one continuous video sequence.
Gathers EXR frames across arcs (sorted by arc name, then frame number),
picks a contiguous subset, converts depth to normalized disparity,
and runs the VACE pipeline.

Usage:
    python -m depth2room.inference.infer_exr \
        --model_dir /path/to/Wan2.1-VACE-1.3B \
        --scene_dir /path/to/scene_mist \
        --checkpoint /path/to/step-200.safetensors
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffsynth.utils.data import save_video
from depth2room.inference import load_pipeline
from depth2room.utils import validate_depth_tensor


def load_exr_depth_z(exr_path: str) -> np.ndarray:
    """Load ViewLayer.Depth.Z from a Blender EXR file."""
    import OpenEXR
    f = OpenEXR.File(exr_path)
    channels = f.channels()
    if "ViewLayer.Depth.Z" in channels:
        depth = channels["ViewLayer.Depth.Z"].pixels
    elif "Depth.Z" in channels:
        depth = channels["Depth.Z"].pixels
    else:
        raise ValueError(f"No depth channel found in {exr_path}. Available: {list(channels.keys())}")
    return depth.astype(np.float32)


def load_exr_mist(exr_path: str) -> np.ndarray:
    """Load ViewLayer.Mist.Z from a Blender EXR file."""
    import OpenEXR
    f = OpenEXR.File(exr_path)
    channels = f.channels()
    if "ViewLayer.Mist.Z" in channels:
        mist = channels["ViewLayer.Mist.Z"].pixels
    elif "Mist.Z" in channels:
        mist = channels["Mist.Z"].pixels
    else:
        raise ValueError(f"No mist channel in {exr_path}. Available: {list(channels.keys())}")
    return mist.astype(np.float32)


def load_exr_rgb(exr_path: str) -> np.ndarray:
    """Load composited RGB from a Blender EXR file. Returns [H, W, 3] float32 in [0, 1]."""
    import OpenEXR
    f = OpenEXR.File(exr_path)
    channels = f.channels()
    for prefix in ["ViewLayer.Combined", "Composite.Combined"]:
        if prefix in channels:
            rgba = channels[prefix].pixels
            if rgba.ndim == 3 and rgba.shape[2] >= 3:
                return np.clip(rgba[:, :, :3], 0, 1).astype(np.float32)
    raise ValueError(f"No RGB channel found in {exr_path}")


def depth_to_disparity_raw(depth: np.ndarray, z_far: float = 200.0,
                           min_filter_size: int = 6, eps: float = 1e-8) -> np.ndarray:
    """Convert Z-depth to disparity with minimum filter for edge artifact removal."""
    from scipy.ndimage import minimum_filter

    valid = (depth > eps) & (depth < z_far) & np.isfinite(depth)
    assert valid.any(), "No valid depth pixels in frame"

    disparity = np.zeros_like(depth)
    disparity[valid] = 1.0 / depth[valid]
    disparity = minimum_filter(disparity, size=min_filter_size)
    return disparity.astype(np.float32)


def compute_global_range(exr_paths: list[str], z_far: float = 200.0,
                         min_filter_size: int = 6) -> tuple[float, float]:
    """Compute global disparity min/max across all frames."""
    global_min = float("inf")
    global_max = float("-inf")

    for exr_path in exr_paths:
        depth = load_exr_depth_z(exr_path)
        disp = depth_to_disparity_raw(depth, z_far=z_far, min_filter_size=min_filter_size)
        valid = disp > 0
        if valid.any():
            global_min = min(global_min, float(disp[valid].min()))
            global_max = max(global_max, float(disp[valid].max()))

    return global_min, global_max


def normalize_disparity(disparity: np.ndarray, d_min: float, d_max: float,
                        eps: float = 1e-8) -> np.ndarray:
    """Normalize disparity to [-1, 1] using global min/max."""
    valid = disparity > 0
    disp_01 = np.zeros_like(disparity)
    disp_01[valid] = (disparity[valid] - d_min) / (d_max - d_min + eps)
    disp_01 = np.clip(disp_01, 0, 1)

    normalized = 2.0 * disp_01 - 1.0
    normalized[~valid] = 0.0
    return normalized.astype(np.float32)


def snap_frame_count(n: int) -> int:
    """Snap frame count down to nearest valid value: 4*k + 1."""
    k = max(1, (n - 1) // 4)
    return 4 * k + 1


def arc_sort_key(name: str):
    """Sort arc names numerically."""
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else name


def gather_all_exr_frames(scene_dir: str) -> list[str]:
    """Gather all EXR frame paths across arcs as one continuous sequence."""
    scene_path = Path(scene_dir)
    arc_dirs = sorted(
        [d for d in scene_path.iterdir() if d.is_dir() and (d / "exr").exists()],
        key=lambda d: arc_sort_key(d.name),
    )

    all_frames = []
    for arc_dir in arc_dirs:
        exr_dir = arc_dir / "exr"
        frames = sorted(exr_dir.glob("*.exr"))
        all_frames.extend([str(f) for f in frames])
    return all_frames


def build_depth_tensor(exr_paths: list[str], height: int = 480, width: int = 832,
                       z_far: float = 200.0, min_filter_size: int = 6,
                       use_mist: bool = False, per_frame: bool = False,
                       percentile: float = 1.0) -> torch.Tensor:
    """Load EXR depth frames, process, normalize, build tensor.

    Returns [3, T, height, width] float32 tensor in [-1, 1].
    """
    source = "Mist" if use_mist else "Z-depth"
    norm = "per-frame" if per_frame else "global"
    print(f"  Loading {source}, {norm} normalization, {percentile}th percentile filtering" +
          ("" if use_mist else f", min_filter={min_filter_size}") + "...")

    raw_disparities = []
    for exr_path in exr_paths:
        if use_mist:
            dist = load_exr_mist(exr_path)
            valid = (dist > 1e-8) & (dist < z_far) & np.isfinite(dist)
            disp = np.zeros_like(dist)
            disp[valid] = 1.0 / dist[valid]
        else:
            depth = load_exr_depth_z(exr_path)
            disp = depth_to_disparity_raw(depth, z_far=z_far, min_filter_size=min_filter_size)
        raw_disparities.append(disp)

    if per_frame:
        frames = []
        for disp in raw_disparities:
            valid = disp > 0
            if valid.any():
                valid_vals = disp[valid]
                d_min = float(np.percentile(valid_vals, percentile))
                d_max = float(np.percentile(valid_vals, 100.0 - percentile))
            else:
                d_min, d_max = 0.0, 1.0
            normalized = normalize_disparity(disp, d_min, d_max)
            frames.append(normalized)
    else:
        all_valid = np.concatenate([d[d > 0] for d in raw_disparities])
        raw_min, raw_max = float(all_valid.min()), float(all_valid.max())
        d_min = float(np.percentile(all_valid, percentile))
        d_max = float(np.percentile(all_valid, 100.0 - percentile))
        print(f"  Global disparity range (p{percentile}-p{100-percentile}): [{d_min:.6f}, {d_max:.6f}]")
        print(f"  (Raw min/max: [{raw_min:.6f}, {raw_max:.6f}])")
        frames = [normalize_disparity(disp, d_min, d_max) for disp in raw_disparities]

    depth_stack = torch.from_numpy(np.stack(frames, axis=0)).float()
    depth_stack = depth_stack.unsqueeze(1)
    depth_stack = F.interpolate(depth_stack, size=(height, width),
                                mode="bilinear", align_corners=False)
    depth_stack = depth_stack.squeeze(1)
    depth_tensor = depth_stack.unsqueeze(0).expand(3, -1, -1, -1).contiguous()
    return depth_tensor


def get_reference_image(exr_path: str, height: int = 480,
                        width: int = 832) -> Image.Image | None:
    """Extract reference image (RGB) from the first EXR frame."""
    try:
        rgb = load_exr_rgb(exr_path)
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb_uint8)
        img = img.resize((width, height), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"  Could not extract reference image: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run VACE depth-to-RGB from EXR files.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to VACE model directory.")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Scene directory containing arc subdirectories.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--prompt", type=str,
                        default="A realistic furnished residential apartment, indoor overhead "
                                "lighting and daylight from rooms, white and light gray walls, "
                                "mixed furniture styles, lived-in and comfortable atmosphere. "
                                "Rooms contain everyday items like bookshelves, sofas, tables, "
                                "kitchen appliances and personal belongings. Bright and evenly "
                                "lit interior, natural colors, casual home photography style, "
                                "photorealistic.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--z_far", type=float, default=200.0)
    parser.add_argument("--no_ref", action="store_true")
    parser.add_argument("--use_mist", action="store_true")
    parser.add_argument("--per_frame", action="store_true")
    parser.add_argument("--percentile", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert os.path.isdir(args.scene_dir), f"Scene directory not found: {args.scene_dir}"
    assert os.path.isdir(args.model_dir), f"Model directory not found: {args.model_dir}"
    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"

    all_frames = gather_all_exr_frames(args.scene_dir)
    assert len(all_frames) > 0, f"No EXR frames found in {args.scene_dir}"
    print(f"Total frames across all arcs: {len(all_frames)}")

    assert args.start_frame < len(all_frames), (
        f"start_frame ({args.start_frame}) >= total frames ({len(all_frames)})"
    )
    end_frame = min(args.start_frame + args.num_frames, len(all_frames))
    selected = all_frames[args.start_frame:end_frame]
    actual_frames = snap_frame_count(len(selected))
    selected = selected[:actual_frames]
    print(f"Selected frames {args.start_frame}-{args.start_frame + actual_frames - 1} "
          f"({actual_frames} frames)")

    if args.output_dir is None:
        scene_name = Path(args.scene_dir).name
        ckpt_name = Path(args.checkpoint).stem if args.checkpoint else "base"
        args.output_dir = os.path.join(
            "infer_output", scene_name, ckpt_name,
            f"frames_{args.start_frame}_{args.start_frame + actual_frames - 1}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print("Building depth tensor...")
    depth_tensor = build_depth_tensor(selected, height=480, width=832, z_far=args.z_far,
                                      use_mist=args.use_mist, per_frame=args.per_frame,
                                      percentile=args.percentile)
    validate_depth_tensor(depth_tensor, label="built depth tensor")
    print(f"Depth tensor: {depth_tensor.shape}, range [{depth_tensor.min():.2f}, {depth_tensor.max():.2f}]")

    ref_image = None
    if not args.no_ref:
        ref_image = get_reference_image(selected[0], height=480, width=832)
        if ref_image:
            print(f"Reference image: {ref_image.size}")

    pipe = load_pipeline(args.model_dir, args.checkpoint, args.lora_alpha, args.device)

    print(f"\nGenerating ({args.steps} steps, cfg={args.cfg_scale}, "
          f"seed={args.seed}, frames={actual_frames})...")
    video = pipe(
        prompt=args.prompt,
        negative_prompt="blurry, low quality, distorted, ugly",
        vace_video=depth_tensor,
        vace_reference_image=ref_image,
        vace_scale=1.0,
        seed=args.seed,
        height=480,
        width=832,
        num_frames=actual_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        sigma_shift=5.0,
    )

    assert video is not None and len(video) > 0, "Pipeline returned empty output"

    gen_path = os.path.join(args.output_dir, "generated.mp4")
    save_video(video, gen_path, fps=16, quality=5)
    print(f"Saved: {gen_path}")

    import matplotlib
    turbo = matplotlib.colormaps["turbo"]
    depth_frames = []
    for t in range(depth_tensor.shape[1]):
        disp_01 = ((depth_tensor[0, t].numpy() + 1.0) / 2.0).clip(0, 1)
        colored = turbo(disp_01)[:, :, :3]
        colored_uint8 = (colored * 255).clip(0, 255).astype(np.uint8)
        depth_frames.append(Image.fromarray(colored_uint8))
    depth_path = os.path.join(args.output_dir, "depth.mp4")
    save_video(depth_frames, depth_path, fps=16, quality=5)

    if ref_image:
        ref_image.save(os.path.join(args.output_dir, "reference.jpg"))

    with open(os.path.join(args.output_dir, "info.json"), "w") as f:
        json.dump({
            "scene_dir": args.scene_dir,
            "prompt": args.prompt,
            "checkpoint": args.checkpoint,
            "num_inference_steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "z_far": args.z_far,
            "start_frame": args.start_frame,
            "num_frames": actual_frames,
            "total_frames_available": len(all_frames),
            "with_ref": ref_image is not None,
            "depth_source": "mist" if args.use_mist else "zdepth",
            "normalization": "per_frame" if args.per_frame else "global",
            "percentile": args.percentile,
            "first_exr": selected[0],
            "last_exr": selected[-1],
        }, f, indent=2)

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
