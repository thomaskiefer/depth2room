#!/usr/bin/env python3
"""Visualize depth maps: side-by-side video or dataset grid overview.

Usage (video mode):
    python -m depth2room.utils.visualize \
        --data_dir /path/to/vace_training_dataset \
        --mode video --clip CLIP_NAME

Usage (grid mode):
    python -m depth2room.utils.visualize \
        --data_dir /path/to/vace_training_dataset \
        --mode grid --num_scenes 6
"""

import argparse
import json
import os

import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def turbo_colormap(x):
    """Apply turbo colormap to [0,1] float array, returns [H,W,3] uint8.

    Dithers in input space (before colormap lookup) so that adjacent depth
    values straddle colormap bin boundaries, breaking visible banding.
    """
    # Dither in input space: ±0.5 colormap bins (256 bins over [0,1])
    noise = np.random.triangular(-1.0, 0, 1.0, size=x.shape) / 256.0
    x_dithered = np.clip(x + noise, 0, 1)
    colored = cm.turbo(x_dithered)[:, :, :3]
    return (colored * 255).clip(0, 255).astype(np.uint8)


def _load_rgb_frames(path, indices):
    """Load specific frame indices from an mp4 file."""
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return [frames[i] for i in indices if i < len(frames)]


def visualize_grid(data_dir, num_scenes=6):
    """Create a PNG grid per scene: RGB / Depth (turbo) / Validity / Reference."""
    meta_path = os.path.join(data_dir, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    total = len(metadata)
    step = max(1, total // num_scenes)
    scene_indices = list(range(0, total, step))[:num_scenes]

    viz_dir = os.path.join(data_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    frame_indices = [0, 20, 40, 60, 80]  # 5 evenly spaced from 81
    n_cols = len(frame_indices)
    label_h = 28  # height for row labels

    print(f"Generating grids for {len(scene_indices)} scenes out of {total}...")

    for si, scene_idx in enumerate(scene_indices):
        entry = metadata[scene_idx]
        clip = entry["clip_name"]

        print(f"  [{si+1}/{len(scene_indices)}] {clip}")

        # --- Load RGB frames ---
        rgb_path = os.path.join(data_dir, entry["rgb_path"])
        rgb_frames = _load_rgb_frames(rgb_path, frame_indices)
        if not rgb_frames:
            print(f"    Skipping: no RGB frames loaded")
            continue
        frame_h, frame_w = rgb_frames[0].shape[:2]

        # --- Load depth [3, 81, H, W] ---
        depth_path = os.path.join(data_dir, entry["depth_path"])
        depth = torch.load(depth_path, map_location="cpu", weights_only=True)
        # Use disparity channel (index 0), map from [-1,1] to [0,1]
        depth_frames = []
        for fi in frame_indices:
            disp = depth[0, fi].numpy()
            disp_01 = (disp + 1.0) / 2.0
            depth_frames.append(turbo_colormap(disp_01))

        # --- Load validity [1, 81, H, W] ---
        validity_path = os.path.join(data_dir, entry["validity_path"])
        validity = torch.load(validity_path, map_location="cpu", weights_only=True)
        validity_frames = []
        for fi in frame_indices:
            mask = validity[0, fi].numpy()
            # white=valid (1), black=invalid (0)
            mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
            validity_frames.append(mask_rgb)

        # --- Load reference image ---
        ref_path = os.path.join(data_dir, entry["ref_path"])
        ref_img = Image.open(ref_path).convert("RGB")
        ref_img = ref_img.resize((frame_w, frame_h), Image.BILINEAR)

        # --- Compose grid ---
        n_rows = 4
        grid_w = n_cols * frame_w
        grid_h = n_rows * (label_h + frame_h)
        canvas = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()

        row_labels = [
            f"RGB  —  Scene {scene_idx}: {clip}",
            "Depth (turbo)",
            "Validity mask",
            "Reference",
        ]
        row_data = [rgb_frames, depth_frames, validity_frames, None]

        for row_idx in range(n_rows):
            y_label = row_idx * (label_h + frame_h)
            y_img = y_label + label_h

            # Draw row label
            color = (100, 255, 100) if row_idx == 0 else (255, 255, 255)
            draw.text((8, y_label + 4), row_labels[row_idx], fill=color, font=font)

            if row_idx < 3:
                # Regular row: 5 frames
                for col_idx in range(n_cols):
                    img = Image.fromarray(row_data[row_idx][col_idx])
                    if img.size != (frame_w, frame_h):
                        img = img.resize((frame_w, frame_h), Image.BILINEAR)
                    canvas.paste(img, (col_idx * frame_w, y_img))
            else:
                # Reference row: single image centered
                x_center = (grid_w - frame_w) // 2
                canvas.paste(ref_img, (x_center, y_img))
                # Add ref_frame_idx annotation
                ref_label = f"ref_frame_idx={entry.get('ref_frame_idx', '?')}"
                draw.text((x_center + 8, y_img + 4), ref_label, fill=(255, 255, 0), font=font)

        out_path = os.path.join(viz_dir, f"scene_{scene_idx:04d}_{clip}.png")
        canvas.save(out_path)
        print(f"    Saved: {out_path}")

    print(f"\nDone. {len(scene_indices)} grids saved to {viz_dir}/")


def _make_title_card(text, width, height, font):
    """Create a title card frame as uint8 numpy array [H, W, 3]."""
    canvas = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(canvas)


def visualize_sidebyside(data_dir, num_scenes=6):
    """Create a single video with RGB | Depth (turbo) | Validity side-by-side for multiple scenes."""
    meta_path = os.path.join(data_dir, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    total = len(metadata)
    step = max(1, total // num_scenes)
    scene_indices = list(range(0, total, step))[:num_scenes]

    viz_dir = os.path.join(data_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Use first scene to get dimensions
    first = metadata[scene_indices[0]]
    depth_t = torch.load(os.path.join(data_dir, first["depth_path"]),
                         map_location="cpu", weights_only=True)
    frame_h, frame_w = depth_t.shape[2], depth_t.shape[3]
    del depth_t

    fps = 16
    vid_w = frame_w * 3
    vid_h = frame_h

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except (OSError, IOError):
        font = ImageFont.load_default()

    out_path = os.path.join(viz_dir, "dataset_sidebyside.mp4")
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = vid_w
    stream.height = vid_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}

    print(f"Generating side-by-side video for {len(scene_indices)} scenes...")

    for si, scene_idx in enumerate(scene_indices):
        entry = metadata[scene_idx]
        clip = entry["clip_name"]
        n_frames = entry["num_frames"]

        print(f"  [{si+1}/{len(scene_indices)}] {clip} ({n_frames} frames)")

        # Title card (~1 second)
        title = f"Scene {scene_idx}: {clip}"
        title_frame = _make_title_card(title, vid_w, vid_h, font)
        for _ in range(fps):
            vf = av.VideoFrame.from_ndarray(title_frame, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)

        # Load data
        rgb_path = os.path.join(data_dir, entry["rgb_path"])
        rgb_frames = _load_rgb_frames(rgb_path, list(range(n_frames)))

        depth_path = os.path.join(data_dir, entry["depth_path"])
        depth = torch.load(depth_path, map_location="cpu", weights_only=True)

        validity_path = os.path.join(data_dir, entry["validity_path"])
        validity = torch.load(validity_path, map_location="cpu", weights_only=True)

        actual_frames = min(n_frames, len(rgb_frames), depth.shape[1], validity.shape[1])

        for i in range(actual_frames):
            rgb = rgb_frames[i]
            if rgb.shape[0] != frame_h or rgb.shape[1] != frame_w:
                rgb = np.array(Image.fromarray(rgb).resize((frame_w, frame_h), Image.BILINEAR))

            disp = depth[0, i].numpy()
            disp_01 = (disp + 1.0) / 2.0
            depth_rgb = turbo_colormap(disp_01)

            mask = validity[0, i].numpy()
            mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

            combined = np.concatenate([rgb, depth_rgb, mask_rgb], axis=1)
            vf = av.VideoFrame.from_ndarray(combined, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)

        del rgb_frames, depth, validity

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Saved: {out_path}")


def visualize_video(data_dir, clip):
    """Original video visualization: side-by-side RGB | Disparity | Linear Depth."""
    out_dir = data_dir

    depth_path = f"{out_dir}/{clip}_depth.pt"
    raw_depth_path = f"{out_dir}/{clip}_raw_depth.pt"
    meta_path = f"{out_dir}/{clip}_depth_meta.json"
    rgb_path = f"{out_dir}/{clip}_rgb.mp4"

    for path in [depth_path, raw_depth_path, meta_path, rgb_path]:
        assert os.path.exists(path), f"File not found: {path}"

    depth = torch.load(depth_path, map_location="cpu", weights_only=True)
    raw_depth = torch.load(raw_depth_path, map_location="cpu", weights_only=True)
    with open(meta_path) as f:
        meta = json.load(f)

    assert depth.ndim == 4 and depth.shape[0] == 3, (
        f"Expected depth tensor [3, T, H, W], got {depth.shape}"
    )
    assert raw_depth.ndim == 4 and raw_depth.shape[0] == 1, (
        f"Expected raw_depth tensor [1, T, H, W], got {raw_depth.shape}"
    )

    print(f"Depth shape: {depth.shape}, range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"Raw depth shape: {raw_depth.shape}, range: [{raw_depth.min():.4f}, {raw_depth.max():.4f}]")
    print(f"z_far: {meta['z_far']}")

    rgb_container = av.open(f"{out_dir}/{clip}_rgb.mp4")
    rgb_frames = []
    for frame in rgb_container.decode(video=0):
        rgb_frames.append(frame.to_ndarray(format="rgb24"))
    rgb_container.close()

    n_frames = min(depth.shape[1], len(rgb_frames))
    print(f"Frames: {n_frames}")

    raw_valid = raw_depth[raw_depth > 0]
    if raw_valid.numel() == 0:
        print("Warning: all raw depth values are zero, using z_far as global_max")
        global_max = meta["z_far"]
    else:
        global_max = min(float(np.percentile(raw_valid.numpy(), 98)), meta["z_far"])

    # Derive dimensions from actual tensor shape
    frame_h, frame_w = depth.shape[2], depth.shape[3]

    # Side-by-side video
    out_path = f"{out_dir}/{clip}_sidebyside.mp4"
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=16)
    stream.width = frame_w * 3
    stream.height = frame_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}

    for i in range(n_frames):
        rgb = rgb_frames[i]
        if rgb.shape[0] != frame_h or rgb.shape[1] != frame_w:
            rgb = np.array(Image.fromarray(rgb).resize((frame_w, frame_h), Image.BILINEAR))

        disp = depth[0, i].numpy()
        disp_01 = (disp + 1.0) / 2.0
        disp_rgb = turbo_colormap(disp_01)

        raw = raw_depth[0, i].numpy()
        raw_01 = np.clip(raw / global_max, 0, 1)
        valid_mask = raw > 0
        raw_mapped = np.where(valid_mask, raw_01, 0.0)
        raw_rgb = turbo_colormap(raw_mapped)

        combined = np.concatenate([rgb, disp_rgb, raw_rgb], axis=1)
        frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Saved: {out_path}")

    # Disparity-only turbo video
    out_path2 = f"{out_dir}/{clip}_depth_turbo.mp4"
    container = av.open(out_path2, mode="w")
    stream = container.add_stream("libx264", rate=16)
    stream.width = frame_w
    stream.height = frame_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}

    for i in range(n_frames):
        disp = depth[0, i].numpy()
        disp_01 = (disp + 1.0) / 2.0
        disp_rgb = turbo_colormap(disp_01)
        frame = av.VideoFrame.from_ndarray(disp_rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Saved: {out_path2}")


def main():
    parser = argparse.ArgumentParser(description="Visualize depth maps.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the prepared dataset.")
    parser.add_argument("--mode", type=str, default="video",
                        choices=["video", "grid", "sidebyside"],
                        help="Visualization mode: 'video' (per-clip), 'grid' (PNG overview), "
                             "or 'sidebyside' (RGB|Depth|Validity video).")
    parser.add_argument("--clip", type=str, default=None,
                        help="Clip name for video mode.")
    parser.add_argument("--num_scenes", type=int, default=6,
                        help="Number of scenes for grid mode.")
    args = parser.parse_args()

    if args.mode == "grid":
        visualize_grid(args.data_dir, num_scenes=args.num_scenes)
    elif args.mode == "sidebyside":
        visualize_sidebyside(args.data_dir, num_scenes=args.num_scenes)
    elif args.mode == "video":
        if args.clip is None:
            parser.error("--clip is required for video mode")
        visualize_video(args.data_dir, args.clip)


if __name__ == "__main__":
    main()
