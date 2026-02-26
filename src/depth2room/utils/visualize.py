#!/usr/bin/env python3
"""Visualize depth maps: side-by-side RGB | Disparity | Linear Depth video.

Usage:
    python -m depth2room.utils.visualize \
        --data_dir /path/to/vace_training_dataset \
        --clip CLIP_NAME
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
from PIL import Image


def turbo_colormap(x):
    """Apply turbo colormap to [0,1] float array, returns [H,W,3] uint8."""
    colored = cm.turbo(x)[:, :, :3]
    return (colored * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize depth maps.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the prepared dataset.")
    parser.add_argument("--clip", type=str, required=True,
                        help="Clip name (e.g. video_id/clip_name or just clip_name).")
    args = parser.parse_args()

    out_dir = args.data_dir
    clip = args.clip

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


if __name__ == "__main__":
    main()
