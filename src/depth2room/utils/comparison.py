#!/usr/bin/env python3
"""
Create side-by-side comparison videos: depth | generated | ground_truth.

Usage:
    python -m depth2room.utils.comparison \
        --eval_dir eval_output/step-200 \
        --data_dir /path/to/vace_training_dataset
"""

import argparse
import json
import os

import imageio
import torch
from PIL import Image, ImageDraw, ImageFont
from diffsynth.utils.data import save_video


def load_metadata(data_dir):
    """Load metadata.json and index by clip_name."""
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    return {e["clip_name"]: e for e in metadata}


def depth_tensor_to_frames(depth_tensor):
    """Convert depth tensor [3, T, H, W] in [-1,1] to list of PIL Images."""
    frames = ((depth_tensor + 1) / 2 * 255).clamp(0, 255).byte()
    num_frames = frames.shape[1]
    images = []
    for t in range(num_frames):
        frame = frames[:, t]
        img = Image.fromarray(frame.permute(1, 2, 0).numpy())
        images.append(img)
    return images


def add_label(img, text, font_size=20):
    """Add a text label at the top of an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    y = 8
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            draw.text((x + dx, y + dy), text, fill="black", font=font)
    draw.text((x, y), text, fill="white", font=font)
    return img


def make_side_by_side(eval_dir, scene_name, data_dir, metadata):
    """Create side-by-side video for a single scene."""
    scene_dir = os.path.join(eval_dir, scene_name)

    info_path = os.path.join(scene_dir, "info.json")
    with open(info_path) as f:
        info = json.load(f)
    clip_name = info["clip_name"]

    meta_entry = metadata.get(clip_name)
    if meta_entry is None:
        print(f"  Clip not found in metadata: {clip_name}")
        return None
    depth_path = os.path.join(data_dir, meta_entry["depth_path"])
    if not os.path.exists(depth_path):
        print(f"  Depth tensor not found: {depth_path}")
        return None

    print(f"  Loading depth tensor...")
    depth_tensor = torch.load(depth_path, map_location="cpu", weights_only=True)
    depth_frames = depth_tensor_to_frames(depth_tensor)

    depth_video_path = os.path.join(scene_dir, "depth.mp4")
    save_video(depth_frames, depth_video_path, fps=16, quality=5)

    generated_path = os.path.join(scene_dir, "generated.mp4")
    reader = imageio.get_reader(generated_path)
    gen_frames = [Image.fromarray(frame) for frame in reader]
    reader.close()

    gt_path = os.path.join(scene_dir, "ground_truth.mp4")
    gt_frames = None
    if os.path.exists(gt_path):
        reader = imageio.get_reader(gt_path)
        gt_frames = [Image.fromarray(frame) for frame in reader]
        reader.close()

    n_frames = min(len(depth_frames), len(gen_frames))
    if gt_frames is not None:
        n_frames = min(n_frames, len(gt_frames))

    print(f"  Compositing {n_frames} frames...")
    comparison_frames = []
    for t in range(n_frames):
        panels = [depth_frames[t].copy(), gen_frames[t].copy()]
        labels = ["Depth", "Generated"]

        if gt_frames is not None:
            panels.append(gt_frames[t].copy())
            labels.append("Ground Truth")

        target_h = panels[0].height
        resized = []
        for panel in panels:
            if panel.height != target_h:
                scale = target_h / panel.height
                panel = panel.resize((int(panel.width * scale), target_h), Image.LANCZOS)
            resized.append(panel)

        for i, (panel, label) in enumerate(zip(resized, labels)):
            add_label(panel, label)

        total_w = sum(p.width for p in resized)
        composite = Image.new("RGB", (total_w, target_h))
        x_offset = 0
        for panel in resized:
            composite.paste(panel, (x_offset, 0))
            x_offset += panel.width

        comparison_frames.append(composite)

    comparison_path = os.path.join(scene_dir, "comparison.mp4")
    save_video(comparison_frames, comparison_path, fps=16, quality=5)
    print(f"  Saved: {comparison_path}")
    return comparison_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = [
            d for d in os.listdir(args.eval_dir)
            if os.path.isdir(os.path.join(args.eval_dir, d))
        ]

    metadata = load_metadata(args.data_dir)
    print(f"Creating comparisons for {len(scenes)} scene(s)...")
    for scene in sorted(scenes):
        print(f"\n[{scene}]")
        make_side_by_side(args.eval_dir, scene, args.data_dir, metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()
