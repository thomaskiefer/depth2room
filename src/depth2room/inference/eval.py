#!/usr/bin/env python3
"""
Evaluate a VACE depth-to-RGB checkpoint by running inference on held-out scenes.

Loads the base Wan2.1-VACE pipeline, applies the LoRA/full checkpoint,
and generates RGB videos from depth tensors.

Usage:
    python -m depth2room.inference.eval \
        --model_dir /path/to/Wan2.1-VACE-1.3B \
        --checkpoint output/lora_depth_vace/epoch-0.safetensors \
        --data_dir /path/to/vace_training_dataset \
        --scenes 3 \
        --output_dir eval_output/epoch-0
"""

import argparse
import json
import os
import random
import shutil

import matplotlib
import numpy as np
import torch
from PIL import Image

from diffsynth.utils.data import save_video
from depth2room.inference import load_pipeline
from depth2room.utils import validate_depth_tensor


def compute_eval_metrics(gen_frames, gt_video_path, device="cpu"):
    """Compute LPIPS and SSIM between generated frames and ground-truth video.

    Args:
        gen_frames: List of PIL Images (generated video frames).
        gt_video_path: Path to ground-truth mp4 video.
        device: torch device for LPIPS computation.

    Returns:
        Dict with avg and per-frame LPIPS/SSIM, or None if GT unavailable.
    """
    if not os.path.exists(gt_video_path):
        return None

    try:
        import cv2
        import lpips
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        print("  Metrics skipped (install depth2room[eval] for LPIPS/SSIM)")
        return None

    # Load GT frames
    cap = cv2.VideoCapture(gt_video_path)
    gt_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gt_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    n = min(len(gen_frames), len(gt_frames))
    if n == 0:
        return None

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    lpips_vals = []
    ssim_vals = []

    for t in range(n):
        gen_np = np.array(gen_frames[t])
        gt_np = gt_frames[t]

        # Resize GT to match generated if needed
        if gt_np.shape[:2] != gen_np.shape[:2]:
            gt_np = np.array(Image.fromarray(gt_np).resize(
                (gen_np.shape[1], gen_np.shape[0]), Image.LANCZOS
            ))

        # LPIPS: expects [1, 3, H, W] in [-1, 1]
        gen_t = torch.from_numpy(gen_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        gt_t = torch.from_numpy(gt_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            lp = lpips_model(gen_t.to(device), gt_t.to(device)).item()
        lpips_vals.append(lp)

        s = ssim_fn(gen_np, gt_np, channel_axis=2, data_range=255)
        ssim_vals.append(float(s))

    del lpips_model

    return {
        "lpips": float(np.mean(lpips_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips_per_frame": lpips_vals,
        "ssim_per_frame": ssim_vals,
        "num_frames_compared": n,
    }


def select_eval_scenes(data_dir, num_scenes, seed=42, eval_only=False):
    """Select scenes for evaluation.

    Args:
        data_dir: Dataset directory containing metadata/captions JSON files.
        num_scenes: Max number of scenes to select.
        seed: Random seed for selection.
        eval_only: If True, use eval_metadata.json/eval_captions.json (held-out
            scenes only). Falls back to full metadata if eval files don't exist.
    """
    if eval_only:
        metadata_path = os.path.join(data_dir, "eval_metadata.json")
        captions_path = os.path.join(data_dir, "eval_captions.json")
        if not os.path.exists(metadata_path):
            print(f"eval_metadata.json not found, falling back to full metadata")
            eval_only = False
    if not eval_only:
        metadata_path = os.path.join(data_dir, "metadata.json")
        captions_path = os.path.join(data_dir, "captions.json")
    assert os.path.exists(metadata_path), f"metadata not found: {metadata_path}"
    assert os.path.exists(captions_path), f"captions not found: {captions_path}"

    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(captions_path) as f:
        captions = json.load(f)

    valid = [e for e in metadata if e["clip_name"] in captions]

    random.seed(seed)
    selected = random.sample(valid, min(num_scenes, len(valid)))

    scenes = []
    for entry in selected:
        clip_name = entry["clip_name"]
        validity_path = entry.get("validity_path", "")
        scenes.append({
            "clip_name": clip_name,
            "depth_path": os.path.join(data_dir, entry["depth_path"]),
            "rgb_path": os.path.join(data_dir, entry["rgb_path"]),
            "ref_path": os.path.join(data_dir, entry["ref_path"]),
            "validity_path": os.path.join(data_dir, validity_path) if validity_path else "",
            "prompt": captions[clip_name],
        })
    return scenes


def run_eval(pipe, scenes, output_dir, num_inference_steps=50, cfg_scale=5.0,
             seed=42, with_ref=True, compute_metrics=True):
    """Run inference on selected scenes and save results."""
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []

    for i, scene in enumerate(scenes):
        clip_name = scene["clip_name"]
        print(f"\n[{i+1}/{len(scenes)}] {clip_name}")
        print(f"  Prompt: {scene['prompt'][:100]}...")

        depth_tensor = torch.load(scene["depth_path"], map_location="cpu", weights_only=True)
        validate_depth_tensor(depth_tensor, label=f"depth for {clip_name}")
        print(f"  Depth: {depth_tensor.shape}, range [{depth_tensor.min():.2f}, {depth_tensor.max():.2f}]")

        validity_mask = None
        if scene.get("validity_path") and os.path.exists(scene["validity_path"]):
            validity_mask = torch.load(scene["validity_path"], map_location="cpu", weights_only=True)
            print(f"  Validity mask: {validity_mask.shape}, valid fraction: {validity_mask.mean():.3f}")

        ref_image = None
        if with_ref and os.path.exists(scene["ref_path"]):
            ref_image = Image.open(scene["ref_path"]).convert("RGB")
            print(f"  Reference: {ref_image.size}")

        print(f"  Generating ({num_inference_steps} steps, cfg={cfg_scale}, seed={seed})...")
        video = pipe(
            prompt=scene["prompt"],
            negative_prompt="blurry, low quality, distorted, ugly",
            vace_video=depth_tensor,
            vace_reference_image=ref_image,
            vace_validity_mask=validity_mask,
            vace_scale=1.0,
            seed=seed,
            height=480,
            width=832,
            num_frames=81,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=5.0,
        )

        scene_dir = os.path.join(output_dir, clip_name)
        os.makedirs(scene_dir, exist_ok=True)

        # Validate pipeline output
        assert video is not None and len(video) > 0, f"Pipeline returned empty output for {clip_name}"

        gen_path = os.path.join(scene_dir, "generated.mp4")
        save_video(video, gen_path, fps=16, quality=5)
        print(f"  Saved: {gen_path}")

        gt_path = os.path.join(scene_dir, "ground_truth.mp4")
        if os.path.exists(scene["rgb_path"]):
            shutil.copy2(scene["rgb_path"], gt_path)

        if ref_image is not None:
            ref_image.save(os.path.join(scene_dir, "reference.jpg"))

        # Save depth keyframes as turbo-colored images
        turbo = matplotlib.colormaps["turbo"]
        for frame_idx, frame_name in [(0, "depth_first"), (40, "depth_mid"), (80, "depth_last")]:
            if frame_idx < depth_tensor.shape[1]:
                disp_01 = ((depth_tensor[0, frame_idx].numpy() + 1.0) / 2.0).clip(0, 1)
                colored = (turbo(disp_01)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(colored).save(os.path.join(scene_dir, f"{frame_name}.jpg"))

        # Save full depth video as turbo-colored mp4
        depth_video_frames = []
        for t in range(depth_tensor.shape[1]):
            disp_01 = ((depth_tensor[0, t].numpy() + 1.0) / 2.0).clip(0, 1)
            colored = (turbo(disp_01)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
            depth_video_frames.append(Image.fromarray(colored))
        depth_video_path = os.path.join(scene_dir, "depth.mp4")
        save_video(depth_video_frames, depth_video_path, fps=16, quality=5)

        # Compute metrics against ground truth
        scene_metrics = None
        if compute_metrics and os.path.exists(scene["rgb_path"]):
            print(f"  Computing metrics...")
            scene_metrics = compute_eval_metrics(video, gt_path)
            if scene_metrics:
                print(f"  LPIPS={scene_metrics['lpips']:.4f}, SSIM={scene_metrics['ssim']:.4f}")
                all_metrics.append({"clip_name": clip_name, **scene_metrics})
                with open(os.path.join(scene_dir, "metrics.json"), "w") as f:
                    json.dump(scene_metrics, f, indent=2)

        with open(os.path.join(scene_dir, "info.json"), "w") as f:
            json.dump({
                "clip_name": clip_name,
                "prompt": scene["prompt"],
                "num_inference_steps": num_inference_steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "with_ref": ref_image is not None,
            }, f, indent=2)

    # Write aggregate summary
    summary = {
        "num_scenes": len(scenes),
        "num_inference_steps": num_inference_steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "with_ref": with_ref,
    }
    if all_metrics:
        summary["avg_lpips"] = float(np.mean([m["lpips"] for m in all_metrics]))
        summary["avg_ssim"] = float(np.mean([m["ssim"] for m in all_metrics]))
        summary["per_scene"] = all_metrics
        print(f"\nAggregate: LPIPS={summary['avg_lpips']:.4f}, SSIM={summary['avg_ssim']:.4f}")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete. Results in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VACE depth-to-RGB checkpoint.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to VACE model directory (e.g. models/Wan2.1-VACE-1.3B/).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned .safetensors checkpoint.")
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ref", action="store_true")
    parser.add_argument("--no_metrics", action="store_true",
                        help="Skip LPIPS/SSIM computation against ground truth.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Use held-out eval scenes only (eval_metadata.json).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.output_dir is None:
        if args.checkpoint:
            name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        else:
            name = "base_model"
        args.output_dir = os.path.join("eval_output", name)

    scenes = select_eval_scenes(args.data_dir, args.scenes, seed=args.seed,
                                eval_only=args.eval_only)
    print(f"Selected {len(scenes)} scenes for evaluation")

    pipe = load_pipeline(args.model_dir, args.checkpoint, args.device)
    run_eval(pipe, scenes, args.output_dir,
             num_inference_steps=args.steps, cfg_scale=args.cfg_scale,
             seed=args.seed, with_ref=not args.no_ref,
             compute_metrics=not args.no_metrics)


if __name__ == "__main__":
    main()
