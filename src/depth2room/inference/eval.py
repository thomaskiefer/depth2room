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

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video
from depth2room.training.training_unit import replace_vace_unit


def load_pipeline(model_dir, checkpoint_path=None, lora_alpha=1.0, device="cuda"):
    """Load the VACE pipeline with optional LoRA checkpoint."""
    print("Loading base VACE pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(f"{model_dir}/diffusion_pytorch_model.safetensors"),
            ModelConfig(f"{model_dir}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(f"{model_dir}/Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(f"{model_dir}/google/umt5-xxl"),
    )

    pipe = replace_vace_unit(pipe)
    print("Replaced VACE unit with depth-aware version")

    if checkpoint_path is not None:
        print(f"Loading LoRA checkpoint: {checkpoint_path}")
        pipe.load_lora(pipe.vace, checkpoint_path, alpha=lora_alpha)
        print(f"LoRA loaded (alpha={lora_alpha})")

    return pipe


def select_eval_scenes(data_dir, num_scenes, seed=42):
    """Select random scenes for evaluation."""
    metadata_path = os.path.join(data_dir, "metadata.json")
    captions_path = os.path.join(data_dir, "captions.json")

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
        scenes.append({
            "clip_name": clip_name,
            "depth_path": os.path.join(data_dir, entry["depth_path"]),
            "rgb_path": os.path.join(data_dir, entry["rgb_path"]),
            "ref_path": os.path.join(data_dir, entry["ref_path"]),
            "prompt": captions[clip_name],
        })
    return scenes


def run_eval(pipe, scenes, output_dir, num_inference_steps=50, cfg_scale=5.0,
             seed=42, with_ref=True):
    """Run inference on selected scenes and save results."""
    os.makedirs(output_dir, exist_ok=True)

    for i, scene in enumerate(scenes):
        clip_name = scene["clip_name"]
        print(f"\n[{i+1}/{len(scenes)}] {clip_name}")
        print(f"  Prompt: {scene['prompt'][:100]}...")

        depth_tensor = torch.load(scene["depth_path"], map_location="cpu", weights_only=True)
        assert depth_tensor.ndim == 4 and depth_tensor.shape[0] == 3, (
            f"Expected depth tensor [3, T, H, W], got {depth_tensor.shape}"
        )
        assert depth_tensor.min() >= -1.0 - 1e-3 and depth_tensor.max() <= 1.0 + 1e-3, (
            f"Depth tensor values out of [-1, 1] range: [{depth_tensor.min():.4f}, {depth_tensor.max():.4f}]"
        )
        print(f"  Depth: {depth_tensor.shape}, range [{depth_tensor.min():.2f}, {depth_tensor.max():.2f}]")

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

        gen_path = os.path.join(scene_dir, "generated.mp4")
        save_video(video, gen_path, fps=16, quality=5)
        print(f"  Saved: {gen_path}")

        gt_path = os.path.join(scene_dir, "ground_truth.mp4")
        if os.path.exists(scene["rgb_path"]):
            shutil.copy2(scene["rgb_path"], gt_path)

        if ref_image is not None:
            ref_image.save(os.path.join(scene_dir, "reference.jpg"))

        for frame_idx, frame_name in [(0, "depth_first"), (40, "depth_mid"), (80, "depth_last")]:
            if frame_idx < depth_tensor.shape[1]:
                depth_frame = depth_tensor[:, frame_idx]
                depth_img = ((depth_frame + 1) / 2 * 255).clamp(0, 255).byte()
                depth_pil = Image.fromarray(depth_img.permute(1, 2, 0).numpy())
                depth_pil.save(os.path.join(scene_dir, f"{frame_name}.jpg"))

        with open(os.path.join(scene_dir, "info.json"), "w") as f:
            json.dump({
                "clip_name": clip_name,
                "prompt": scene["prompt"],
                "num_inference_steps": num_inference_steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "with_ref": ref_image is not None,
            }, f, indent=2)

    print(f"\nEvaluation complete. Results in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VACE depth-to-RGB checkpoint.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to VACE model directory (e.g. models/Wan2.1-VACE-1.3B/).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA .safetensors checkpoint.")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ref", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.output_dir is None:
        if args.checkpoint:
            name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        else:
            name = "base_model"
        args.output_dir = os.path.join("eval_output", name)

    scenes = select_eval_scenes(args.data_dir, args.scenes, seed=args.seed)
    print(f"Selected {len(scenes)} scenes for evaluation")

    pipe = load_pipeline(args.model_dir, args.checkpoint, args.lora_alpha, args.device)
    run_eval(pipe, scenes, args.output_dir,
             num_inference_steps=args.steps, cfg_scale=args.cfg_scale,
             seed=args.seed, with_ref=not args.no_ref)


if __name__ == "__main__":
    main()
