"""
Background subprocess for inference visualization at checkpoints.

Launched by ModelLogger._launch_inference_viz_subprocess(). Loads its own
inference pipeline, generates scenes (with/without reference), and logs
side-by-side comparison videos to an existing wandb run.

Each logged video shows: Generated | Depth (turbo) | Validity mask
Ground truth and reference are logged separately for comparison.

This runs as a separate process so it doesn't block the distributed training
job or risk NCCL timeouts.
"""

import argparse
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image


def frames_to_numpy(frames):
    """Convert list of PIL Images to numpy array (T, H, W, C)."""
    return np.stack([np.array(f) for f in frames])


def depth_to_turbo_frames(depth_tensor):
    """Convert depth tensor [C, T, H, W] to turbo-colored numpy (T, H, W, 3)."""
    import matplotlib
    turbo = matplotlib.colormaps["turbo"]
    T = depth_tensor.shape[1]
    out = []
    for t in range(T):
        disp = ((depth_tensor[0, t].numpy() + 1.0) / 2.0).clip(0, 1)
        colored = (turbo(disp)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        out.append(colored)
    return np.stack(out)


def validity_to_frames(validity_mask):
    """Convert validity mask [1, T, H, W] to grayscale numpy (T, H, W, 3).

    Valid pixels = white, invalid = dark red.
    """
    T = validity_mask.shape[1]
    out = []
    for t in range(T):
        mask = validity_mask[0, t].numpy()  # (H, W), values in {0, 1}
        frame = np.zeros((*mask.shape, 3), dtype=np.uint8)
        frame[mask > 0.5] = [255, 255, 255]  # valid = white
        frame[mask <= 0.5] = [80, 20, 20]    # invalid = dark red
        out.append(frame)
    return np.stack(out)


def compose_sidebyside(gen_np, depth_np, validity_np):
    """Compose Generated | Depth | Validity side-by-side.

    All inputs: (T, H, W, 3) uint8 numpy arrays.
    Returns: (T, H, W*3, 3) uint8 numpy array.
    """
    return np.concatenate([gen_np, depth_np, validity_np], axis=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_metadata", required=True)
    parser.add_argument("--val_base_path", required=True)
    parser.add_argument("--val_indices", required=True, help="Comma-separated dataset indices")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--viz_steps", type=int, default=15)
    parser.add_argument("--step_number", type=int, required=True)
    parser.add_argument("--wandb_project", required=True)
    parser.add_argument("--wandb_entity", required=True)
    parser.add_argument("--wandb_run_id", required=True)
    args = parser.parse_args()

    val_indices = [int(x) for x in args.val_indices.split(",")]
    step = args.step_number

    print(f"[viz_worker] Starting inference viz for step {step}")
    print(f"[viz_worker] Checkpoint: {args.checkpoint}")
    print(f"[viz_worker] Scenes: {len(val_indices)} at indices {val_indices}")

    try:
        import wandb
        from depth2room.inference import load_pipeline
        from depth2room.training.dataset import VACEDepthDataset

        # Resume the existing wandb run to log to the same dashboard
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_run_id,
            resume="must",
        )

        # Load validation dataset (same config as training, no repeat, no ref dropout)
        val_dataset = VACEDepthDataset(
            base_path=args.val_base_path,
            metadata_path=args.val_metadata,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            repeat=1,
            ref_drop_prob=0.0,
        )

        print(f"[viz_worker] Loading inference pipeline...")
        pipe = load_pipeline(args.model_dir, args.checkpoint, device="cuda")

        log_dict = {}
        fps = 16

        for scene_idx, dataset_idx in enumerate(val_indices):
            tag = f"s{scene_idx}"
            data = val_dataset[dataset_idx]
            depth_tensor = data["vace_video_tensor"]
            validity_mask = data.get("vace_validity_mask")
            prompt = data["prompt"]

            ref_image = None
            ref_val = data.get("vace_reference_image")
            if ref_val is not None:
                if isinstance(ref_val, list) and len(ref_val) > 0:
                    ref_image = ref_val[0]
                elif isinstance(ref_val, Image.Image):
                    ref_image = ref_val

            inference_kwargs = dict(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted",
                vace_video=depth_tensor,
                vace_validity_mask=validity_mask,
                vace_scale=1.0,
                seed=42,
                height=args.height, width=args.width, num_frames=args.num_frames,
                cfg_scale=5.0,
                num_inference_steps=args.viz_steps,
                sigma_shift=5.0,
            )

            # Prepare condition videos (shared across with_ref/no_ref)
            depth_np = depth_to_turbo_frames(depth_tensor)  # (T, H, W, 3)
            if validity_mask is not None:
                validity_np = validity_to_frames(validity_mask)  # (T, H, W, 3)
            else:
                # No validity mask — show all-white (all valid)
                validity_np = np.full_like(depth_np, 255)

            # Generate WITH reference
            print(f"[viz_worker] Scene {scene_idx}: generating with reference ({args.viz_steps} steps)...")
            frames_with_ref = pipe(**inference_kwargs, vace_reference_image=ref_image)
            gen_np = frames_to_numpy(frames_with_ref)  # (T, H, W, 3)
            # Side-by-side: Generated | Depth | Validity
            sbs = compose_sidebyside(gen_np, depth_np, validity_np)
            log_dict[f"viz/{tag}_with_ref"] = wandb.Video(
                sbs.transpose(0, 3, 1, 2),  # (T, C, H, W*3)
                fps=fps,
                caption=f"step-{step} {tag} with_ref | depth | validity",
            )

            # Generate WITHOUT reference
            print(f"[viz_worker] Scene {scene_idx}: generating without reference ({args.viz_steps} steps)...")
            frames_no_ref = pipe(**inference_kwargs, vace_reference_image=None)
            gen_np = frames_to_numpy(frames_no_ref)
            sbs = compose_sidebyside(gen_np, depth_np, validity_np)
            log_dict[f"viz/{tag}_no_ref"] = wandb.Video(
                sbs.transpose(0, 3, 1, 2),
                fps=fps,
                caption=f"step-{step} {tag} no_ref | depth | validity",
            )

            # Ground truth side-by-side: GT | Depth | Validity
            gt_frames = data.get("video")
            if gt_frames and len(gt_frames) > 0:
                gt_np = frames_to_numpy(gt_frames)
                sbs_gt = compose_sidebyside(gt_np, depth_np, validity_np)
                log_dict[f"viz/{tag}_gt"] = wandb.Video(
                    sbs_gt.transpose(0, 3, 1, 2),
                    fps=fps,
                    caption=f"{tag} ground_truth | depth | validity",
                )

            # Reference image (static)
            if ref_image is not None:
                log_dict[f"viz/{tag}_reference"] = wandb.Image(
                    ref_image, caption=f"{tag} reference",
                )

            print(f"[viz_worker] Scene {scene_idx}: done ({len(log_dict)} items so far)")

        wandb.log(log_dict, step=step)
        print(f"[viz_worker] Logged {len(log_dict)} viz items to wandb at step {step}")

        del pipe
        torch.cuda.empty_cache()
        wandb.finish()

    except Exception as e:
        print(f"[viz_worker] Failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
