#!/usr/bin/env python3
"""
Inference script for depth-to-RGB video generation using VACE Wan2.1 native backend.

Requires the VACE repository. Set VACE_ROOT to point at the vace directory:
    export VACE_ROOT=/path/to/Wan2.1/vace

Supports:
  - Single depth video inference (from .pt tensor or .mp4)
  - Optional fine-tuned weight loading
  - Autoregressive long video generation (81-frame chunks)
  - Batch evaluation with LPIPS / SSIM metrics against ground-truth
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F

# VACE native backend (optional dependency)
VACE_ROOT = os.environ.get("VACE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..", "VACE", "vace"))
if os.path.isdir(VACE_ROOT) and VACE_ROOT not in sys.path:
    sys.path.insert(0, VACE_ROOT)

from models.wan import WanVace
from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def load_video_tensor(path, device="cpu"):
    """Load a video tensor from .pt file or .mp4 video.

    Works for both depth and RGB tensors. Returns [C, T, H, W] in [-1, 1].
    """
    assert os.path.exists(path), f"Video file not found: {path}"

    if path.endswith(".pt"):
        tensor = torch.load(path, map_location=device, weights_only=True)
        assert tensor.ndim == 4, f"Expected 4D tensor, got shape {tensor.shape}"
        # Detect layout: expected [3, T, H, W]. Also handle [T, 3, H, W] and [T, H, W, 3].
        if tensor.shape[0] == 3:
            pass  # already [C, T, H, W]
        elif tensor.shape[1] == 3:
            logging.info("Detected [T, C, H, W] layout, transposing to [C, T, H, W]")
            tensor = tensor.permute(1, 0, 2, 3)
        elif tensor.shape[3] == 3:
            logging.info("Detected [T, H, W, C] layout, transposing to [C, T, H, W]")
            tensor = tensor.permute(3, 0, 1, 2)
        else:
            logging.warning(
                "Could not auto-detect tensor layout (shape=%s), assuming [C, T, H, W]",
                tensor.shape,
            )
        if tensor.max() > 2.0:
            logging.info("Auto-rescaling from [0,255] to [-1,1]")
            tensor = tensor.float() / 127.5 - 1.0
        elif tensor.min() >= 0.0 and tensor.max() <= 1.0:
            logging.info("Auto-rescaling from [0,1] to [-1,1]")
            tensor = tensor.float() * 2.0 - 1.0
        return tensor.float()
    else:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        assert len(frames) > 0, f"No frames read from video: {path}"
        arr = np.stack(frames, axis=0)
        tensor = torch.from_numpy(arr).float().permute(3, 0, 1, 2)
        tensor = tensor / 127.5 - 1.0
        return tensor


def tensor_to_frames(tensor):
    """Convert [C, T, H, W] tensor in [-1,1] to list of uint8 numpy [H,W,C]."""
    tensor = (tensor + 1.0) * 127.5
    tensor = tensor.clamp(0, 255).to(torch.uint8)
    tensor = tensor.permute(1, 2, 3, 0).cpu().numpy()
    return [tensor[i] for i in range(tensor.shape[0])]


def save_video(frames, path, fps=16):
    """Save list of uint8 RGB frames as mp4."""
    if len(frames) == 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()


def load_finetuned_weights(model, finetuned_path):
    """Load full fine-tuned DiT weights into the model.

    Supports both .safetensors (from training logger) and .pt/.bin (legacy).
    """
    logging.info(f"Loading fine-tuned weights from {finetuned_path}")
    if finetuned_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(finetuned_path)
    else:
        state_dict = torch.load(finetuned_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys: {missing[:10]}...")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected[:10]}...")
    logging.info("Fine-tuned weights loaded.")


def compute_metrics(output_tensor, gt_tensor, lpips_model=None):
    """Compute LPIPS and SSIM between output and ground-truth.

    Args:
        output_tensor: [C, T, H, W] generated video tensor in [-1, 1].
        gt_tensor: [C, T, H, W] ground-truth video tensor in [-1, 1].
        lpips_model: Optional pre-initialized LPIPS model. If None, creates one.
    """
    from skimage.metrics import structural_similarity as ssim_fn

    cleanup_lpips = False
    if lpips_model is None:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(output_tensor.device)
        lpips_model.eval()
        cleanup_lpips = True

    T = min(output_tensor.shape[1], gt_tensor.shape[1])
    lpips_values = []
    ssim_values = []

    for t in range(T):
        out_frame = output_tensor[:, t, :, :].unsqueeze(0)
        gt_frame = gt_tensor[:, t, :, :].unsqueeze(0)

        with torch.no_grad():
            lp = lpips_model(out_frame, gt_frame).item()
        lpips_values.append(lp)

        out_np = ((out_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        gt_np = ((gt_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        s = ssim_fn(out_np, gt_np, channel_axis=2, data_range=255)
        ssim_values.append(s)

    if cleanup_lpips:
        del lpips_model

    return {
        "lpips": float(np.mean(lpips_values)),
        "ssim": float(np.mean(ssim_values)),
        "lpips_per_frame": [float(v) for v in lpips_values],
        "ssim_per_frame": [float(v) for v in ssim_values],
    }


def run_single_inference(
    wan_vace, depth_tensor, prompt, output_dir,
    context_scale=0.5, num_inference_steps=50, seed=2025,
    size=(480, 832), frame_num=81, gt_tensor=None, sample_name="output",
    lpips_model=None,
):
    """Run depth-to-RGB generation for a single depth video chunk."""
    device = wan_vace.device
    H, W = size

    if depth_tensor.shape[2] != H or depth_tensor.shape[3] != W:
        depth_tensor = F.interpolate(
            depth_tensor.permute(1, 0, 2, 3), size=(H, W),
            mode="bilinear", align_corners=False,
        ).permute(1, 0, 2, 3)

    T = depth_tensor.shape[1]
    if T < frame_num:
        logging.warning("Depth has %d frames but need %d — padding last frame", T, frame_num)
        pad = depth_tensor[:, -1:, :, :].expand(-1, frame_num - T, -1, -1)
        depth_tensor = torch.cat([depth_tensor, pad], dim=1)
    elif T > frame_num:
        depth_tensor = depth_tensor[:, :frame_num, :, :]

    depth_tensor = depth_tensor.to(device)

    src_video = [depth_tensor]
    src_mask = [torch.ones_like(depth_tensor, device=device)]
    src_ref_images = [None]

    logging.info(f"Running inference: {frame_num} frames at {H}x{W}")

    output_video = wan_vace.generate(
        prompt, src_video, src_mask, src_ref_images,
        size=(H, W), frame_num=frame_num,
        context_scale=context_scale, shift=16,
        sample_solver="unipc", sampling_steps=num_inference_steps,
        guide_scale=5.0, seed=seed, offload_model=True,
    )

    assert output_video is not None, "Pipeline returned None"
    if not torch.isfinite(output_video).all():
        logging.warning("Pipeline output contains NaN/Inf values")

    os.makedirs(output_dir, exist_ok=True)
    out_frames = tensor_to_frames(output_video.cpu())
    out_path = os.path.join(output_dir, f"{sample_name}.mp4")
    save_video(out_frames, out_path, fps=16)
    logging.info(f"Saved output to {out_path}")

    metrics = None
    if gt_tensor is not None:
        gt_tensor = gt_tensor.to(output_video.device)
        if gt_tensor.shape[2] != H or gt_tensor.shape[3] != W:
            gt_tensor = F.interpolate(
                gt_tensor.permute(1, 0, 2, 3), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(1, 0, 2, 3)
        if gt_tensor.shape[1] > frame_num:
            gt_tensor = gt_tensor[:, :frame_num, :, :]
        metrics = compute_metrics(output_video, gt_tensor, lpips_model=lpips_model)
        logging.info(f"Metrics: LPIPS={metrics['lpips']:.4f}, SSIM={metrics['ssim']:.4f}")

        with open(os.path.join(output_dir, f"{sample_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return output_video, metrics


def run_autoregressive_inference(
    wan_vace, depth_tensor, prompt, output_dir,
    context_scale=0.5, num_inference_steps=50, seed=2025,
    size=(480, 832), chunk_frames=81, overlap_frames=20, discard_last=4,
):
    """Generate long video autoregressively, chaining 81-frame chunks."""
    device = wan_vace.device
    H, W = size

    if depth_tensor.shape[2] != H or depth_tensor.shape[3] != W:
        depth_tensor = F.interpolate(
            depth_tensor.permute(1, 0, 2, 3), size=(H, W),
            mode="bilinear", align_corners=False,
        ).permute(1, 0, 2, 3)

    total_frames = depth_tensor.shape[1]
    effective_per_chunk = chunk_frames - overlap_frames - discard_last
    num_chunks = math.ceil(total_frames / effective_per_chunk)

    logging.info(f"Autoregressive: {total_frames} total, {num_chunks} chunks")

    os.makedirs(output_dir, exist_ok=True)
    all_output_frames = []
    prev_output_tensor = None

    for chunk_idx in range(num_chunks):
        start = chunk_idx * effective_per_chunk
        end = min(start + chunk_frames, total_frames)

        chunk_depth = depth_tensor[:, start:end, :, :]
        T_chunk = chunk_depth.shape[1]
        if T_chunk < chunk_frames:
            pad = chunk_depth[:, -1:, :, :].expand(-1, chunk_frames - T_chunk, -1, -1)
            chunk_depth = torch.cat([chunk_depth, pad], dim=1)

        chunk_depth = chunk_depth.to(device)

        if chunk_idx == 0 or prev_output_tensor is None:
            src_video = [chunk_depth]
            src_mask = [torch.ones_like(chunk_depth, device=device)]
        else:
            if discard_last > 0:
                prev_overlap = prev_output_tensor[
                    :, -(overlap_frames + discard_last):-discard_last, :, :
                ]
            else:
                prev_overlap = prev_output_tensor[:, -overlap_frames:, :, :]

            depth_new = chunk_depth[:, overlap_frames:, :, :]
            composed_video = torch.cat([prev_overlap, depth_new], dim=1)

            mask_keep = torch.zeros(3, overlap_frames, H, W, device=device, dtype=chunk_depth.dtype)
            mask_gen = torch.ones(3, chunk_frames - overlap_frames, H, W, device=device, dtype=chunk_depth.dtype)
            composed_mask = torch.cat([mask_keep, mask_gen], dim=1)

            src_video = [composed_video]
            src_mask = [composed_mask]

        src_ref_images = [None]

        logging.info(f"Generating chunk {chunk_idx + 1}/{num_chunks} (frames {start}-{end})")

        video = wan_vace.generate(
            prompt, src_video, src_mask, src_ref_images,
            size=(H, W), frame_num=chunk_frames,
            context_scale=context_scale, shift=16,
            sample_solver="unipc", sampling_steps=num_inference_steps,
            guide_scale=5.0, seed=seed, offload_model=True,
        )

        prev_output_tensor = video.clone()

        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:03d}.mp4")
        chunk_frames_np = tensor_to_frames(video.cpu())
        save_video(chunk_frames_np, chunk_path, fps=16)

        actual_needed = end - start
        is_last = chunk_idx == num_chunks - 1
        if is_last:
            trimmed = chunk_frames_np[:actual_needed]
        else:
            keep = min(actual_needed, len(chunk_frames_np) - discard_last)
            trimmed = chunk_frames_np[:keep]

        if len(all_output_frames) > 0:
            all_output_frames.extend(trimmed[overlap_frames:])
        else:
            all_output_frames.extend(trimmed)

        del video, src_video, src_mask
        gc.collect()
        torch.cuda.empty_cache()

    final_path = os.path.join(output_dir, "final_video.mp4")
    save_video(all_output_frames, final_path, fps=16)
    logging.info(f"Saved final video: {len(all_output_frames)} frames to {final_path}")
    return all_output_frames


def run_batch_evaluation(
    wan_vace, test_dir, prompt, output_dir,
    context_scale=0.5, num_inference_steps=50, seed=2025,
    size=(480, 832), frame_num=81,
):
    """Evaluate over a test set of depth tensors."""
    depth_dir = os.path.join(test_dir, "depth")
    gt_dir = os.path.join(test_dir, "rgb")
    assert os.path.isdir(depth_dir)

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".pt")])
    logging.info(f"Found {len(depth_files)} test samples")

    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []

    # Create LPIPS model once for the entire batch
    lpips_model = None
    if os.path.isdir(gt_dir):
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(wan_vace.device)
            lpips_model.eval()
            logging.info("LPIPS model loaded for batch evaluation")
        except ImportError:
            logging.warning("lpips not installed, skipping metrics")

    for i, depth_file in enumerate(depth_files):
        depth_path = os.path.join(depth_dir, depth_file)
        sample_name = os.path.splitext(depth_file)[0]
        logging.info(f"[{i+1}/{len(depth_files)}] {sample_name}")

        depth_tensor = load_video_tensor(depth_path)
        gt_tensor = None
        gt_path = os.path.join(gt_dir, depth_file)
        if os.path.exists(gt_path):
            gt_tensor = load_video_tensor(gt_path)

        _, metrics = run_single_inference(
            wan_vace, depth_tensor, prompt, output_dir,
            context_scale, num_inference_steps, seed, size, frame_num,
            gt_tensor, sample_name, lpips_model=lpips_model,
        )

        if metrics:
            metrics["sample"] = sample_name
            all_metrics.append(metrics)

        gc.collect()
        torch.cuda.empty_cache()

    del lpips_model

    if all_metrics:
        summary = {
            "num_samples": len(all_metrics),
            "avg_lpips": float(np.mean([m["lpips"] for m in all_metrics])),
            "avg_ssim": float(np.mean([m["ssim"] for m in all_metrics])),
            "per_sample": all_metrics,
        }
        with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Eval done: LPIPS={summary['avg_lpips']:.4f}, SSIM={summary['avg_ssim']:.4f}")


def get_parser():
    parser = argparse.ArgumentParser(description="Depth-to-RGB inference using VACE Wan2.1 native backend")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="vace-1.3B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--finetuned_path", type=str, default=None)

    parser.add_argument("--src_video", type=str, default=None)
    parser.add_argument("--gt_video", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/depth2rgb")
    parser.add_argument("--prompt", type=str, default="A photorealistic scene.")

    parser.add_argument("--context_scale", type=float, default=0.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--size", type=str, default="480p", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=81)

    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--overlap_frames", type=int, default=20)
    parser.add_argument("--discard_last", type=int, default=4)

    parser.add_argument("--batch_eval", action="store_true")
    parser.add_argument("--test_dir", type=str, default=None)

    parser.add_argument("--device_id", type=int, default=0)

    return parser


def main():
    setup_logging()
    args = get_parser().parse_args()

    assert os.path.isdir(args.model_path), f"Model path not found: {args.model_path}"

    if (args.frame_num - 1) % 4 != 0:
        old = args.frame_num
        args.frame_num = ((args.frame_num - 1) // 4) * 4 + 1
        logging.warning(f"frame_num adjusted from {old} to {args.frame_num}")

    cfg = WAN_CONFIGS[args.model_name]
    size = SIZE_CONFIGS[args.size]

    wan_vace = WanVace(
        config=cfg, checkpoint_dir=args.model_path,
        device_id=args.device_id, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
    )

    if args.finetuned_path is not None:
        load_finetuned_weights(wan_vace.model, args.finetuned_path)
        wan_vace.model.to(wan_vace.device)

    if args.batch_eval:
        assert args.test_dir is not None
        run_batch_evaluation(
            wan_vace, args.test_dir, args.prompt, args.output_dir,
            args.context_scale, args.num_inference_steps, args.seed, size, args.frame_num,
        )
    elif args.autoregressive:
        assert args.src_video is not None
        depth_tensor = load_video_tensor(args.src_video)
        run_autoregressive_inference(
            wan_vace, depth_tensor, args.prompt, args.output_dir,
            args.context_scale, args.num_inference_steps, args.seed, size,
            args.frame_num, args.overlap_frames, args.discard_last,
        )
    else:
        assert args.src_video is not None
        depth_tensor = load_video_tensor(args.src_video)
        gt_tensor = None
        if args.gt_video:
            gt_tensor = load_video_tensor(args.gt_video)
        run_single_inference(
            wan_vace, depth_tensor, args.prompt, args.output_dir,
            args.context_scale, args.num_inference_steps, args.seed, size,
            args.frame_num, gt_tensor,
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()
