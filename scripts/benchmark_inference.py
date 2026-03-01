"""
Benchmark inference timing on GH200 with validation data.

Tests different step counts and measures wall-clock time per generation.
"""

import os
import time
import torch
from depth2room.inference import load_pipeline
from depth2room.training.dataset import VACEDepthDataset
from PIL import Image


def main():
    model_dir = os.environ["VACE_MODEL_DIR"]
    data_dir = os.environ["DATA_DIR"]

    print("Loading validation dataset...")
    val_dataset = VACEDepthDataset(
        base_path=data_dir,
        metadata_path=os.path.join(data_dir, "metadata_eval.csv"),
        num_frames=81,
        height=480,
        width=832,
        repeat=1,
        ref_drop_prob=0.0,
    )
    print(f"  {len(val_dataset)} validation samples")

    print("Loading pipeline (base VACE weights, no fine-tuned checkpoint)...")
    t0 = time.time()
    pipe = load_pipeline(model_dir, checkpoint_path=None, device="cuda")
    load_time = time.time() - t0
    print(f"  Pipeline loaded in {load_time:.1f}s")

    # GPU memory after loading
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  GPU memory: {mem_alloc:.1f} GB allocated, {mem_reserved:.1f} GB reserved")

    # Load first validation sample
    data = val_dataset[0]
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

    print(f"\n  Prompt: {prompt[:80]}...")
    print(f"  Depth: {depth_tensor.shape}")
    print(f"  Reference: {'yes' if ref_image else 'no'}")

    base_kwargs = dict(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted",
        vace_video=depth_tensor,
        vace_validity_mask=validity_mask,
        vace_reference_image=ref_image,
        vace_scale=1.0,
        seed=42,
        height=480, width=832, num_frames=81,
        cfg_scale=5.0,
        sigma_shift=5.0,
    )

    # Benchmark different step counts
    for steps in [15, 25, 50]:
        print(f"\n{'='*60}")
        print(f"Benchmarking {steps} steps...")

        # Warmup (first run includes CUDA kernel compilation)
        if steps == 15:
            print("  Warmup run...")
            _ = pipe(**base_kwargs, num_inference_steps=steps)
            torch.cuda.synchronize()

        t0 = time.time()
        frames = pipe(**base_kwargs, num_inference_steps=steps)
        torch.cuda.synchronize()
        gen_time = time.time() - t0

        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {steps} steps: {gen_time:.1f}s ({gen_time/steps:.2f}s/step)")
        print(f"  Output: {len(frames)} frames")
        print(f"  GPU peak memory: {mem_peak:.1f} GB")

        # Reset peak memory tracker
        torch.cuda.reset_peak_memory_stats()

    # Also test without reference
    print(f"\n{'='*60}")
    print("Benchmarking 15 steps WITHOUT reference...")
    no_ref_kwargs = dict(base_kwargs)
    no_ref_kwargs["vace_reference_image"] = None
    t0 = time.time()
    frames = pipe(**no_ref_kwargs, num_inference_steps=15)
    torch.cuda.synchronize()
    gen_time = time.time() - t0
    print(f"  15 steps (no ref): {gen_time:.1f}s")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Pipeline load: {load_time:.1f}s")
    print(f"  GPU memory after load: {mem_alloc:.1f} GB")
    print("  See per-step timings above")
    print(f"\n  For 30 generations (15 clips × 2) at 15 steps:")
    print(f"    1 GPU: ~{30 * gen_time:.0f}s estimated")
    print(f"    4 GPUs: ~{8 * gen_time:.0f}s estimated")


if __name__ == "__main__":
    main()
