"""Inference utilities for depth2room."""

import os

import torch

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from depth2room.training.training_unit import replace_vace_unit


def load_pipeline(model_dir, checkpoint_path=None, lora_alpha=1.0, device="cuda"):
    """Load the VACE pipeline with optional LoRA checkpoint.

    Shared by eval.py and infer_exr.py.
    """
    assert os.path.isdir(model_dir), f"Model directory not found: {model_dir}"
    if checkpoint_path is not None:
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

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
