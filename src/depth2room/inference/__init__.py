"""Inference utilities for depth2room."""

import os

import torch

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from depth2room.training.training_unit import replace_vace_unit, patch_pipeline_for_validity_mask


def load_pipeline(model_dir, checkpoint_path=None, device="cuda"):
    """Load the VACE pipeline with optional fine-tuned checkpoint.

    Shared by eval.py and infer_exr.py.
    The returned pipeline accepts vace_validity_mask at inference.
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

    # Expand Context Embedder: 96 -> 160 input channels for validity mask
    old_conv = pipe.vace.vace_patch_embedding
    in_ch_old = old_conv.weight.shape[1]
    if in_ch_old == 96:
        in_ch_new = 160
        out_ch = old_conv.weight.shape[0]
        new_conv = torch.nn.Conv3d(
            in_ch_new, out_ch,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        new_conv.weight.data.zero_()
        new_conv.weight.data[:, :in_ch_old] = old_conv.weight.data
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
        pipe.vace.vace_patch_embedding = new_conv
        print(f"Expanded vace_patch_embedding: {in_ch_old} -> {in_ch_new} input channels")

    if checkpoint_path is not None:
        from safetensors.torch import load_file
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = load_file(checkpoint_path)
        missing, unexpected = pipe.vace.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys")
        print("Checkpoint loaded")

    # Patch __call__ to accept and forward vace_validity_mask
    pipe = patch_pipeline_for_validity_mask(pipe)

    return pipe
