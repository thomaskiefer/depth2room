#!/bin/bash
# ============================================================================
# VACE Depth-Conditioned LoRA Training
# ============================================================================
# Trains a LoRA adapter on the VACE module of Wan2.1-VACE-1.3B for
# depth-conditioned video generation using the CAD-Estate dataset.
# ============================================================================

set -euo pipefail

# ---- Paths (override via environment variables) ----
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VACE_MODEL_DIR="${VACE_MODEL_DIR:?Set VACE_MODEL_DIR to your Wan2.1-VACE model directory}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to your prepared training dataset directory}"
METADATA_CSV="${DATA_DIR}/metadata.csv"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/lora_depth_vace}"

# ---- NCCL timeout mitigation for distributed checkpoint saves ----
export NCCL_TIMEOUT=1800

# ---- Launch training ----
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m depth2room.training.train \
    --task "sft" \
    --model_paths '["'"${VACE_MODEL_DIR}/diffusion_pytorch_model.safetensors"'","'"${VACE_MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth"'","'"${VACE_MODEL_DIR}/Wan2.1_VAE.pth"'"]' \
    --tokenizer_path "${VACE_MODEL_DIR}/google/umt5-xxl" \
    --dataset_base_path "${DATA_DIR}" \
    --dataset_metadata_path "${METADATA_CSV}" \
    --data_file_keys "video,vace_reference_image" \
    --extra_inputs "vace_video_tensor,vace_reference_image" \
    --output_path "${OUTPUT_DIR}" \
    --remove_prefix_in_ckpt "pipe.vace." \
    --lora_base_model "vace" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --save_steps 200 \
    --use_gradient_checkpointing \
    --dataset_num_workers 0 \
    --gradient_accumulation_steps 1
