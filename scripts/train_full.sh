#!/bin/bash
# ============================================================================
# VACE Depth-Conditioned Full Fine-Tuning
# ============================================================================
# Full fine-tuning of the VACE module of Wan2.1-VACE-1.3B for
# depth-conditioned video generation using the CAD-Estate dataset.
# ============================================================================

set -euo pipefail

# ---- Paths (override via environment variables) ----
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VACE_MODEL_DIR="${VACE_MODEL_DIR:?Set VACE_MODEL_DIR to your Wan2.1-VACE model directory}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to your prepared training dataset directory}"
METADATA_CSV="${DATA_DIR}/metadata.csv"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/full_depth_vace}"

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
    --trainable_models "vace" \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --use_gradient_checkpointing_offload \
    --initialize_model_on_cpu \
    --dataset_num_workers 4 \
    --save_steps 200 \
    --gradient_accumulation_steps 1
