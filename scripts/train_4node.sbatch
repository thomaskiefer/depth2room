#!/bin/bash
#SBATCH --job-name=depth2room
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --account=a115
#SBATCH --output=/iopsstor/scratch/cscs/thomaskiefer/logs/train_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/thomaskiefer/logs/train_%j.err

SCRIPTS_DIR=/iopsstor/scratch/cscs/thomaskiefer/depth2room/scripts

srun -ul --environment=${SCRIPTS_DIR}/ngc-pytorch-nccl.toml bash -c '
set -euo pipefail

# Only rank 0 per node installs (avoid pip race)
if [ "${SLURM_LOCALID}" = "0" ]; then
    pip install -e /iopsstor/scratch/cscs/thomaskiefer/DiffSynth-Studio --quiet 2>&1 | tail -1
    pip install accelerate wandb --quiet 2>&1 | tail -1
    touch /tmp/pip_done_${SLURM_JOB_ID}
else
    while [ ! -f /tmp/pip_done_${SLURM_JOB_ID} ]; do sleep 1; done
fi

export VACE_MODEL_DIR=/iopsstor/scratch/cscs/thomaskiefer/VACE/models/Wan2.1-VACE-1.3B
export DATA_DIR=/iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/vace_training_dataset
export PYTHONPATH=/iopsstor/scratch/cscs/thomaskiefer/depth2room/src:${PYTHONPATH:-}
export NCCL_TIMEOUT=1800

# Distributed env vars (srun-based launch, no torchrun needed)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export ACCELERATE_MIXED_PRECISION=bf16

python3 -m depth2room.training.train \
    --task "sft" \
    --model_paths "[\"${VACE_MODEL_DIR}/diffusion_pytorch_model.safetensors\",\"${VACE_MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${VACE_MODEL_DIR}/Wan2.1_VAE.pth\"]" \
    --tokenizer_path "${VACE_MODEL_DIR}/google/umt5-xxl" \
    --dataset_base_path "${DATA_DIR}" \
    --dataset_metadata_path "${DATA_DIR}/metadata.csv" \
    --extra_inputs "vace_video_tensor,vace_reference_image,vace_validity_mask" \
    --output_path "/capstor/scratch/cscs/thomaskiefer/depth2room/output/full_depth_vace" \
    --remove_prefix_in_ckpt "pipe.vace." \
    --trainable_models "vace" \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --num_epochs 5 \
    --dataset_repeat 10 \
    --use_gradient_checkpointing \
    --initialize_model_on_cpu \
    --dataset_num_workers 8 \
    --save_steps 300 \
    --gradient_accumulation_steps 1 \
    # To resume from a checkpoint, replace the line above with:
    # --gradient_accumulation_steps 1 \
    # --resume_from_checkpoint "/capstor/scratch/cscs/thomaskiefer/depth2room/output/full_depth_vace/resume-N"
'
