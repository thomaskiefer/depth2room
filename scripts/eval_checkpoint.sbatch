#!/bin/bash
# ============================================================================
# Evaluate a single VACE checkpoint by running inference on held-out scenes.
# Generates videos BOTH with and without reference frame.
#
# Usage:
#   CHECKPOINT=/path/to/step-200.safetensors sbatch eval_checkpoint.sbatch
#   # or evaluate all checkpoints:
#   bash eval_all_checkpoints.sh /path/to/output/full_depth_vace
# ============================================================================
#SBATCH --job-name=eval-ckpt
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --account=a115
#SBATCH --output=/iopsstor/scratch/cscs/thomaskiefer/logs/eval_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/thomaskiefer/logs/eval_%j.err

SCRIPTS_DIR=/iopsstor/scratch/cscs/thomaskiefer/depth2room/scripts

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/step-NNN.safetensors}"

srun --environment=${SCRIPTS_DIR}/ngc-pytorch-nccl.toml bash -c '
set -euo pipefail

pip install -e /iopsstor/scratch/cscs/thomaskiefer/DiffSynth-Studio --quiet 2>&1 | tail -1

export PYTHONPATH=/iopsstor/scratch/cscs/thomaskiefer/depth2room/src:${PYTHONPATH:-}

VACE_MODEL_DIR=/iopsstor/scratch/cscs/thomaskiefer/VACE/models/Wan2.1-VACE-1.3B
DATA_DIR=/iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/vace_training_dataset
CKPT_NAME=$(basename '"${CHECKPOINT}"' .safetensors)

echo "Evaluating checkpoint: '"${CHECKPOINT}"'"

# Run 1: WITH reference frame
echo ""
echo "=== Eval with reference frame ==="
python3 -m depth2room.inference.eval \
    --model_dir "${VACE_MODEL_DIR}" \
    --checkpoint '"${CHECKPOINT}"' \
    --data_dir "${DATA_DIR}" \
    --scenes 5 \
    --steps 30 \
    --output_dir "/iopsstor/scratch/cscs/thomaskiefer/depth2room/output/eval/${CKPT_NAME}/with_ref" \
    --eval_only \
    --no_metrics \
    --device cuda

# Run 2: WITHOUT reference frame
echo ""
echo "=== Eval without reference frame ==="
python3 -m depth2room.inference.eval \
    --model_dir "${VACE_MODEL_DIR}" \
    --checkpoint '"${CHECKPOINT}"' \
    --data_dir "${DATA_DIR}" \
    --scenes 5 \
    --steps 30 \
    --output_dir "/iopsstor/scratch/cscs/thomaskiefer/depth2room/output/eval/${CKPT_NAME}/no_ref" \
    --eval_only \
    --no_ref \
    --no_metrics \
    --device cuda

echo ""
echo "Done. Results in /iopsstor/scratch/cscs/thomaskiefer/depth2room/output/eval/${CKPT_NAME}/"
'
