#!/bin/bash
#SBATCH --job-name=bench-infer
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --account=a115
#SBATCH --output=/iopsstor/scratch/cscs/thomaskiefer/logs/bench_infer_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/thomaskiefer/logs/bench_infer_%j.err

SCRIPTS_DIR=/iopsstor/scratch/cscs/thomaskiefer/depth2room/scripts

srun --environment=${SCRIPTS_DIR}/ngc-pytorch-nccl.toml bash -c '
set -euo pipefail

pip install -e /iopsstor/scratch/cscs/thomaskiefer/DiffSynth-Studio --quiet 2>&1 | tail -1

export PYTHONPATH=/iopsstor/scratch/cscs/thomaskiefer/depth2room/src:${PYTHONPATH:-}
export VACE_MODEL_DIR=/iopsstor/scratch/cscs/thomaskiefer/VACE/models/Wan2.1-VACE-1.3B
export DATA_DIR=/iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/vace_training_dataset

python3 /iopsstor/scratch/cscs/thomaskiefer/depth2room/scripts/benchmark_inference.py
'
