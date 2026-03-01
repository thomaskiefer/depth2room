#!/bin/bash
#SBATCH --job-name=fix_ref
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=a115
#SBATCH --output=/iopsstor/scratch/cscs/thomaskiefer/logs/fix_ref_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/thomaskiefer/logs/fix_ref_%j.err

srun --environment=ngc-pytorch-26.01 bash -c '
pip install av --quiet 2>&1 | tail -1
export PYTHONPATH=/iopsstor/scratch/cscs/thomaskiefer/depth2room/src:${PYTHONPATH:-}
python -m depth2room.data.fix_ref_frames
'
