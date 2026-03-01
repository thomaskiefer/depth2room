#!/bin/bash
#SBATCH --job-name=viz
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=a115
#SBATCH --output=/iopsstor/scratch/cscs/thomaskiefer/logs/viz_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/thomaskiefer/logs/viz_%j.err

srun --environment=ngc-pytorch-26.01 bash -c '
cd /iopsstor/scratch/cscs/thomaskiefer/depth2room
pip install av matplotlib --quiet 2>&1 | tail -1
export PYTHONPATH=/iopsstor/scratch/cscs/thomaskiefer/depth2room/src:${PYTHONPATH:-}

python -m depth2room.utils.visualize \
    --data_dir /iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/vace_training_dataset \
    --mode grid --num_scenes 6

python -m depth2room.utils.visualize \
    --data_dir /iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/vace_training_dataset \
    --mode sidebyside --num_scenes 6
'
