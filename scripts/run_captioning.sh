#!/bin/bash
# ============================================================================
# Full captioning pipeline: CapRL-Qwen3VL-4B (VLM) + gpt-oss-120b (LLM)
#
# 4 nodes × 4 GPUs = 16 shards in parallel, 22 min on debug partition.
# Fits debug QOS: 4 nodes × 22 min = 88 node-minutes (limit: 90).
# Resume-safe: resubmit to pick up where interrupted.
#
# Usage:
#   FRESH=1 sbatch scripts/run_captioning.sh  # first run (clears old shards)
#   sbatch scripts/run_captioning.sh          # resume
# ============================================================================
#SBATCH --job-name=caption
#SBATCH --partition=debug
#SBATCH --time=00:22:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --account=a115
#SBATCH --output=/iopsstor/scratch/cscs/thomaskiefer/logs/caption_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/thomaskiefer/logs/caption_%j.err

set -euo pipefail

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HOME=/iopsstor/scratch/cscs/thomaskiefer/.cache/huggingface

NUM_SHARDS=16
DATASET_DIR=/iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/vace_training_dataset
ANNOTATIONS_DIR=/iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/annotations

# Fresh start: delete old shard files
if [[ "${FRESH:-0}" == "1" ]]; then
    echo "FRESH=1: removing old shard files..."
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        rm -f "${DATASET_DIR}/captions_shard${i}.json"
        rm -f "${DATASET_DIR}/captions_vlm_shard${i}.json"
    done
fi

echo "=== Captioning: ${NUM_SHARDS} shards on ${SLURM_JOB_NUM_NODES} nodes (4 GPUs each) ==="
echo "Start: $(date)"

srun --environment=ngc-pytorch-26.01 bash -c '
set -euo pipefail
SHARD_ID=${SLURM_PROCID}
NUM_SHARDS='"${NUM_SHARDS}"'
DATASET_DIR='"${DATASET_DIR}"'
ANNOTATIONS_DIR='"${ANNOTATIONS_DIR}"'

# Pin each task to its local GPU
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}

echo "[shard ${SHARD_ID}/${NUM_SHARDS}] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"

cd /iopsstor/scratch/cscs/thomaskiefer/depth2room

# Only one task per node installs; others wait
LOCK="/tmp/pip_install_done_${SLURM_JOB_ID}"
if [[ "${SLURM_LOCALID}" == "0" ]]; then
    pip install -e ".[data]" --quiet 2>&1 | tail -1
    touch "${LOCK}"
else
    while [[ ! -f "${LOCK}" ]]; do sleep 1; done
fi

python -m depth2room.data.generate_captions \
    --dataset_dir ${DATASET_DIR} \
    --annotations_dir ${ANNOTATIONS_DIR} \
    --shard_id ${SHARD_ID} --num_shards ${NUM_SHARDS}

echo "[shard ${SHARD_ID}] Done: $(date)"
'

echo "All tasks finished: $(date)"

# --- Auto-merge: if all shards exist, combine into captions.json ---
ALL_DONE=true
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    if [[ ! -f "${DATASET_DIR}/captions_shard${i}.json" ]]; then
        echo "Shard ${i} not yet complete."
        ALL_DONE=false
    fi
done

if [[ "${ALL_DONE}" == "true" ]]; then
    echo "All ${NUM_SHARDS} shards complete — merging into captions.json..."
    python3 -c "
import json
merged = {}
for i in range(${NUM_SHARDS}):
    path = '${DATASET_DIR}/captions_shard' + str(i) + '.json'
    with open(path) as f:
        shard = json.load(f)
    merged.update(shard)
    print(f'  shard {i}: {len(shard)} captions')
out = '${DATASET_DIR}/captions.json'
with open(out, 'w') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
print(f'Merged {len(merged)} total captions -> {out}')
"
else
    echo "Resubmit to continue: sbatch scripts/run_captioning.sh"
fi

echo "End: $(date)"
