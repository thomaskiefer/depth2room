#!/bin/bash
# ============================================================================
# Submit eval jobs for all checkpoints in a training output directory.
#
# Usage:
#   bash eval_all_checkpoints.sh /path/to/output/full_depth_vace
#   bash eval_all_checkpoints.sh  # uses default output path
# ============================================================================

set -euo pipefail

OUTPUT_DIR="${1:-/iopsstor/scratch/cscs/thomaskiefer/depth2room/output/full_depth_vace}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SBATCH="${SCRIPT_DIR}/eval_checkpoint.sbatch"

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Error: Output directory not found: ${OUTPUT_DIR}"
    exit 1
fi

# Find all checkpoint files, sorted by step number
CHECKPOINTS=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "*.safetensors" | sort -t- -k2 -n)

if [ -z "${CHECKPOINTS}" ]; then
    echo "No checkpoints found in ${OUTPUT_DIR}"
    exit 0
fi

echo "Found checkpoints in ${OUTPUT_DIR}:"
SUBMITTED=0
for CKPT in ${CHECKPOINTS}; do
    CKPT_NAME=$(basename "${CKPT}" .safetensors)
    EVAL_DIR="/iopsstor/scratch/cscs/thomaskiefer/depth2room/output/eval/${CKPT_NAME}"

    # Skip if already evaluated (check both with_ref and no_ref subdirs)
    if [ -f "${EVAL_DIR}/with_ref/summary.json" ] && [ -f "${EVAL_DIR}/no_ref/summary.json" ]; then
        echo "  [skip] ${CKPT_NAME} (already evaluated)"
        continue
    fi

    echo "  [submit] ${CKPT_NAME}"
    CHECKPOINT="${CKPT}" sbatch "${EVAL_SBATCH}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Submitted ${SUBMITTED} evaluation jobs."
