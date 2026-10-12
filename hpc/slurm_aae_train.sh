#!/bin/bash
#
# slurm_aae_train.sh — SLURM batch job for final AAE training (§4.2)
#
# Runs the final training with validated hyperparameters across 3 seeds,
# saves the AAE checkpoint, and writes aae_final_metrics.csv.
#
# Usage:
#   SEEDS="42 123 7" ./hpc_connect.sh batch hpc/slurm_aae_train.sh
# ==============================================================================

# ── SLURM Directives ────────────────────────────────────────────────────────
#SBATCH --job-name=am01_aae_train
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --mail-type=END,FAIL

# ── Environment ─────────────────────────────────────────────────────────────
set -euo pipefail
module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export SCRATCH_PROJECT="am01"

# ── Working directory on scratch ────────────────────────────────────────────
SCRATCH_DIR="${SCRATCH:-${HOME}/scratch}"
mkdir -p "${SCRATCH_DIR}/${SCRATCH_PROJECT}"
cd "${SCRATCH_DIR}/${SCRATCH_PROJECT}"
mkdir -p logs

# ── Sync project code from home ─────────────────────────────────────────────
if [[ -d "${HOME}/am01_project" ]]; then
    rsync -a \
        --exclude='.venv/' \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='.ipynb_checkpoints/' \
        --exclude='data/raw/' \
        --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
        "${HOME}/am01_project/" "${SCRATCH_DIR}/${SCRATCH_PROJECT}/"
    echo "Synced project to scratch."
fi

# ── Activate uv environment ─────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    echo "--- Syncing frozen deps on scratch ---"
    uv sync --frozen --quiet
fi

# ── Phase 4.2: Final AAE training with validated HPs (3 seeds) ──────────────
echo "=== Phase 4.2: Final AAE training (3 seeds) ==="
SEEDS="${SEEDS:-42 123 7}"
uv run python -m src.models.train_aae \
    --config config/params_validated_aae.yaml \
    --seeds ${SEEDS}

# ── Verify outputs ──────────────────────────────────────────────────────────
echo "=== Outputs ==="
ls -lh reports/checkpoints/aae_final.pth reports/tables/aae_final_metrics.csv 2>/dev/null || true

# ── Fetch results back to $HOME ─────────────────────────────────────────────
RESULTS_DIR="${SCRATCH_DIR}/${SCRATCH_PROJECT}"
HOME_PROJECT="${HOME}/am01_project"

mkdir -p "${HOME_PROJECT}/reports/checkpoints" "${HOME_PROJECT}/reports/tables" "${HOME_PROJECT}/logs"

rsync -a --progress --ignore-existing \
    "${RESULTS_DIR}/reports/checkpoints/" "${HOME_PROJECT}/reports/checkpoints/" 2>/dev/null || true
rsync -a --progress --ignore-existing \
    "${RESULTS_DIR}/reports/tables/" "${HOME_PROJECT}/reports/tables/" 2>/dev/null || true

cp "${RESULTS_DIR}/logs"/am01_aae_train_*.log "${HOME_PROJECT}/logs/" 2>/dev/null || true

echo "=== Results fetched to ${HOME_PROJECT}/reports/ ==="
echo "Job completed."
