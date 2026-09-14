#!/bin/bash
#
# slurm_ae_train.sh — SLURM batch job for final AE training (§3.2)
#
# Runs the final training with validated hyperparameters across 3 seeds,
# saves the baseline checkpoint, and writes ae_final_metrics.csv.
#
# Usage:
#   SEEDS="42 123 7" ./hpc_connect.sh batch hpc/slurm_ae_train.sh
# ==============================================================================

# ── SLURM Directives ────────────────────────────────────────────────────────
#SBATCH --job-name=am01_ae_train
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
        --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
        "${HOME}/am01_project/" "${SCRATCH_DIR}/${SCRATCH_PROJECT}/"
    echo "Synced project to scratch."
fi

# ── Activate uv environment ─────────────────────────────────────────────────
# Clear any stale VIRTUAL_ENV inherited from a remote .bashrc (setup_env.sh
# auto-activates ~/am01_project/.venv, which is wrong on scratch)
unset VIRTUAL_ENV 2>/dev/null || true

if command -v uv &>/dev/null; then
    echo "--- Syncing frozen deps on scratch ---"
    uv sync --frozen --quiet
fi

# ── Phase 3.2: Final AE training with validated HPs (3 seeds) ───────────────
# Preprocess if data/processed/ is missing (first run on a fresh scratch sync)
if [[ ! -f "data/processed/train.npy" ]]; then
    echo "=== data/processed/ missing — running preprocessing from data/raw/ ==="
    uv run python -m src.data.preprocessing
fi

echo "=== Phase 3.2: Final AE training (seeds=${SEEDS}) ==="
SEEDS="${SEEDS:-42 123 7}"
MAX_EPOCHS="${MAX_EPOCHS:-}"

TRAIN_CMD="uv run python -m src.models.train_ae \
    --config config/params_validated_ae.yaml \
    --seeds ${SEEDS}"
if [[ -n "${MAX_EPOCHS}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --max-epochs ${MAX_EPOCHS}"
fi
eval "${TRAIN_CMD}"

# ── Verify outputs ──────────────────────────────────────────────────────────
echo "=== Outputs ==="
ls -lh reports/checkpoints/ae_baseline.pth reports/tables/ae_final_metrics.csv 2>/dev/null || true

# ── Fetch results back to $HOME ─────────────────────────────────────────────
RESULTS_DIR="${SCRATCH_DIR}/${SCRATCH_PROJECT}"
HOME_PROJECT="${HOME}/am01_project"

mkdir -p "${HOME_PROJECT}/reports/checkpoints" "${HOME_PROJECT}/reports/tables" "${HOME_PROJECT}/logs"

# No --ignore-existing: new results must overwrite stale ones
rsync -a \
    "${RESULTS_DIR}/reports/checkpoints/" "${HOME_PROJECT}/reports/checkpoints/" 2>/dev/null || true
rsync -a \
    "${RESULTS_DIR}/reports/tables/" "${HOME_PROJECT}/reports/tables/" 2>/dev/null || true

# Log — SLURM writes to ~/jobs/logs/ (relative to sbatch submission dir), not to scratch
cp ~/jobs/logs/am01_ae_train_*.log "${HOME_PROJECT}/logs/" 2>/dev/null || true

echo "=== Results fetched to ${HOME_PROJECT}/reports/ ==="
echo "Job completed."
