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

echo "=== SLURM job started: $(date) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "SEEDS: ${SEEDS:-default}"
echo "MAX_EPOCHS: ${MAX_EPOCHS:-default}"

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
# Preprocess if data/processed/ is missing OR incomplete OR validation set too small
NEED_PREPROCESS=0
# List of ALL files that must exist after correct preprocessing
REQUIRED_PROCESSED_FILES=(
    "data/processed/train.npy"
    "data/processed/val.npy"
    "data/processed/test_normal.npy"
    "data/processed/test_anomaly.npy"
    "data/processed/scaler.pkl"
    "data/processed/selected_columns.npy"
)

# Check if ALL required files exist
ALL_EXIST=1
for f in "${REQUIRED_PROCESSED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "=== Missing processed file: $f — running preprocessing ==="
        ALL_EXIST=0
        NEED_PREPROCESS=1
        break
    fi
done

if [[ ${ALL_EXIST} -eq 1 ]]; then
    # Check validation set size (need at least window_size+1 samples)
    VAL_SAMPLES=$(uv run python -c 'import numpy as np, sys;
try:
    arr = np.load("data/processed/val.npy", mmap_mode="r")
    print(arr.shape[0])
except Exception:
    print(0)
' 2>/dev/null || echo 0)
    if [[ ${VAL_SAMPLES:-0} -lt 33 ]]; then
        echo "=== Validation set too small (${VAL_SAMPLES} samples, need >=33) — re-running preprocessing ==="
        NEED_PREPROCESS=1
    fi
fi

if [[ ${NEED_PREPROCESS} -eq 1 ]]; then
    # Verify raw data exists and has expected size before attempting preprocessing
    if [[ ! -f "data/raw/KukaNormal.npy" || ! -f "data/raw/KukaSlow.npy" || ! -f "data/raw/KukaColumnNames.npy" ]]; then
        echo "ERROR: Raw data files missing in data/raw/!"
        echo "  Required: KukaNormal.npy, KukaSlow.npy, KukaColumnNames.npy"
        echo "  Run ./hpc_connect.sh deploy first to upload them, or upload manually:"
        echo "  scp data/raw/*.npy polito-hpc:~/am01_project/data/raw/"
        exit 1
    fi
    # Log raw data shapes for debugging
    echo "=== Raw data verification ==="
    uv run python -c 'import numpy as np
normal = np.load("data/raw/KukaNormal.npy", mmap_mode="r")
slow = np.load("data/raw/KukaSlow.npy", mmap_mode="r")
cols = np.load("data/raw/KukaColumnNames.npy", allow_pickle=True)
print(f"KukaNormal: {normal.shape}")
print(f"KukaSlow: {slow.shape}")
print(f"ColumnNames: {cols.shape}")
'
    uv run python -m src.data.preprocessing
fi

# Resolve SEEDS default BEFORE first use (fixes "unbound variable" error)
SEEDS="${SEEDS:-42 123 7}"
MAX_EPOCHS="${MAX_EPOCHS:-}"

echo "=== Phase 3.2: Final AE training (seeds=${SEEDS}) ==="

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
