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
#SBATCH --gres=gpu:1
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
export SCRATCH_PROJECT="am01"
echo "Allocated GPU (SLURM GRES): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# The project is an APPLICATION run from the project root (`python -m src...`):
# it is not installed into the venv. Building it would require downloading
# setuptools from PyPI, which fails on the cluster (no PyPI access).
# UV_NO_SYNC=1 keeps the `uv run` calls from re-syncing (and re-building).
export UV_NO_SYNC=1

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

    # ── Content-based re-sync of the sources (--checksum) ───────────────────
    # The quick check above (size+mtime) can skip files whose timestamp did
    # not change, leaving a STALE copy on scratch. This guarantees the code
    # that runs is byte-identical to the one deployed in ~/am01_project.
    rsync -a --checksum \
        --include='*/' --include='*.py' --include='*.yaml' --include='*.sh' --exclude='*' \
        "${HOME}/am01_project/src/" "${SCRATCH_DIR}/${SCRATCH_PROJECT}/src/"
    rsync -a --checksum \
        --include='*/' --include='*.yaml' --include='*.sh' --exclude='*' \
        "${HOME}/am01_project/config/" "${SCRATCH_DIR}/${SCRATCH_PROJECT}/config/"
    echo "Sources re-synced with --checksum."
fi

# ── Activate uv environment ─────────────────────────────────────────────────
# Clear any stale VIRTUAL_ENV inherited from a remote .bashrc (setup_env.sh
# auto-activates ~/am01_project/.venv, which is wrong on scratch)
unset VIRTUAL_ENV 2>/dev/null || true

if command -v uv &>/dev/null; then
    echo "--- Syncing frozen deps on scratch ---"
    uv sync --frozen --no-install-project --quiet
fi



# ── Phase 3.2: Final AE training with validated HPs (3 seeds) ───────────────
# Preprocessing is ALWAYS executed: data/processed/ may be stale from a previous
# run (different hyperparameters, window size, or scaler), so re-generating it
# guarantees the training step consumes a consistent, up-to-date dataset.
# Verify raw data exists before attempting preprocessing
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
uv run python -m src.data.preprocessing --config config/params_validated_ae.yaml

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

# ── Phase 3.3: Evaluation sul test set ─────────────────────────────────────────
# Evaluation is already done inside train_ae.py (test metrics + confusion matrix).
# No separate evaluation CLI call needed — avoids duplicate output.

# ── Verify outputs ─────────────────────────────────────────────────────────────
echo "=== Outputs ==="
ls -lh reports/checkpoints/ae_baseline.pth reports/tables/ae_final_metrics.csv reports/figures/ae_confusion_matrix.png 2>/dev/null || true

# ── Fetch results back to $HOME ─────────────────────────────────────────────
RESULTS_DIR="${SCRATCH_DIR}/${SCRATCH_PROJECT}"
HOME_PROJECT="${HOME}/am01_project"

# ── Debug: verify figures exist before sync ────────────────────────────────────
echo "=== Debug: figures on scratch ==="
ls -la "${RESULTS_DIR}/reports/figures/" 2>/dev/null || echo "No figures dir on scratch"

mkdir -p "${HOME_PROJECT}/reports/checkpoints" "${HOME_PROJECT}/reports/tables" \
         "${HOME_PROJECT}/reports/figures" "${HOME_PROJECT}/logs"

# No --ignore-existing: new results must overwrite stale ones
# --delete ensures stale artifacts are removed
# Use --info=progress2 for visibility, no head truncation
rsync -a --delete --info=progress2 \
    "${RESULTS_DIR}/reports/checkpoints/" "${HOME_PROJECT}/reports/checkpoints/"
rsync -a --delete --info=progress2 \
    "${RESULTS_DIR}/reports/tables/" "${HOME_PROJECT}/reports/tables/"
rsync -a --delete --info=progress2 \
    "${RESULTS_DIR}/reports/figures/" "${HOME_PROJECT}/reports/figures/"

# Sync processed data
rsync -a --delete --info=progress2 --include='*.npy' --include='*.pkl' --include='*.json' --exclude='*' \
    "${RESULTS_DIR}/data/processed/" "${HOME_PROJECT}/data/processed/"

# Log — SLURM writes to ~/jobs/logs/ (relative to sbatch submission dir), not to scratch
cp ~/jobs/logs/am01_ae_train_*.log "${HOME_PROJECT}/logs/" 2>/dev/null || true

# Debug: verify figures synced to home
echo "=== Debug: figures on home ==="
ls -la "${HOME_PROJECT}/reports/figures/" 2>/dev/null || echo "No figures dir on home"

echo "=== Results fetched to ${HOME_PROJECT}/reports/ ==="
echo "Job completed."

