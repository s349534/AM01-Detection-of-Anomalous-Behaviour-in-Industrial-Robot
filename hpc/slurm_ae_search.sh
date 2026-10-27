#!/bin/bash
#
# slurm_ae_search.sh — SLURM batch job for AE hyperparameter validation search (§4.8.5/§4.8.6)
#
# Runs the exhaustive search (all 50 combinations) or random subset over the vanilla Autoencoder, then analyzes results.
# All artefacts (CSV + YAML + plots + log) are written to reports/ and synced back
# to $HOME/am01_project/ so hpc_connect.sh batch can fetch them to your machine.
#
# Customize --n-iter and --max-val-epochs below, or pass via env:
#   MAX_VAL_EPOCHS=50 ./hpc_connect.sh batch hpc/slurm_ae_search.sh          # exhaustive (default)
#   N_ITER=20 MAX_VAL_EPOCHS=50 ./hpc_connect.sh batch hpc/slurm_ae_search.sh # random subset
# ==============================================================================

# ── SLURM Directives ────────────────────────────────────────────────────────
#SBATCH --job-name=am01_ae_search
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 1 task = 1 processo Python (NO MPI)
#SBATCH --cpus-per-task=6            # 6 core per quel task (1 main + 4 worker + 1 margine)
#SBATCH --gres=gpu:1
#SBATCH --time=04:30:00              # 4.5h: 50 configs × 3 seeds = 150 runs con num_workers=4
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

# Resolve N_ITER / MAX_VAL_EPOCHS from env (with defaults)
# N_ITER: if not set, runs exhaustive search (all combinations)
N_ITER="${N_ITER:-}"
MAX_VAL_EPOCHS="${MAX_VAL_EPOCHS:-50}"

echo "=== SLURM job started: $(date) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "N_ITER: ${N_ITER:-exhaustive}"
echo "MAX_VAL_EPOCHS: ${MAX_VAL_EPOCHS}"

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
        --exclude='*.npz' \
        --exclude='outputs/' --exclude='reports/figures/' \
        --exclude='reports/tables/ae_final_metrics.csv' \
        "${HOME}/am01_project/" "${SCRATCH_DIR}/${SCRATCH_PROJECT}/"
    echo "Synced project to scratch."

    # Content-based re-sync of the sources (--checksum): the quick check
    # (size+mtime) above can skip files whose timestamp did not change,
    # leaving a STALE copy of the code on scratch.
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
    echo "--- PyTorch / device check ---"
    uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
fi


# ── Phase 3.1: Validation search (§4.8.6) ───────────────────────────────────
# Preprocessing is ALWAYS executed: data/processed/ may be stale from a previous
# run (different hyperparameters, window size, or scaler), so re-generating it
# guarantees the search step consumes a consistent, up-to-date dataset.
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
uv run python -m src.data.preprocessing --config config/params.yaml

# Build command args for run_search_ae
# --seed is for ParameterSampler sampling only; training seeds are fixed: [7, 42, 123]
# (--val-seeds) per config (§4.8.6)
SEARCH_ARGS=("--seed" "42" "--max-val-epochs" "${MAX_VAL_EPOCHS}" "--no-resume")
if [[ -n "${N_ITER}" ]]; then
    SEARCH_ARGS+=("--n-iter" "${N_ITER}")
    echo "=== Phase 3.1: AE validation search (random ${N_ITER} configs × 3 seeds, max ${MAX_VAL_EPOCHS} epochs) ==="
else
    echo "=== Phase 3.1: AE validation search (EXHAUSTIVE - all 20 configs × 3 seeds = 60 runs, max ${MAX_VAL_EPOCHS} epochs) ==="
fi

uv run python -m src.validation.run_search_ae "${SEARCH_ARGS[@]}"

# ── Phase 3.1b: Analyze results → params_validated_ae.yaml + plots (§4.8.8) ─
echo "=== Phase 3.1b: Analyzing results ==="
uv run python -m src.validation.analyze_results

# ── Verify outputs ──────────────────────────────────────────────────────────
echo "=== Outputs ==="
ls -lh reports/tables/validation_results_ae.csv reports/figures/sensitivity_*.png config/params_validated_ae.yaml 2>/dev/null || true

# ── Fetch results back to $HOME (so hpc_connect.sh batch can download) ──────
RESULTS_DIR="${SCRATCH_DIR}/${SCRATCH_PROJECT}"
HOME_PROJECT="${HOME}/am01_project"

mkdir -p "${HOME_PROJECT}/reports/tables" "${HOME_PROJECT}/reports/figures" "${HOME_PROJECT}/config"
mkdir -p "${HOME_PROJECT}/logs"

# Remove stale training residue from previous runs.
# Validation phase does NOT produce ae_final_metrics.csv — clear it from BOTH
# scratch (so it doesn't sync back to home) and home (the source of truth for
# the local download).
rm -f "${RESULTS_DIR}/reports/tables/ae_final_metrics.csv" \
      "${HOME_PROJECT}/reports/tables/ae_final_metrics.csv" 2>/dev/null || true

# CSV + YAML + plots --delete: remove stale files not present in this run's output.
rsync -a --delete \
    "${RESULTS_DIR}/reports/tables/" "${HOME_PROJECT}/reports/tables/" 2>/dev/null || true
rsync -a --delete \
    "${RESULTS_DIR}/reports/figures/" "${HOME_PROJECT}/reports/figures/" 2>/dev/null || true

# params_validated_ae.yaml (produced by analyze_results)
rsync -a \
    "${RESULTS_DIR}/config/" "${HOME_PROJECT}/config/" 2>/dev/null || true

# data/processed/ — sync from scratch to home so cmd_batch can download
rsync -a --include='*.npy' --include='*.pkl' --include='*.json' --exclude='*' \
    "${RESULTS_DIR}/data/processed/" "${HOME_PROJECT}/data/processed/" 2>/dev/null || true

# Log — SLURM writes to ~/jobs/logs/ (relative to sbatch submission dir), not to scratch
cp ~/jobs/logs/am01_ae_search_*.log "${HOME_PROJECT}/logs/" 2>/dev/null || true

echo "=== Results fetched to ${HOME_PROJECT}/reports/ and ${HOME_PROJECT}/config/ ==="
echo "Job completed."
