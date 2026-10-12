#!/bin/bash
#
# slurm_aae_search.sh — SLURM batch job for AAE hyperparameter validation search (§4.8.5/§4.8.6)
#
# Runs the full random search over the Adversarial Autoencoder, then analyzes results.
# All artefacts (CSV + YAML + plots + log) are written to reports/ and synced back
# to $HOME/am01_project/ so hpc_connect.sh batch can fetch them to your machine.
#
# Customize --n-iter and --max-val-epochs below, or pass via env:
#   N_ITER=30 MAX_VAL_EPOCHS=50 ./hpc_connect.sh batch hpc/slurm_aae_search.sh
# ==============================================================================

# ── SLURM Directives ────────────────────────────────────────────────────────
#SBATCH --job-name=am01_aae_search
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

# Resolve N_ITER / MAX_VAL_EPOCHS from env (with defaults)
N_ITER="${N_ITER:-25}"
MAX_VAL_EPOCHS="${MAX_VAL_EPOCHS:-50}"

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
        --exclude='outputs/' --exclude='reports/figures/' \
        "${HOME}/am01_project/" "${SCRATCH_DIR}/${SCRATCH_PROJECT}/"
    echo "Synced project to scratch."
fi

# ── Activate uv environment ─────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    echo "--- Syncing frozen deps on scratch ---"
    uv sync --frozen --quiet
    echo "--- PyTorch / device check ---"
    uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
fi

# ── Phase 4.1: AAE Validation search (§4.8.6) ───────────────────────────────
echo "=== Phase 4.1: AAE validation search (${N_ITER} iterations, max ${MAX_VAL_EPOCHS} epochs) ==="
uv run python -m src.validation.run_search_aae \
    --n-iter "${N_ITER}" \
    --seed 42 \
    --max-val-epochs "${MAX_VAL_EPOCHS}"

# ── Phase 4.1b: Analyze results → params_validated_aae.yaml + plots (§4.8.8) ─
echo "=== Phase 4.1b: Analyzing AAE results ==="
uv run python -m src.validation.analyze_results --aae

# ── Verify outputs ──────────────────────────────────────────────────────────
echo "=== Outputs ==="
ls -lh reports/tables/validation_results_aae.csv reports/figures/sensitivity_aae_*.png config/params_validated_aae.yaml 2>/dev/null || true

# ── Fetch results back to $HOME (so hpc_connect.sh batch can download) ──────
RESULTS_DIR="${SCRATCH_DIR}/${SCRATCH_PROJECT}"
HOME_PROJECT="${HOME}/am01_project"

mkdir -p "${HOME_PROJECT}/reports/tables" "${HOME_PROJECT}/reports/figures" "${HOME_PROJECT}/config"
mkdir -p "${HOME_PROJECT}/logs"

# CSV + YAML + plots
rsync -a --include='validation_results_aae.csv' --include='aae_final_metrics.csv' \
          --include='params_validated_aae.yaml' \
          --include='sensitivity_aae_*.png' --include='*.png' \
          --include='*' --exclude='*' \
    "${RESULTS_DIR}/reports/" "${HOME_PROJECT}/reports/" 2>/dev/null || true

rsync -a --include='params_validated_aae.yaml' \
          --include='*.yaml' --include='*' --exclude='*' \
    "${RESULTS_DIR}/config/" "${HOME_PROJECT}/config/" 2>/dev/null || true

# Log
cp "${RESULTS_DIR}/logs"/am01_aae_search_*.log "${HOME_PROJECT}/logs/" 2>/dev/null || true

echo "=== Results fetched to ${HOME_PROJECT}/reports/ and ${HOME_PROJECT}/config/ ==="
echo "Job completed."
