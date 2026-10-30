#!/bin/bash
#
# slurm_job_template.sh — SLURM batch job template for AM01 on PolitO HPC Legion
#
# Usage:
#   # 1. Customize the #SBATCH directives and command below
#   # 2. Upload and submit:
#   ./hpc_connect.sh submit hpc/slurm_job_template.sh
#
# Partitions available on Legion:
#   cpu_sapphire, cpu_sapphire_ext  — CPU nodes
#   gpu_a40, gpu_a40_ext            — NVIDIA A40 (108 GPUs)
#   gpu_a100                        — NVIDIA A100 (4 GPUs)
#   gpu_h200                        — NVIDIA H200
#   cpu_skylake, cpu_skylake_ext    — Legacy CPU nodes
#   gpu_v100, gpu_v100_ext          — Legacy V100 GPUs
#
# *_ext partitions are for external users / VPN connections
# ==============================================================================

# ── SLURM Directives ────────────────────────────────────────────────────────
# Job name (shown in squeue)
#SBATCH --job-name=am01_train

# Partition / queue
#SBATCH --partition=gpu_a40

# Number of nodes
#SBATCH --nodes=1

# Number of tasks (CPU cores) — adjust per partition
#SBATCH --ntasks-per-node=8

# Number of GPUs (only for GPU partitions)
#SBATCH --gpus=1

# Walltime limit (HH:MM:SS) — keep generous for training, trim for smoke tests
#SBATCH --time=00:30:00

# Memory per node (default is usually fine)
#SBATCH --mem=64GB

# Output and error logs — unified into a single .log file so that print()
# and logger.info() appear together in order, with severity tags from the
# logging format configured in src/main.py (stream=sys.stdout).
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# Mail notifications (optional)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mungolo@studenti.polito.it

# ── Environment setup ───────────────────────────────────────────────────────
set -euo pipefail

# Load modules (if needed)
module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

# Set up environment
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1   # flush stdout subito → `tail -f` del log mostra l'output riga per riga
export CUDA_VISIBLE_DEVICES=0
export SCRATCH_PROJECT="am01"
# The project is an APPLICATION run from the project root (`python -m src...`):
# it is not installed into the venv. Building it would require downloading
# setuptools from PyPI, which fails on the cluster (no PyPI access).
# UV_NO_SYNC=1 keeps the `uv run` calls from re-syncing (and re-building).
export UV_NO_SYNC=1

# ── Main execution ────────────────────────────────────────────────────────

# On compute nodes, BeeGFS scratch ($SCRATCH) is set; on the login node (or some
# batch contexts) it is NOT exported, so guard it under `set -u` — this fixes
# `slurm_script: line N: SCRATCH: unbound variable`.
SCRATCH_DIR="${SCRATCH:-${HOME}/scratch}"
mkdir -p "${SCRATCH_DIR}/${SCRATCH_PROJECT}"
cd "${SCRATCH_DIR}/${SCRATCH_PROJECT}"
echo "Running on scratch: $(pwd)  (SCRATCH=${SCRATCH:-<unset, used fallback ${SCRATCH_DIR}})"

# Create logs directory early (also used by src/main.py for run artefacts)
mkdir -p "$(pwd)/logs"

# Sync project code from home to scratch — exclude heavy/secret dirs to keep
# the transfer small. The uv venv is NOT synced: it is rebuilt below with
# `uv sync --frozen` from the shared uv cache (~/.cache/uv), which is fast.
if [[ -d "${HOME}/am01_project" ]]; then
    rsync -a \
        --exclude='.venv/' \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='.ipynb_checkpoints/' \
        --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
        --exclude='outputs/' --exclude='reports/figures/' \
        "${HOME}/am01_project/" "${SCRATCH_DIR}/${SCRATCH_PROJECT}/"
    echo "Synced project to scratch (excluded .venv/.git/data/raw)."

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

# Activate uv environment — rebuild the venv on scratch from the frozen lockfile.
# `uv run` is network/credential-free once the shared cache is populated.
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

# ── Preprocessing (always executed) ──────────────────────────────────────────
# data/processed/ may be stale from a previous run (different hyperparameters,
# window size, or scaler), so re-generate it every time to guarantee consistency.
echo "=== Running preprocessing from data/raw/ ==="
uv run python -m src.data.preprocessing --config config/params.yaml

# ── Run your training script ────────────────────────────────────────────────
# Replace with your actual training command when the pipeline is implemented.
echo "Starting AM01 training..."
uv run python src/main.py --config config/config.yaml

echo "Job completed successfully."

# ── Fetch automatico dei risultati su $HOME (visibile da login node) ──────
# Copia data/processed/*.npy, *.pkl e il log su $HOME/am01_project/ così
# ``hpc_connect.sh`` può scaricarli localmente con un solo comando.
RESULTS_DIR="${SCRATCH_DIR}/${SCRATCH_PROJECT}"
HOME_PROJECT="${HOME}/am01_project"

# data/processed/*.npy, *.pkl, *.json (no --ignore-existing: overwrite stale)
mkdir -p "${HOME_PROJECT}/data/processed"
rsync -a --info=progress2 \
    --include='*.npy' --include='*.pkl' --include='*.json' --exclude='*' \
    "${RESULTS_DIR}/data/processed/" "${HOME_PROJECT}/data/processed/"

# data/models/*.pth (se esistono)
mkdir -p "${HOME_PROJECT}/data/models"
rsync -a --info=progress2 \
    --include='*.pth' --include='*.pt' --exclude='*' \
    "${RESULTS_DIR}/data/models/" "${HOME_PROJECT}/data/models/"

# reports/figures/*.png (confusion matrix, sensitivity plots, etc.)
mkdir -p "${HOME_PROJECT}/reports/figures"
rsync -a --info=progress2 \
    --include='*.png' --exclude='*' \
    "${RESULTS_DIR}/reports/figures/" "${HOME_PROJECT}/reports/figures/"

# reports/tables/*.csv (validation results, final metrics)
mkdir -p "${HOME_PROJECT}/reports/tables"
rsync -a --info=progress2 \
    --include='*.csv' --exclude='*' \
    "${RESULTS_DIR}/reports/tables/" "${HOME_PROJECT}/reports/tables/"

# reports/checkpoints/*.pth
mkdir -p "${HOME_PROJECT}/reports/checkpoints"
rsync -a --info=progress2 \
    --include='*.pth' --exclude='*' \
    "${RESULTS_DIR}/reports/checkpoints/" "${HOME_PROJECT}/reports/checkpoints/"

# Log — SLURM writes to ~/jobs/logs/ (relative to sbatch submission dir), not to scratch
mkdir -p "${HOME_PROJECT}/logs"
cp ~/jobs/logs/am01_train_*.log "${HOME_PROJECT}/logs/" 2>/dev/null || true

echo "Results fetched to ${HOME_PROJECT}/data/processed/, ${HOME_PROJECT}/reports/ and ${HOME_PROJECT}/logs/"