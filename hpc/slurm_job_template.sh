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

# ── Main execution ──────────────────────────────────────────────────────────

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
        --exclude='data/raw/' \
        --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
        --exclude='outputs/' --exclude='reports/figures/' \
        "${HOME}/am01_project/" "${SCRATCH_DIR}/${SCRATCH_PROJECT}/"
    echo "Synced project to scratch (excluded .venv/.git/data/raw)."
fi

# Activate uv environment — rebuild the venv on scratch from the frozen lockfile.
# `uv run` is network/credential-free once the shared cache is populated.
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

# data/processed/*.npy, *.pkl, *.json
mkdir -p "${HOME_PROJECT}/data/processed"
rsync -a --ignore-existing \
    --include='*.npy' --include='*.pkl' --include='*.json' --exclude='*' \
    "${RESULTS_DIR}/data/processed/" "${HOME_PROJECT}/data/processed/" 2>/dev/null || true

# data/models/*.pth (se esistono)
mkdir -p "${HOME_PROJECT}/data/models"
rsync -a --ignore-existing \
    --include='*.pth' --include='*.pt' --exclude='*' \
    "${RESULTS_DIR}/data/models/" "${HOME_PROJECT}/data/models/" 2>/dev/null || true

# logs/am01_train_*.log
mkdir -p "${HOME_PROJECT}/logs"
cp "${RESULTS_DIR}/logs"/am01_train_*.log "${HOME_PROJECT}/logs/" 2>/dev/null || true

echo "Results fetched to ${HOME_PROJECT}/data/processed/ and ${HOME_PROJECT}/logs/"
