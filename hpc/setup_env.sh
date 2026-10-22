#!/bin/bash
#
# setup_env.sh — One-time environment setup on PolitO HPC Legion
#
# This script should be run on the HPC login node to:
#   1. Install uv (if not present)
#   2. Create a virtual environment
#   3. Sync project dependencies from pyproject.toml
#   4. Register Jupyter kernel
#
# Prerequisites:
#   - SSH access to HPC (ssh polito-hpc)
#   - Project code on HPC (upload first: ./hpc_connect.sh upload . ~/am01_project)
#   - Network access to PyPI (from compute nodes or login nodes)
#
# Usage:
#   # Option A: Run remotely via hpc_connect.sh
#   ./hpc_connect.sh exec "bash -s" < setup_env.sh
#
#   # Option B: Upload and run on HPC
#   ./hpc_connect.sh upload hpc/setup_env.sh ~/
#   ./hpc_connect.sh exec "cd ~ && bash setup_env.sh"
#
#   # Option C: Manual run on HPC
#   ssh polito-hpc
#   cd ~/am01_project
#   bash hpc/setup_env.sh

set -euo pipefail

# ── Project directory (upload the repo to this location on the HPC) ──────────
# NOTE: this script runs *on the HPC*, so $HOME resolves to the remote home
# (e.g. /home/mungolo). Do NOT pass ~/am01_project literally from the local
# machine — use an absolute remote path or let $HOME expand here.
PROJECT_DIR="${PROJECT_DIR:-$HOME/am01_project}"
echo "Project directory: ${PROJECT_DIR}"
# `cd` must NOT quote ~ ; use $HOME (already expanded) for portability.
cd "${PROJECT_DIR}" || { echo "ERROR: project dir not found: ${PROJECT_DIR}"; exit 1; }

echo "=========================================="
echo "  AM01 HPC Environment Setup"
echo "  PolitO Legion Cluster"
echo "=========================================="
echo ""

# ── Step 1: Check system info ───────────────────────────────────────────────
echo "--- System Information ---"
echo "User:  $(whoami)"
echo "Host:  $(hostname)"
echo "Home:  $HOME"
echo "SCRATCH: ${SCRATCH:-<only on compute nodes>}"
echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"
echo "uv:   $(uv --version 2>/dev/null || echo 'not found')"
echo ""

# ── Step 2: Install uv if not present ───────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "--- Installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
else
    echo "uv already installed: $(uv --version)"
fi

# ── Step 3: Set Python version ──────────────────────────────────────────────
# Use .python-version from the project (3.13). uv will fetch the exact build.
PYTHON_VERSION_FILE=".python-version"
if [[ -f "${PYTHON_VERSION_FILE}" ]]; then
    DESIRED_PYTHON=$(cat "${PYTHON_VERSION_FILE}")
    echo "--- Setting Python version to ${DESIRED_PYTHON} ---"
    uv python find "${DESIRED_PYTHON}" >/dev/null 2>&1 || {
        echo "Python ${DESIRED_PYTHON} not found. Installing via uv..."
        uv python install "${DESIRED_PYTHON}"
    }
fi

# ── Step 4: Sync dependencies from the frozen lockfile ──────────────────────
echo ""
echo "--- Syncing project dependencies (frozen lockfile) ---"
uv sync --frozen
# Re-export pinned requirements.txt for any pip-based fallback
uv export --no-dev --no-hashes -o requirements.txt 2>/dev/null || true

# ── Step 5: Register Jupyter kernel ─────────────────────────────────────────
echo ""
echo "--- Registering Jupyter kernel ---"
uv run python -m ipykernel install --user --name "am01-hpc" --display-name "AM01 (HPC)"
echo "Jupyter kernel 'am01-hpc' registered."

# ── Step 6: Verify ──────────────────────────────────────────────────────────
echo ""
echo "--- Verification ---"
uv run python -c "
import numpy, pandas, sklearn, torch
print(f'numpy={numpy.__version__}')
print(f'pandas={pandas.__version__}')
print(f'sklearn={sklearn.__version__}')
print(f'torch={torch.__version__}')
from src.data import dataset, preprocessing
from src.models import autoencoder, adversarial_ae
from src.utils import metrics, visualization
print('All project modules import successfully!')
"

echo ""
# ── Step 7.5: Check for data/raw/ (needed by preprocessing) ────────────────
if [[ ! -d "${PROJECT_DIR}/data/raw" ]]; then
    echo ""
    echo "WARNING: data/raw/ not found on cluster."
    echo "  Preprocesing will fail without the Kuka .npy files."
    echo "  Upload once:  ./hpc_connect.sh upload data/raw/ ~/am01_project/data/raw/"
else
    echo "data/raw/ present — preprocessing can run."
fi

echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run './hpc_connect.sh interactive' for a compute node session"
echo "  2. On compute node, cd to \$SCRATCH and run your training"
echo "  3. Or one-shot: ./hpc_connect.sh batch hpc/slurm_job_template.sh"
echo ""
echo "BeeGFS scratch path: \${SCRATCH:-Not available (need compute node)}"
echo "Project directory:   ${PROJECT_DIR}"
echo "Home directory:      $HOME"

# ── Step 7: Persist uv on PATH for future non-interactive SLURM jobs ─────────
# SLURM batch jobs do not source .bashrc interactively; make sure uv and the
# venv are available so `uv run` works inside sbatch scripts.
if grep -q "AM01 / uv environment" "$HOME/.bashrc" 2>/dev/null; then
    echo "AM01 uv environment block already in ~/.bashrc — skipping (no duplicates)."
else
    cat >> "$HOME/.bashrc" <<EOF

# >>> AM01 / uv environment >>>
export PATH="\$HOME/.local/bin:\$PATH"
[ -d "\$HOME/am01_project/.venv" ] && source "\$HOME/am01_project/.venv/bin/activate" 2>/dev/null || true
# <<< AM01 / uv environment <<<
EOF
    echo "Added uv path + venv sourcing to ~/.bashrc for batch jobs."
fi
