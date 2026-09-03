#!/bin/bash
#
# hpc_connect.sh — Automated SSH connection and command execution for PolitO HPC Legion
#
# Prerequisites:
#   - SSH key already configured in ~/.ssh/config (Host: polito-hpc)
#   - SSH key has no passphrase (or use ssh-agent)
#   - VPN active if connecting from outside PoliTO network
#
# Usage:
#   ./hpc_connect.sh connect                    # Interactive SSH session to login node
#   ./hpc_connect.sh exec "command"             # Run a single command non-interactively
#   ./hpc_connect.sh interactive [partition] [walltime]   # Interactive SLURM session (GPU)
#   ./hpc_connect.sh upload <local> <remote>    # scp upload to HPC
#   ./hpc_connect.sh upload-project             # rsync entire project to HPC (excludes .venv/.git)
#   ./hpc_connect.sh download <remote> <local>  # scp download from HPC
#   ./hpc_connect.sh submit <slurm_script>      # Submit a SLURM batch job
#   ./hpc_connect.sh batch <slurm_script>       # Deploy + submit + stream log + auto-fetch results
#   ./hpc_connect.sh deploy                     # Upload-project + one-shot environment setup
#   ./hpc_connect.sh scratch                    # SSH to compute node (via srun) to access $SCRATCH
#   ./hpc_connect.sh check                      # Print connectivity and environment info
#   ./hpc_connect.sh keycheck                   # Verify key-based auth (no password prompt)
#   ./hpc_connect.sh keycopy                    # Register public key via ssh-copy-id (needs password)
#
# Environment variables:
#   HPC_USER           — Override SSH user (default: mungolo)
#   HPC_HOST           — Override SSH host alias (default: polito-hpc)
#   PROJECT_DIR_REMOTE — Remote project path (default: ~/am01_project)
#   SSH_PASSWORD       — If set, use sshpass for password auth (requires sshpass)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
HPC_USER="${HPC_USER:-mungolo}"
HPC_HOST="${HPC_HOST:-polito-hpc}"
# Use $HOME literal (NOT $HOME expanded locally, NOT ~/ which POSIX sh
# inside ssh neutralises when single-quoted) so the REMOTE shell expands it
# to the HPC home (e.g. /home/mungolo), never the local /c/Users/... path.
PROJECT_DIR_REMOTE="${PROJECT_DIR_REMOTE:-\$HOME/am01_project}"
SSH_OPTS="-o StrictHostKeyChecking=yes -o ConnectTimeout=10"
# Things never to upload to the HPC (large / secret / local-only)
RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='.ipynb_checkpoints/'
    --exclude='data/raw/'
    --exclude='*.pth'
    --exclude='*.pt'
    --exclude='*.ckpt'
    --exclude='outputs/'
    --exclude='reports/figures/'
)

# ── Resolve paths relative to this script & project root (cwd-independent)
# hpc_connect.sh lives in <root>/hpc, so project root is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Helper: build SSH host string ───────────────────────────────────────────────
ssh_target() {
    # If HPC_HOST is an alias in ~/.ssh/config, just use it
    if grep -q "^Host ${HPC_HOST}$" ~/.ssh/config 2>/dev/null; then
        echo "${HPC_HOST}"
    else
        echo "${HPC_USER}@${HPC_HOST}"
    fi
}

# ── Helper: decide sshpass vs direct ssh ─────────────────────────────────────
# Uses key-based auth by default (no passphrase key). Set SSH_PASSWORD to fall
# back to sshpass for password auth (requires sshpass installed locally).
ssh_run() {
    if [[ -n "${SSH_PASSWORD:-}" ]]; then
        if ! command -v sshpass >/dev/null 2>&1; then
            echo "ERROR: SSH_PASSWORD set but 'sshpass' is not installed." >&2
            echo "Either remove the passphrase from your key or install sshpass." >&2
            exit 1
        fi
        sshpass -o "${SSH_PASSWORD}" ssh ${SSH_OPTS} "$@"
    else
        ssh ${SSH_OPTS} "$@"
    fi
}

# ── Commands ────────────────────────────────────────────────────────────────

cmd_connect() {
    echo "Connecting to PolitO HPC Legion (login node)..."
    echo "NOTE: BeeGFS scratch (\$SCRATCH) is only accessible from compute nodes."
    echo "      Use 'interactive' to get a compute node session."
    echo ""
    ssh_run "$(ssh_target)"
}

cmd_exec() {
    local command="$1"
    echo "Executing on HPC: ${command}"
    ssh_run "$(ssh_target)" "${command}"
}

cmd_interactive() {
    local partition="${2:-gpu_a40}"
    local time="${3:-04:00:00}"
    echo "Requesting interactive SLURM session..."
    echo "  Partition: ${partition}"
    echo "  Walltime:  ${time}"
    echo ""
    echo "Once on the compute node, you can access \$SCRATCH (BeeGFS)."
    ssh_run "$(ssh_target)" "srun --pty -p ${partition} --time=${time} bash"
}

cmd_scratch() {
    echo "Requesting compute node session for BeeGFS scratch access..."
    ssh_run "$(ssh_target)" 'srun --pty -p cpu_sapphire_ext --time=04:00:00 bash -c "echo \"Connected to compute node. SCRATCH=\${SCRATCH}\"; cd \${SCRATCH} && exec bash"'
}

cmd_upload() {
    local local_path="$1"
    local remote_path="$2"
    echo "Uploading ${local_path} -> ${HPC_HOST}:${remote_path}"
    ssh_run "$(ssh_target)" "mkdir -p $(dirname "${remote_path}")" 2>/dev/null || true
    if [[ -n "${SSH_PASSWORD:-}" ]]; then
        sshpass -o "${SSH_PASSWORD}" scp -r "${local_path}" "${HPC_HOST}:${remote_path}"
    else
        scp -r "${local_path}" "$(ssh_target):${remote_path}"
    fi
    echo "Upload complete."
}

cmd_download() {
    local remote_path="$1"
    local local_path="$2"
    echo "Downloading ${HPC_HOST}:${remote_path} -> ${local_path}"
    if [[ -n "${SSH_PASSWORD:-}" ]]; then
        sshpass -o "${SSH_PASSWORD}" scp -r "${HPC_HOST}:${remote_path}" "${local_path}"
    else
        scp -r "$(ssh_target):${remote_path}" "${local_path}"
    fi
    echo "Download complete."
}

cmd_submit() {
    local script="$1"
    if [[ ! -f "${script}" ]]; then
        echo "ERROR: SLURM script not found: ${script}"
        exit 1
    fi
    echo "Uploading and submitting SLURM job: ${script}"
    local basename_script
    basename_script=$(basename "${script}")
    # SLURM does NOT create the --output log directory itself; create ~/jobs/logs
    # on the login node (where sbatch runs) BEFORE submitting, otherwise the job
    # fails silently with no log files written.
    ssh_run "$(ssh_target)" "mkdir -p ~/jobs/logs"
    scp "${script}" "$(ssh_target):~/jobs/${basename_script}"
    ssh_run "$(ssh_target)" "cd ~/jobs && sbatch ${basename_script}"
}

cmd_check() {
    echo "=== HPC Connection Check ==="
    echo "SSH target:    $(ssh_target)"
    echo "SSH key:       ~/.ssh/id_ed25519"
    echo "SSH config:    ~/.ssh/config (Host: ${HPC_HOST})"
    if [[ -n "${SSH_PASSWORD:-}" ]]; then
        echo "Auth method:   password (via sshpass)"
    else
        echo "Auth method:   key-based (no passphrase)"
    fi
    echo ""
    echo "Testing SSH connectivity..."
    ssh_run "$(ssh_target)" 'echo "SSH connection OK"; echo; echo "=== Remote Environment ==="; echo "User: $(whoami)"; echo "Host: $(hostname)"; echo "Home: $HOME"; echo "SCRATCH: ${SCRATCH:-<only on compute nodes>}"; echo; echo "=== uv (if installed) ==="; uv --version 2>/dev/null || echo "(uv will be installed by setup_env.sh)"; echo; echo "=== SLURM Partitions ==="; sinfo -s 2>/dev/null || echo "(sinfo not available on login node)"'
}

cmd_keycheck() {
    echo "=== SSH key auth check (BatchMode, no prompt) ==="
    local target
    target=$(ssh_target)
    if [[ "${target}" == "${HPC_HOST}" ]]; then
        # Alias: test via alias with -o BatchMode
        ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=10 "${target}" 'echo "key-auth OK as $(whoami) on $(hostname)"' 2>&1 || {
            echo "FAIL: key-based auth did not succeed (or off-network / no VPN)." >&2
            return 1
        }
    else
        ssh ${SSH_OPTS} -o BatchMode=yes "${target}" 'echo "key-auth OK as $(whoami) on $(hostname)"' 2>&1 || {
            echo "FAIL: key-based auth did not succeed (or off-network / no VPN)." >&2
            return 1
        }
    fi
}

# ── First-time key registration via ssh-copy-id (prompts for your PoliTO password)
cmd_keycopy() {
    echo "=== Copying public key to HPC (ssh-copy-id) ==="
    echo "Public key: $(ssh-keygen -l -f ~/.ssh/id_ed25519 2>/dev/null | awk '{print $2}')"
    echo "You will be prompted for your PoliTO account password ONCE."
    echo ""
    local target
    target=$(ssh_target)
    if [[ "${target}" == "${HPC_HOST}" ]]; then
        ssh-copy-id -i ~/.ssh/id_ed25519.pub "${target}"
    else
        ssh-copy-id -i ~/.ssh/id_ed25519.pub "${HPC_USER}@${HPC_HOST}"
    fi
    echo ""
    echo "Key registered. Verify with: ./hpc_connect.sh keycheck"
}

# ── Upload the whole project (idempotent: rsync, excludes large/secret) ──────
#   usage:  ./hpc_connect.sh upload-project [--dry-run]
cmd_upload_project() {
    local target
    target=$(ssh_target)
    local dry=0
    [[ "${1:-}" == "--dry-run" ]] && dry=1
    echo "Uploading project -> ${target}:${PROJECT_DIR_REMOTE}/"
    echo "Local source: ${PROJECT_ROOT}"
    if [[ "$dry" == "1" ]]; then
        echo "[dry-run] remote target: ${target}:${PROJECT_DIR_REMOTE}  (remote expands '\$HOME')"
        echo "[dry-run] local source : ${PROJECT_ROOT}"
        echo "[dry-run] excludes     :"
        printf '    %s\n' "${RSYNC_EXCLUDES[@]}"
        echo "[dry-run] tool          : $(command -v rsync >/dev/null 2>&1 && echo rsync || echo 'tar|ssh (rsync missing)')"
        return 0
    fi
    # Ensure remote dir exists first (needed for rsync mkpath via ssh).
    # ${PROJECT_DIR_REMOTE} is the literal "$HOME/am01_project": NOT single-quoted
    # here so the REMOTE shell expands $HOME (POSIX sh inside ssh would not
    # expand ~ inside single quotes).
    ssh_run "$(ssh_target)" "mkdir -p ${PROJECT_DIR_REMOTE}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -avz -e "ssh ${SSH_OPTS}" "${RSYNC_EXCLUDES[@]}" "${PROJECT_ROOT}/" "${target}:${PROJECT_DIR_REMOTE}"
    else
        # Fallback: pipe a tar stream over ssh so the EXCLUDES are still honoured
        # (plain scp has no --exclude support and would upload .venv/.git).
        # -C PROJECT_ROOT changes dir locally so "./" == the project root.
        local tar_excludes=()
        for e in "${RSYNC_EXCLUDES[@]}"; do
            e="${e#--exclude=}"; e="${e%/}"
            tar_excludes+=( --exclude="./${e}" )
        done
        if [[ -n "${SSH_PASSWORD:-}" ]]; then
            tar -cf - "${tar_excludes[@]}" -C "${PROJECT_ROOT}" . | \
                sshpass -o "${SSH_PASSWORD}" ssh ${SSH_OPTS} "$(ssh_target)" \
                    "mkdir -p ${PROJECT_DIR_REMOTE} && cd ${PROJECT_DIR_REMOTE} && tar -xf -"
        else
            tar -cf - "${tar_excludes[@]}" -C "${PROJECT_ROOT}" . | \
                ssh ${SSH_OPTS} "$(ssh_target)" \
                    "mkdir -p ${PROJECT_DIR_REMOTE} && cd ${PROJECT_DIR_REMOTE} && tar -xf -"
        fi
    fi
    echo "Upload complete."
    echo "To set up the environment:  ./hpc_connect.sh exec \"cd \${PROJECT_DIR_REMOTE} && bash setup_env.sh\""
}

# ── Full deploy: upload + one-shot environment setup ──────────────────────────
#   usage:  ./hpc_connect.sh deploy            # [--dry-run] -> dry-run the upload only
#           ./hpc_connect.sh deploy --dry-run  # show what would be uploaded + setup
cmd_deploy() {
    local dry=0
    [[ "${1:-}" == "--dry-run" ]] && dry=1
    cmd_upload_project $([[ "$dry" == "1" ]] && echo "--dry-run")
    if [[ "$dry" == "1" ]]; then
        echo "[dry-run] would then run setup_env.sh on the HPC (install uv, uv sync --frozen, register kernel)."
        return 0
    fi
    echo ""
    echo "=== Running one-shot environment setup on HPC ==="
    # ${PROJECT_DIR_REMOTE} is literal "$HOME/am01_project": the remote shell
    # expands $HOME. We forward it to setup_env.sh (PROJECT_DIR=...) which also
    # expands $HOME — kept single-quoted so it is passed literally.
    ssh_run "$(ssh_target)" "cd ${PROJECT_DIR_REMOTE} && PROJECT_DIR=${PROJECT_DIR_REMOTE} bash -s" < "${SCRIPT_DIR}/setup_env.sh"
    echo ""
    echo "Deploy complete. The uv venv is ready at ${PROJECT_DIR_REMOTE}/.venv"
    echo "Start working: ./hpc_connect.sh interactive"
}

# ── Batch: deploy + submit + stream log + fetch results ──────────────────────
#   usage:  ./hpc_connect.sh batch <slurm_script>
#           ./hpc_connect.sh batch --dry-run <slurm_script>   # preview only
#
#   After submitting, streams the unified log (stdout+stderr) live via tail -f.
#   Ctrl-C detaches the stream — the job keeps running on SLURM. When the job
#   finishes, results (data/processed/*.npy, *.pkl) and the latest log are
#   downloaded to the local project directory automatically.
cmd_batch() {
    local dry=0
    local script
    if [[ "${1:-}" == "--dry-run" ]]; then dry=1; shift; fi
    script="$1"
    [[ -n "${script}" ]] || { echo "ERROR: usage: ./hpc_connect.sh batch [--dry-run] <slurm_script>"; exit 1; }

    cmd_deploy $([[ "$dry" == "1" ]] && echo "--dry-run")
    [[ "$dry" == "1" ]] && { echo "[dry-run] would then submit ${script} (via upload+sbatch)."; return 0; }

    echo ""

    # ── Upload SLURM script ─────────────────────────────────────────────────
    local basename_script
    basename_script=$(basename "${script}")
    ssh_run "$(ssh_target)" "mkdir -p ~/jobs/logs"
    scp "${script}" "$(ssh_target):~/jobs/${basename_script}"
    echo "Uploaded ${basename_script} → ~/jobs/"

    # ── Submit and capture job ID ───────────────────────────────────────────
    local jid
    jid=$(ssh_run "$(ssh_target)" "cd ~/jobs && sbatch ${basename_script}" 2>&1 \
          | sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p')
    if [[ -z "${jid:-}" ]]; then
        echo "ERROR: could not extract job ID from sbatch output."
        echo "Try manually: ssh $(ssh_target) 'cd ~/jobs && sbatch ${basename_script}'"
        return 1
    fi
    echo "Submitted batch job ${jid}"
    echo ""

    # ── Stream log live until job completes ─────────────────────────────────
    # The SLURM template writes to logs/am01_train_<JID>.log (unified stdout+stderr).
    # Ctrl-C detaches — the job continues on SLURM.
    echo "=== Live log (Ctrl-C to detach, job continues) ==="
    echo "  tail -f ~/jobs/logs/am01_train_${jid}.log"
    echo ""
    ssh_run "$(ssh_target)" "
        set +e
        LOG=\"~/jobs/logs/am01_train_${jid}.log\"
        # Wait for log file to appear (job may be PENDING)
        for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
            [[ -f \$LOG ]] && break
            sleep 5
        done
        # Stream live
        tail -n +1 -f \$LOG 2>/dev/null &
        TAIL_PID=\$!
        # Poll every 15s until the job leaves the queue
        while squeue -j ${jid} -h -o '%T' 2>/dev/null | grep -q .; do
            sleep 15
        done
        kill \$TAIL_PID 2>/dev/null || true
        wait \$TAIL_PID 2>/dev/null || true
        echo ''
        echo '=== Job ${jid} completed ==='
        tail -8 \$LOG 2>/dev/null || true
    "

    # ── Auto-download results ───────────────────────────────────────────────
    echo ""
    echo "=== Fetching results to local ==="
    mkdir -p ./data/processed ./logs

    if command -v rsync >/dev/null 2>&1; then
        rsync -av --progress \
            "$(ssh_target):~/am01_project/data/processed/" \
            ./data/processed/ 2>/dev/null || true
    else
        # Fallback: scp individual files
        for f in train.npy val.npy test_normal.npy test_anomaly.npy \
                 scaler.pkl selected_columns.npy preprocessing_config.json; do
            scp "$(ssh_target):~/am01_project/data/processed/${f}" \
                ./data/processed/ 2>/dev/null || true
        done
    fi

    # Download latest log (fetched to ~/am01_project/logs/ by SLURM post-run rsync)
    local latest_log
    latest_log=$(ssh_run "$(ssh_target)" "ls -t ~/am01_project/logs/am01_train_*.log 2>/dev/null | head -1")
    if [[ -n "${latest_log}" ]]; then
        scp "$(ssh_target):${latest_log}" ./logs/ 2>/dev/null || true
        echo "Log downloaded → ./logs/$(basename "${latest_log}")"
    fi

    echo ""
    echo "Done.  Results: ./data/processed/   Log: ./logs/"
}

cmd_help() {
    cat << 'HELP'
hpc_connect.sh — Automated SSH connection for PolitO HPC Legion

Usage:
  ./hpc_connect.sh check                                    Show connection & env info
  ./hpc_connect.sh keycheck                                 Verify key auth (no prompt)
  ./hpc_connect.sh keycopy                                  Register public key (ssh-copy-id, needs password)
  ./hpc_connect.sh connect                                  Interactive SSH to login node
  ./hpc_connect.sh exec "command string"                    Run command non-interactively
  ./hpc_connect.sh interactive [partition] [walltime]       Interactive SLURM session
  ./hpc_connect.sh scratch                                  Compute node w/ BeeGFS access
  ./hpc_connect.sh upload <local_path> <remote_path>        Upload files to HPC
  ./hpc_connect.sh upload-project                           rsync entire project to HPC
  ./hpc_connect.sh download <remote_path> <local_path>      Download files from HPC
  ./hpc_connect.sh submit <slurm_script.sh>                 Submit SLURM batch job
  ./hpc_connect.sh deploy                                   Upload-project + one-shot setup
  ./hpc_connect.sh batch <slurm_script.sh>                Deploy + submit + stream log + fetch results

Examples:
  ./hpc_connect.sh check
  ./hpc_connect.sh keycheck
  ./hpc_connect.sh exec "ls \$HOME"
  ./hpc_connect.sh interactive gpu_a40_ext 08:00:00
  ./hpc_connect.sh upload data/ ~/jobs/data/
  ./hpc_connect.sh download ~/results/ ./local_results/
  ./hpc_connect.sh deploy
  ./hpc_connect.sh batch hpc/slurm_job_template.sh

Environment:
  HPC_USER           SSH username (default: mungolo)
  HPC_HOST           SSH host alias (default: polito-hpc)
  PROJECT_DIR_REMOTE Remote project path (default: ~/am01_project)
  SSH_PASSWORD       If set, uses sshpass (default: key-based auth)

Partitions available:
  cpu_sapphire, cpu_sapphire_ext, gpu_a40, gpu_a40_ext,
  gpu_a100, gpu_h200, cpu_skylake, cpu_skylake_ext,
  gpu_v100, gpu_v100_ext
HELP
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
    local subcommand="${1:-help}"
    shift || true

    case "${subcommand}" in
        connect)          cmd_connect "$@" ;;
        exec)             cmd_exec "$@" ;;
        interactive)      cmd_interactive "$@" ;;
        scratch)          cmd_scratch "$@" ;;
        upload)           cmd_upload "$@" ;;
        upload-project)   cmd_upload_project "$@" ;;
        download)         cmd_download "$@" ;;
        submit)           cmd_submit "$@" ;;
        check)            cmd_check "$@" ;;
        keycheck)         cmd_keycheck "$@" ;;
        keycopy)          cmd_keycopy "$@" ;;
        deploy)           cmd_deploy "$@" ;;
        batch)            cmd_batch "$@" ;;
        help|--help|-h)   cmd_help "$@" ;;
        *)
            echo "Unknown command: ${subcommand}"
            echo "Run './hpc_connect.sh help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
