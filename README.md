# AM01 - Detection of Anomalous Behaviour in Industrial Robot

## Project Information
- **Project Code**: 2026/AM01
- **Project Owner**: Alessio Mascolini - alessio.mascolini@polito.it
- **Title**: Detection of Anomalous Behaviour in Industrial Robot

## Project Overview
This project involves implementing an adversarial autoencoder (AAE) for anomaly detection on a Kuka industrial robot dataset. The dataset consists of time-series data collected from the robot's various sensors, including joint angle positions, velocity, current and power usage values.

The problem to solve is understanding when, due to an error in configuration or aging, the robot is moving more slowly or less precisely than normal. The AAE model will be trained to learn the normal movement patterns of the robot using the training data. Once the model has learned these patterns, it should be able to identify any deviations from them and flag them as anomalies.

The aim of the project is to evaluate whether an adversarial component improves the performance of a traditional autoencoder to detect anomalies. The choice of appropriate metrics to evaluate your result will be part of the examination.

## Applications
- Industrial, Anomaly Detection

## References
- Kim S. et al., "Towards a Rigorous Evaluation of Time-series Anomaly Detection", 2022

## Challenges
- Using adversarial training
- Handling time series data
- Comparing models

## Project Structure
```
AM01-Detection-of-Anomalous-Behaviour-in-Industrial-Robot/
├── data/
│   ├── raw/
│   │   └── KukaVelocityDataset/
│   └── processed/
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── main.py
├── notebooks/
├── tests/
├── config/
├── reports/
├── docs/
├── hpc/                 # SSH/SLURM automation scripts for PoliTO Legion
├── scripts/
├── .venv/               # uv-managed virtual env (gitignored)
├── .python-version      # Python version pin (3.13)
├── pyproject.toml       # single source of truth (deps)
├── uv.lock              # locked dependency versions
└── requirements.txt     # auto-generated from uv.lock
```

## Dataset Description

### Files
| File | Shape | Description |
|------|-------|-------------|
| KukaNormal.npy | (233,792, 86) | Normal robot movement data |
| KukaSlow.npy | (41,538, 87) | Anomalous/slow movement data |
| KukaColumnNames.npy | (87,) | Feature column names |

### Features
- **Power metrics**: apparent_power, current, frequency, phase_angle, power, power_factor, reactive_power, voltage
- **Sensors**: Accelerometer (AccX/Y/Z), Gyroscope (GyroX/Y/Z), Joint angles (q1-q4), Temperature (temp)

## Installation

This project is managed with [`uv`](https://docs.astral.sh/uv/) — `pyproject.toml`
and `uv.lock` are the single source of truth for dependencies. `requirements.txt`
is auto-generated from the locked versions (`uv export`).

```bash
# First-time setup (creates .venv and installs locked deps)
uv sync

# (optional) register the venv as a Jupyter kernel locally
uv run python -m ipykernel install --user --name am01 --display-name "AM01"
```

## Usage

Run everything through `uv run` so the project venv is used:

```bash
# Launch the pipeline entry point
uv run python src/main.py

# Launch notebooks (uses the am01-hpc kernel when on the cluster)
uv run jupyter notebook
```

## HPC Execution (PoliTO Legion cluster)

All HPC automation lives in [`hpc/`](hpc/README.md). The workflow is designed to
run your code in an isolated `uv` virtualenv on the cluster — no manual
environment juggling.

**Prerequisites**
- SSH key-based auth must be registered on the HPC. Your key `~/.ssh/id_ed25519`
  is already configured in `~/.ssh/config` as the `polito-hpc` alias. Upload its
  **public** part once:
  ```bash
  cat ~/.ssh/id_ed25519.pub   # copy, then register via the PoliTO HPC portal
  ```
- From **outside** the PoliTO network you need VPN first — contact `5050@polito.it`.

**Quick reference (HPC) — run from the project root, inside `hpc/`**

| Goal | Command | Notes |
| --- | --- | --- |
| Check key + connection | `./hpc_connect.sh keycheck` | must print `key-auth OK` |
| Upload project + build env | `./hpc_connect.sh deploy` | rsync to `~/am01_project`, `uv sync --frozen` (Py3.13, kernel `am01-hpc`) |
| Run training on GPU (one-shot) | `./hpc_connect.sh batch hpc/slurm_job_template.sh` | deploy + `sbatch` on `gpu_a40` (uncongested vs `gpu_a40_ext`) |
| Submit only (already deployed) | `./hpc_connect.sh submit hpc/slurm_job_template.sh` | syncs template to `~/jobs/`, then runs `sbatch` |
| Read the SLURM log | `./hpc_connect.sh exec "cat ~/jobs/logs/am01_train_<JID>.out"` | `<JID>` is the id `sbatch` prints |
| Interactive GPU shell | `./hpc_connect.sh interactive gpu_a40 04:00:00` | `srun --pty`; `$SCRATCH` available on the compute node |

**One-shot: deploy + submit a SLURM job (fully automated)**
```bash
cd hpc && ./hpc_connect.sh batch hpc/slurm_job_template.sh
```
`batch` does `deploy` + `submit` in one move: it rsyncs the project to
`~/am01_project` (excluding `.venv`, `.git`, data, checkpoints), runs `setup_env.sh`
remotely (installs `uv`, downloads the Python from `.python-version`, syncs the
frozen `uv.lock`, registers the Jupyter kernel), then submits the SLURM job.

**Step-by-step**
```bash
cd hpc
./hpc_connect.sh keycheck          # verify your key works on the HPC
./hpc_connect.sh deploy            # upload project + create the remote uv venv
./hpc_connect.sh interactive gpu_a40 08:00:00   # get a GPU compute node (gpu_a40, not gpu_a40_ext — less congested)
# ... train on the compute node, results land under $SCRATCH ...
./hpc_connect.sh submit hpc/slurm_job_template.sh   # upload template to ~/jobs + sbatch
#
# NOTE: the template ships with `torch>=2.7,<2.8` (cu126, CUDA 12.6) which matches the
# Legion A40 driver (570.x). PyTorch 2.13 / CUDA 13 wheels are NOT compatible with
# that driver and report `CUDA available: False`. Verified GPU run (job 1911144,
# partition gpu_a40): "CUDA available: True — GPU: NVIDIA A40", ExitCode 0:0.
./hpc_connect.sh download ~/results ./local_results/
```

> `setup_env.sh` runs `uv sync --frozen` (uses the locked `uv.lock` exactly,
> reproducible). Each component (`src`, `notebooks`, `tests`) runs inside this
> single dedicated `.venv` via `uv run`, so there is one isolated environment on
> the cluster just like locally.

## TODO
- [ ] Data exploration and preprocessing
- [ ] Baseline autoencoder implementation
- [ ] Adversarial autoencoder implementation
- [ ] Model training and evaluation
- [ ] Results comparison
- [ ] Final reporting

## License
Academic project for Machine Learning in Applications course - Politecnico di Torino