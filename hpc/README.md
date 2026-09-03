# Script HPC — cluster "Legion" (Politecnico di Torino)

Questi script ti permettono di **caricare il progetto**, **costruire l'ambiente `uv`**
e **eseguire il training su SLURM** senza connetterti in interattivo ogni volta.
Usano **key-based SSH** (nessuna password interattiva durante l'esecuzione).

> **Nota**: il progetto assume un alias SSH di nome `polito-hpc`. Se usi un altro
> nome, vedi [3 — Configura l'alias](#3-configura-lalias) qui sotto.

---

## 1. Prerequisiti — cosa devi preparare (una tantum)

### 1.1 SSH key
Se non hai una chiave SSH **senza passphrase** (necessaria per l'auth non interattiva):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""   # -N "" => nessuna passphrase
```
Aggiungi l'alias nel tuo file `~/.ssh/config` (creane uno nuovo se non esiste):
```
Host polito-hpc
    HostName hpc-legionlogin.polito.it
    User <TUO_USER_POLITO>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking yes
    ConnectTimeout 10
    BatchMode yes                 # blocca l'auth a chiave (nessun prompt password)
```
> `<TUO_USER_POLITO>` è il tuo login PoliTO (es. `utente`). `HostName` è
> `hpc-legionlogin.polito.it` (login node — endpoint pubblico del cluster).
> Salva il file, poi rendilo privato:
> `chmod 600 ~/.ssh/config`.

### 1.2 Registra la chiave SUL CLUSTER (password PoliTO una sola volta)
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub polito-hpc
```
Poi verifica:
```bash
cd hpc && ./hpc_connect.sh keycheck     # deve stampare: "key-auth OK as <user> on <hostname>"
```

### 1.3 Network
- **In campus**: SSH diretto, niente VPN.
- **Fuori campus**: devi attivare la **VPN** PoliTO (richiesta a `5050@polito.it`)
  prima di usare gli script.

---

## 2. Comandi — cosa fa ciascuno e quando usarlo

Eseguili dalla cartella `hpc/` (i comandi accettano percorsi relativi al **radice del
progetto**, quindi posizionati sempre lì):

```bash
cd hpc
```

| Comando | Cosa fa | Quando |
| --- | --- | --- |
| `./hpc_connect.sh keycheck` | Verifica che l'auth a chiave funzioni (senza aprire sessione) | Prima di ogni sessione / se l'auth dà errore |
| `./hpc_connect.sh check` | Mostra user, host, `$SCRATCH`, partizioni disponibili | Diagnostica iniziale |
| `./hpc_connect.sh deploy` | ① `rsync` del progetto su `~/am01_project` (esclude `.venv`, `.git`, `data/raw`, checkpoint); ② `uv sync --frozen` (Python 3.13 richiesto da `.python-version`), registra kernel Jupyter `am01-hpc` | **Prima volta** e dopo ogni cambiamento di dipendenze (`pyproject.toml`/`uv.lock`) |
| `./hpc_connect.sh batch slurm_job_template.sh` | `deploy` + `sbatch` + **stream log live** (`tail -f`) + **auto-fetch risultati**. Uno-shot completo: upload → build → submit → vedi output in tempo reale → a fine job scarica `data/processed/*.npy` `.pkl` e il log localmente | **Start uno-shot**: vai dal repository vuoto al risultato in locale |
| `./hpc_connect.sh submit slurm_job_template.sh` | Carica il template in `~/jobs/` + `sbatch` (ritorna subino) | Se il progetto è **già** deployato e vuoi rilanciare solo il job |
| `./hpc_connect.sh interactive gpu_a40 04:00:00` | `srun --pty` su un nodo compute con GPU | Sviluppo interattivo — **solo da qui hai `$SCRATCH`** |
| `./hpc_connect.sh exec "comando"` | Esegue `comando` una volta sola sul login node | Comandi ad hoc (es. `ls`, `cat <log>`, `nvidia-smi`) |
| `./hpc_connect.sh download <remoto> <locale>` | Copia file/dati dal cluster | Portare i risultati a casa |

> **Perché `gpu_a40` e non `gpu_a40_ext`?** `gpu_a40` è meno affollata e basta la
> VPN (o il campus). `gpu_a40_ext` va bene se `gpu_a40` è piena.

Puoi cambiare alias/host/directory di destinazione via env:
```bash
export HPC_HOST=polito-hpc                       # alias SSH (default)
export PROJECT_DIR_REMOTE='$HOME/am01_project'   # dove caricare il progetto (remote)
```

---

## 3. Esempio completo — start da zero

```bash
cd hpc
./hpc_connect.sh keycheck                                   # 0. registra la chiave
./hpc_connect.sh batch hpc/slurm_job_template.sh            # 1. upload + build env + sbatch + stream + fetch
```
`sbatch` stampa l'id del job:
```
Submitted batch job 1911144
```
Poi **il comando mantiene aperta la connessione** e streamma il log live (`tail -f`)
fino a quando il job non termina. Alla fine scarica **automaticamente** i risultati:
```
=== Fetching results to local ===
Results: ./data/processed/   Log: ./logs/
```

---

### Vedere l'esecuzione al volo (live, come in locale)

Con `batch` il log viene streammato automaticamente finché il job è in esecuzione.
Puoi staccare in qualsiasi momento con **Ctrl-C**: **il job continua** su SLURM.
I risultati verranno comunque scaricati quando il job termina.

Se invece preferisci il controllo manuale:

1. **Stream via `tail -f`** su log unico (`.log` contiene stdout+stderr con tag):
   ```bash
   ./hpc_connect.sh exec 'tail -n +1 -f ~/jobs/logs/am01_train_<JID>.log'
   ```
   Il template forza `PYTHONUNBUFFERED=1` e `src/main.py` logga su `sys.stdout`
   (con tag `%(levelname)s`), così `print()` e `logger.info()` compaiono insieme
   nel file `.log`. `Ctrl-C` per staccare.

2. **Controlla lo stato**, poi leggi il log unico:
   ```bash
   ./hpc_connect.sh exec 'squeue -j <JID> -h -o "%T"'
   ./hpc_connect.sh exec 'cat ~/jobs/logs/am01_train_<JID>.log'
   ```

3. **Sessione interattiva** (output identico a locale):
   ```bash
   ./hpc_connect.sh interactive gpu_a40 02:00:00
   uv run python src/main.py --config config/config.yaml
   ```
   Consigliato per il **debug**; per training lungo usa `batch` (con streaming).

## 4. Dove finiscono i file

- **Log**: `sbatch` scrive stdout+stderr (uniti dal `#SBATCH --output=logs/...`) in
  **`~/jobs/logs/am01_train_<JID>.log`** **sul login node**. Il comando `submit`
  crea `~/jobs/logs/` automaticamente (SLURM **non** crea le directory `#SBATCH`).
- **Risultati** (`data/processed/*.npy`, `*.pkl`, `*.log`): il template SLURM esegue
  un **post-run rsync** da `$SCRATCH_DIR/am01/` → `$HOME/am01_project/` così sono
  disponibili anche da login node.
- **`batch`** scarica automaticamente `data/processed/` + il log più recente in
  `./data/processed/` e `./logs/` della tua macchina locale alla fine del job.

---

## 5. Troubleshooting (essenziale)

| Sintomo | Causa / soluzione |
| --- | --- |
| `Permission denied (publickey)` | Chiave non registrata: `ssh-copy-id -i ~/.ssh/id_ed25519.pub polito-hpc` (chiede password PoliTO) |
| Job non parte / `JobLaunchFailure` | Usa `gpu_a40` (non `gpu_a40_ext`); o per un test veloce una CPU: `./hpc_connect.sh submit slurm_job_template.sh` dopo aver messo `#SBATCH --partition=cpu_sapphire_ext` |
| `slurm_script: line N: SCRATCH: unbound variable` | Template obsoleto: usa sempre `hpc/slurm_job_template.sh` aggiornato (ora usa `SCRATCH_DIR="${SCRATCH:-${HOME}/scratch}"`) |
| `CUDA available: False` su GPU | La wheel di torch è troppo nuova per il driver del nodo: usa `torch>=2.7,<2.8` in `pyproject.toml` (cu126, compatibile con driver 570.x). Verifica con `nvidia-smi` (deve mostrare l'A40) |
| Job in coda (`PENDING`) molto tempo | Partizione affollata — riduci `--time` nel template o passa a `cpu_sapphire_ext` per un test rapido |
| Nessun file di log dopo `sbatch` | `~/jobs/logs/` non esiste → `submit` lo crea; se usi `sbatch` manuale: `mkdir -p ~/jobs/logs` prima |
| `rsync: command not found` | `hpc_connect.sh` usa automaticamente un fallback `tar|ssh` che onora gli stessi `--exclude` |

---

## 6. File in questa cartella

| File | Ruolo |
| --- | --- |
| `hpc_connect.sh` | Dispatcher SSH/SLURM (upload, deploy, submit, batch, interactive, exec) |
| `setup_env.sh` | Installa `uv`, crea il venv con `uv sync --frozen` (Python 3.13), registra kernel Jupyter `am01-hpc` (run in automatico da `deploy`) |
| `slurm_job_template.sh` | Template di job batch SLURM: rsync → scratch, `uv sync`, torch/CUDA check, lancia `src/main.py` |
