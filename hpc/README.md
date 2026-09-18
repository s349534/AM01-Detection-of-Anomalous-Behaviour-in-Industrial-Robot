# Script HPC — cluster "Legion" (Politecnico di Torino)

Questi script ti permettono di **caricare il progetto**, **costruire l'ambiente `uv`**
e **eseguire il training su SLURM** senza connetterti in interattivo ogni volta.
Usano **key-based SSH** (nessuna password interattiva durante l'esecuzione).

> **Nota**: il progetto assume un alias SSH di nome `polito-hpc`. Se usi un altro
> nome, vedi [3 — Configura l'alias](#3-configura-lalias) qui sotto.

---

## 1. Prerequisiti — cosa devi preparare (una tantum)

### 1.0 Dati RAW — obbligatori su HPC per il preprocessing automatico

Gli script SLURM (`slurm_ae_search.sh`, `slurm_ae_train.sh`) avviano automaticamente il preprocessing (`src.data.preprocessing`) se `data/processed/train.npy` non esiste. **Il preprocessing richiede i file `.npy` grezzi in `~/am01_project/data/raw/` sul cluster.**

```bash
# I file richiesti:
~/am01_project/data/raw/KukaNormal.npy      # (233792, 86) — dati normali
~/am01_project/data/raw/KukaSlow.npy        # (41538, 87)   — dati anomali
~/am01_project/data/raw/KukaColumnNames.npy # (87,)         — nomi feature
```

**Due modi per caricarli:**

1. **Automatico (consigliato)**: `./hpc_connect.sh deploy` include `data/raw/` nel rsync (vedi `RSYNC_EXCLUDES` in `hpc_connect.sh` — NON è escluso).
2. **Manuale** (se il deploy non li carica, es. account HPC fresco):
   ```bash
   scp data/raw/KukaNormal.npy data/raw/KukaSlow.npy data/raw/KukaColumnNames.npy polito-hpc:~/am01_project/data/raw/
   ```

> ⚠️ **Senza raw data su HPC, il preprocessing produce output troncati** (es. 30 campioni validation invece di ~46k), facendo fallire la validazione HP (`window_size=32 > val_samples=30`).

---

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
**⚠️ Importante**: `BatchMode yes` nel config SSH impedisce il prompt password.
Per `ssh-copy-id` devi disabilitarlo temporaneamente:

```bash
# Opzione consigliata: disabilita BatchMode solo per questo comando
ssh-copy-id -o BatchMode=no -i ~/.ssh/id_ed25519.pub polito-hpc
```
Ti verrà chiesta la **password PoliTO** una sola volta.

Poi verifica che l'auth a chiave funzioni (ora `BatchMode yes` va bene):
```bash
cd hpc && ./hpc_connect.sh keycheck     # deve stampare: "key-auth OK as <user> on <hostname>"
```

### 1.3 Configurazione per-utente via `.env` (consigliato per team)
Per non hardcare credenziali negli script, usa il template `.env.example` nella root del progetto:

```bash
# 1. Copia il template (una volta per macchina/utente)
cp .env.example .env

# 2. Modifica .env con i TUOI valori (editor a scelta)
#    HPC_USER=tuo_username_polito
#    HPC_HOST=polito-hpc
#    PROJECT_DIR_REMOTE='$HOME/am01_project'

# 3. Carica le variabili prima di lanciare gli script
source .env && ./hpc_connect.sh check
```

> **Nota**: `.env` è in `.gitignore` — non viene committato. Ogni membro del team ha il suo `.env` locale.

### 1.4 Network
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
| `./hpc_connect.sh deploy` | ① `rsync` del progetto su `~/am01_project` (incluso `data/raw/`; esclude `.venv`, `.git`, checkpoint); ② `uv sync --frozen` (Python 3.13 richiesto da `.python-version`), registra kernel Jupyter `am01-hpc` | **Prima volta** e dopo ogni cambiamento di dipendenze (`pyproject.toml`/`uv.lock`). `data/raw/` è caricato una volta sola; deploy successivi usano rsync incrementale. |
| `./hpc_connect.sh batch slurm_job_template.sh` | `deploy` + `sbatch` + **stream log live** (`tail -f`) + **auto-fetch risultati**. Uno-shot completo: upload → build → submit → vedi output in tempo reale → a fine job scarica `reports/`, `data/processed/`, `config/` e il log **nella root del progetto** (non in `hpc/`) | **Start uno-shot**: vai dal repository vuoto al risultato in locale. <br> `batch` fa `cd` automatico alla root del progetto (dove sta `pyproject.toml`) prima di scaricare, anche se lanciato da `hpc/`. |
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
  **`~/jobs/logs/am01_<phase>_<JID>.log`** **sul login node**. Il comando `submit`
  crea `~/jobs/logs/` automaticamente (SLURM **non** crea le directory `#SBATCH`).
  Il log va in `~/jobs/logs/` — NON in scratch (questo era la causa del bug "no log file found").
- **Code**: sincronizzato da `~/am01_project/` a `$SCRATCH/am01/` su compute node
  all'inizio del job (esclude `.venv`, `.git`, `__pycache__`, checkpoints `.pth`).
- **Dati raw**: caricati una volta in `~/am01_project/data/raw/` via `deploy` (ora incluso nel rsync).
- **Dati processati**: generati su compute node in `data/processed/` da preprocessing
  automatico, poi sincronizzati a `~/am01_project/data/processed/` con post-run rsync.
- **`batch`** scarica automaticamente `data/processed/`, `reports/`, `config/`, e il
  log più recente in locale (`./data/processed/`, `./reports/`, `./config/`, `./logs/`).

---

## 5. Troubleshooting (essenziale)

| Sintomo | Causa / soluzione |
| --- | --- |
| `Permission denied (publickey)` | Chiave non registrata: `ssh-copy-id -o BatchMode=no -i ~/.ssh/id_ed25519.pub polito-hpc` (serve `-o BatchMode=no` perché il config ha `BatchMode yes` che blocca il prompt password) |
| `ssh-copy-id` fallisce con `Permission denied` | `BatchMode yes` in `~/.ssh/config` impedisce il prompt password. Usa: `ssh-copy-id -o BatchMode=no -i ~/.ssh/id_ed25519.pub polito-hpc` |
| Job non parte / `JobLaunchFailure` | Usa `gpu_a40` (non `gpu_a40_ext`); o per un test veloce una CPU: `./hpc_connect.sh submit slurm_job_template.sh` dopo aver messo `#SBATCH --partition=cpu_sapphire_ext` |
| `slurm_script: line N: SCRATCH: unbound variable` | Template obsoleto: usa sempre `hpc/slurm_job_template.sh` aggiornato (ora usa `SCRATCH_DIR="${SCRATCH:-${HOME}/scratch}"`) |
| `CUDA available: False` su GPU | La wheel di torch è troppo nuova per il driver del nodo: usa `torch>=2.7,<2.8` in `pyproject.toml` (cu126, compatibile con driver 570.x). Verifica con `nvidia-smi` (deve mostrare l'A40) |
| Job in coda (`PENDING`) molto tempo | Partizione affollata — riduci `--time` nel template o passa a `cpu_sapphire_ext` per un test rapido |
| **Nessun file di log dopo `sbatch`** | `~/jobs/logs/` non esiste → `submit` lo crea; se usi `sbatch` manuale: `mkdir -p ~/jobs/logs` prima. **Attenzione**: il log va in `~/jobs/logs/`, NON in scratch. Se il job termina con "no log file found", controlla `ls ~/jobs/logs/am01_*_<JID>.log` |
| **Log non scaricati da `batch` quando il job fallisce presto** | Il post-run rsync nello script SLURM copia i log in `~/am01_project/logs/` SOLO se il job completa. Se il job fallisce prima (es. errore variabile non definita), i log restano in `~/jobs/logs/`. **Fix applicato**: `cmd_batch` ora controlla anche `~/jobs/logs/` come fallback. |
| **Validation set minuscolo (es. 30 campioni invece di ~46k)** | I dati raw (`data/raw/KukaNormal.npy`, `KukaSlow.npy`, `KukaColumnNames.npy`) mancavano su HPC al momento del preprocessing, oppure i dati processati troncati da una run precedente sono stati riutilizzati. **Fix applicato**: script SLURM ora verificano la dimensione del validation set e rilanciano il preprocessing se < 33 campioni; verificano anche l'esistenza dei raw data prima di preprocessare. Assicurati di fare `./hpc_connect.sh deploy` (include `data/raw/`) prima di lanciare i job. |
| **50 epoche ma CSV mostra 2 epoche** | `run_search_ae.py` aveva `resume=True` di default → saltava run_id esistenti. Fix: usa `--no-resume` (default ora `False`). Rimuovi il CSV vecchio da `reports/tables/` prima di rilanciare. |
| **Risultati non aggiornati** | Gli script SLURM usavano `--ignore-existing` su rsync. Fix applicato: rsync sovrascrive sempre. Se i risultati sembrano vecchi, forza `--no-resume`. |
| `scp: stat local "slurm_ae_search.sh": No such file` | `cmd_batch` fa `cd` alla root del progetto prima del fetch; un path relativo come `hpc/slurm_ae_search.sh` non si risolve più. **Fix applicato**: `cmd_batch` risolve l'path assoluto dello script prima di `scp`. |
| `rsync: command not found` | `hpc_connect.sh` usa automaticamente un fallback `tar|ssh` che onora gli stessi `--exclude` |

---

## 6. File in questa cartella

| File | Ruolo |
| --- | --- |
| `hpc_connect.sh` | Dispatcher SSH/SLURM (upload, deploy, submit, batch, interactive, exec). Include `data/raw/` nel deploy rsync. |
| `setup_env.sh` | Installa `uv`, crea il venv con `uv sync --frozen` (Python 3.13), registra kernel Jupyter `am01-hpc` (run in automatico da `deploy`) |
| `slurm_job_template.sh` | Template generico SLURM: rsync → scratch, `uv sync`, torch/CUDA check, preprocessing trigger, lancia `src/main.py` |
| `slurm_ae_search.sh` | **Fase 3.1**: validation search (`run_search_ae.py --no-resume`) → `validation_results_ae.csv` + `params_validated_ae.yaml` + 3 plots. **Default: exhaustive (all 50 combos)**; use `N_ITER` for random subset. Preprocessing automatico su compute node. |
| `slurm_ae_train.sh` | **Fase 3.2**: training finale 3 seed (`train_ae.py`) → `ae_baseline.pth` + `ae_final_metrics.csv`. Preprocessing saltato se `data/processed/` esiste. |

## 7. Workflow completo — da zero a risultati

```bash
cd hpc

# 0. Configurazione per-utente (una volta per macchina)
cp ../.env.example ../.env
# Modifica ../.env con il tuo username PoliTO: HPC_USER=tuo_username
source ../.env

# 1. Prerequisiti: verifica chiave SSH + auth
./hpc_connect.sh keycheck
# Se fallisce: ssh-copy-id -o BatchMode=no -i ~/.ssh/id_ed25519.pub polito-hpc

# 2. Deploy (una volta): upload codice + data/raw + uv sync + kernel Jupyter
./hpc_connect.sh deploy

# 3. Validazione HP (Fase 3.1): exhaustive search (50 combinazioni) × 50 epoche
MAX_VAL_EPOCHS=50 ./hpc_connect.sh batch hpc/slurm_ae_search.sh

# 4. Al termine, verifica localmente:
ls reports/tables/validation_results_ae.csv
ls reports/figures/sensitivity_ae_*.png
ls config/params_validated_ae.yaml
ls logs/am01_ae_search_*.log

# 5. Training finale (Fase 3.2): 3 seed
SEEDS="42 123 7" ./hpc_connect.sh batch slurm_ae_train.sh

# 6. Risultati finali:
ls reports/checkpoints/ae_baseline.pth
ls reports/tables/ae_final_metrics.csv
```

### Cosa fa `batch`

`cmd_batch` esegue un flusso completo:
1. **Deploy**: rsync del progetto (ora con `data/raw/`) + `uv sync --frozen`
2. **Upload script**: copia lo script SLURM in `~/jobs/`
3. **sbatch**: sottomette il job, ottiene JID
4. **Stream log**: `tail -f ~/jobs/logs/am01_*_<JID>.log` in tempo reale (Ctrl-C per staccare; il job continua)
5. **Auto-fetch**: al termine, rsync di `reports/`, `config/`, `data/processed/` da `~/am01_project/` → locale

### Preprocessing automatico

Gli script SLURM (`slurm_ae_search.sh`, `slurm_ae_train.sh`, `slurm_job_template.sh`)
verificano se `data/processed/train.npy` esiste **E** se il validation set ha almeno 33 campioni (necessario per `window_size=32` dello search space). Se manca o è troppo piccolo, lanciano:
```bash
uv run python -m src.data.preprocessing
```
Questo evita di dover caricare manualmente i dati processati — basta avere
`data/raw/` su HPC (incluso nel `deploy` rsync).

**Flusso dati completo:**
1. **Locale**: `data/raw/` contiene i 3 file `.npy` grezzi (KukaNormal, KukaSlow, ColumnNames)
2. **Deploy**: `./hpc_connect.sh deploy` fa rsync di tutto il progetto (incluso `data/raw/`) → `~/am01_project/` su HPC
3. **SLURM job**: fa rsync `~/am01_project/` → `$SCRATCH/am01/` su compute node
4. **Preprocessing** (su compute node): legge `data/raw/`, scrive `data/processed/` (train.npy, val.npy, test_normal.npy, test_anomaly.npy, scaler.pkl, selected_columns.npy)
5. **Training/Validation**: usa `data/processed/`
6. **Post-run rsync**: copia `data/processed/`, `reports/`, `config/`, log da `$SCRATCH` → `~/am01_project/`
7. **Auto-fetch** (`batch`): scarica da `~/am01_project/` → locale (`./data/processed/`, `./reports/`, `./config/`, `./logs/`)

### Variabili d'ambiente personalizzabili

| Variabile | Default | Descrizione |
|---|---|---|
| `N_ITER` | (none - exhaustive) | Numero di iterazioni random search. Se non specificato, esegue TUTTE le 50 combinazioni exhaustive. |
| `MAX_VAL_EPOCHS` | 50 | Epoche massime per ogni run di validazione |
| `SEEDS` | "42 123 7" | Seed per training finale (solo `slurm_ae_train.sh`) |
| `SCRATCH_PROJECT` | am01 | Nome della directory su scratch |

```bash
# Esempio: search veloce 5 iterazioni, 10 epoche, CPU
N_ITER=5 MAX_VAL_EPOCHS=10 ./hpc_connect.sh batch hpc/slurm_ae_search.sh
```
