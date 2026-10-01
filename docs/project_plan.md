# Project Plan — AM01: Anomalous Behaviour Detection in Industrial Robot

> **Single source of truth** per il progetto. Va aggiornato man mano che il lavoro
> procede, le scelte cambiano o le priorità si ribilanciano. È il documento da
> leggere prima di ogni sessione di lavoro per ricordare *dove siamo*, *dove
> stiamo andando* e *perché abbiamo fatto queste scelte*.

---

## 0. TL;DR

- **Obiettivo**: implementare un **Adversarial Autoencoder (AAE)** per anomaly
  detection su dati time-series di un robot Kuka, e **dimostrare** se la
  componente avversariale migliora un autoencoder tradizionale.
- **Approccio**: bottom-up — dati → preprocessing → baseline AE → AAE →
  valutazione comparativa rigorosa.
- **Architettura**: **sequence-aware** con finestra scorrevole di W timestep.
  Encoder 1D-Conv + decoder speculare. Scelta motivata dal paper di
  riferimento (Kim S. et al.) e dalla natura dinamica del problema.
- **Domanda di ricerca**: *La componente avversariale di un AAE migliora le
  performance di anomaly detection rispetto a un autoencoder vanilla su dati
  time-series di un robot industriale?*
- **Vincoli accademici**: consegna = codice + report (template LaTeX del PO) +
  presentazione 20 min. Valutazione 70% gruppo + 30% individuale.

---

## 1. Specifica del progetto (dalla presentazione)

### 1.1 Cosa chiede la traccia
> "The project involves implementing an **adversarial autoencoder (AAE)** for
> anomaly detection on a **Kuka industrial robot dataset**. The dataset consists
> of **time-series data** collected from the robot's various sensors, including
> joint angle positions, velocity, current and power usage values. The problem
> to solve is understanding when, due to an error in configuration or aging, the
> robot is moving **more slowly or less precisely** than normal.
> the robot using the training data. Once the model has learned these patterns, it should be able to identify any deviations from them and flag them as anomalies.
> The aim of the project is to evaluate whether an adversarial component improves the performance of a traditional autoencoder to detect anomalies. 
> The choice of appropriate metrics to evaluate your result will be part of the examination."

### 1.2 Requisiti espliciti
1. AAE addestrato su dati **normali** (semi-supervised).
2. Rilevazione deviazioni → flag anomalie.
3. **Confronto** AAE vs autoencoder tradizionale.
4. **La scelta delle metriche appropriate è parte della valutazione** (quindi va
   motivata, non solo elencata).

### 1.3 Riferimento chiave
- **Kim S. et al., "Towards a Rigorous Evaluation of Time-series Anomaly
  Detection", 2022** — da leggere perché la traccia lo cita esplicitamente.

---

## 2. Dataset

### 2.1 File disponibili (in `data/raw/KukaVelocityDataset/`)

| File | Shape | Note |
|---|---|---|
| `KukaNormal.npy` | `(233792, 86)` | Movimenti normali del robot |
| `KukaSlow.npy` | `(41538, 87)` | Movimenti anomali/lenti |
| `KukaColumnNames.npy` | `(87,)` | Nomi delle 87 colonne |

### 2.2 Struttura del dataset (RISOLTO in Fase 1)
- **`KukaSlow` ha 87 colonne vs 86 di `KukaNormal`** → la 87ª colonna è `anomaly`,
  una label binaria (valore sempre 1) aggiunta SOLO in KukaSlow. Non è una feature
  ma l'etichetta della classe anomala. Rimossa in Fase 2 per allineare le due
  matrici a 86 colonne. Non esiste timestamp: l'ordine riga = ordine temporale.
  - **Alternativa considerata**: salvare `anomaly` come `y_train.npy`. Rifiutata:
    le etichette sono implicite (0 per normal, 1 per anomaly) e gestite dal
    `KukaDataset` via parametro `label`, eliminando rischi di desincronizzazione.
- 86 feature eterogenee: potenza, accelerometro, giroscopio, angoli giunti,
  temperatura, fattore di potenza, tensione, corrente, … (lista completa in
  README.md).
- **Campionamento regolare**: i dati sono ordinati temporalmente (lag-1 AC =
  0.99+), nessun gap. L'indice di riga corrisponde all'istante temporale.

### 2.3 Strategia di split
- **KukaNormal NON va nel training per intero.** Va diviso in tre sottoinsiemi
  (70/15/15), mentre `KukaSlow` viene diviso 50/50 tra validation e test.
- **Training (70% di KukaNormal, ~163k)**: il modello impara la distribuzione
  normale.
- **Validation set (HP selection)**:
  - 15% di KukaNormal (~35k) → `val_normal`: per early stopping + calibrazione soglia (99° percentile errore ricostruzione)
  - 50% di KukaSlow (~21k) → `val_anomaly`: per calcolo PR-AUC/ROC-AUC/F1 nella selezione HP
  - Totale ~56k campioni etichettati (normal + anomaly)
- **Test set finale (report only)**:
  - 15% di KukaNormal (~35k) → `test_normal`: classe "normale" per metriche finali
  - 50% di KukaSlow (~21k) → `test_anomaly`: classe "anomala" per metriche finali
  - Totale ~56k campioni etichettati

**Modalità di split**: **temporale** (no shuffle).
- **Temporale** (scelta adottata): simula il deployment reale (addestri su
  passato, valuti su futuro), preserva l'ordine cronologico. È la scelta del
  paper Kim S. et al. (2022) e coerente con lag-1 AC = 0.99+.
- **Random** (shuffle con seed): statisticamente train/val/test identici, ma
  inappropriato qui: con lag-1 AC = 0.99+ lo shuffle romperebbe la continuità
  temporale e causerebbe data leakage (informazione futura in passato).
- Confermato in Fase 1: i dati sono una sessione continua (nessun timestamp,
  lag-1 AC elevata), quindi lo split temporale è corretto.

> **Perché solo normali in training?** L'AAE impara la distribuzione del
> "comportamento normale"; in inference un input con alta ricostruzione errore
> è, per definizione, un'anomalia. Mescolare anomale in training "avvelena" il
> modello.
>
> **Perché NON usare KukaNormal al 100% nel training?** Senza val set non
> possiamo fare early stopping né calibrare la soglia; senza test_norm non
> abbiamo un riferimento "vero negativo" pulito su cui riportare metriche
> oneste.
>
> **Perché KukaSlow diviso 50/50 tra val e test?** Per calcolare PR-AUC/ROC-AUC
> durante la HP selection servono anomalie nel validation set. Senza, `best_val_pr_auc`
> verrebbe calcolato sul test set (data leakage). Il 50% di slow in validation
> permette selezione HP onesta; il resto in test garantisce report finale pulito.

---

## 3. Approccio metodologico

### 3.1 Perché bottom-up
Iniziamo dai dati perché:
1. Le scelte di preprocessing dipendono da *cosa c'è dentro* (range, outlier,
   cardinalità).
2. La forma del modello dipende da come trattiamo le time-series (point-wise vs
   finestra).
3. Le metriche dipendono da *che tipo di sbilanciamento* abbiamo.
4. Implementare modelli su dati sporchi porta a conclusioni sbagliate.

### 3.2 Roadmap a fasi

#### **Fase 1 — Esplorazione dati** (Notebook 01)
**Obiettivo**: capire il dataset *prima* di qualsiasi trasformazione.

- Caricare i 3 `.npy`.
- Stampare shape, dtype, sample values, memoria occupata.
- Identificare la 87ª colonna di `KukaSlow` e decidere cosa farne.
- Statistiche per feature: media, std, min, max, %NaN, %zeri, range.
- Confronto distribuzioni Normal vs Slow (istogrammi, boxplot) per identificare
  feature discriminanti.
- Heatmap correlazioni tra feature.
- Correlazione di ogni feature con la velocità (segnale chiave del problema).
- PCA 2D e t-SNE 2D per vedere se le classi sono visivamente separabili.
- Verifica se i dati sono ordinati temporalmente e se il campionamento è
  regolare.

**Output atteso**: notebook eseguito end-to-end + breve commento scritto sulle
decisioni da prendere (feature da scartare, tipo normalizzazione, finestra
temporale).

#### **Fase 2 — Preprocessing** (Notebook 02 + `src/data/preprocessing.py`)
**Obiettivo**: pipeline riproducibile, serializzabile, testabile.

- Rimozione colonna `anomaly` da `KukaSlow` (etichetta, non feature) + rimozione
  4 feature costanti (verificate su entrambi i dataset).
- **NO clipping** — i valori di saturazione sensore vengono gestiti da
  StandardScaler + MAE loss (vedi §7 #9).
- **Split** (temporale, no shuffle):
  - `KukaNormal`: 70% train, 15% val_normal, 15% test_normal
  - `KukaSlow`: 50% val_anomaly, 50% test_anomaly
- **Normalizzazione**: `StandardScaler` (fit **solo** sul train) — motivazione §4.
- Windowing on-the-fly in `KukaDataset` (non pre-computato) — motivazione §4.1.
- Salvataggio:
  - `data/processed/{train, val_normal, val_anomaly, test_normal, test_anomaly}.npy` (5 file)
  - `data/processed/scaler.pkl`
  - `data/processed/selected_columns.npy` (82 nomi, dtype `<U37`)
  - Le etichette sono implicite (0 = normal, 1 = anomaly), non salvate separatamente.
- Logica in `src/data/preprocessing.py` + `src/data/dataset.py` (riutilizzabile, testabile).
- 33 test in `tests/test_dataset.py` (tutti passanti).

**Configurazione**: tutti gli iperparametri in `config/params.yaml`, letti via
`src/utils/config.py`.

#### **Fase 3.0 — Costruzione modello AE parametrico**
**Obiettivo**: implementare l'architettura AE (vedi §4.1.2) con costruttore
che accetta gli iperparametri da validare. Senza questo, la validazione (3.1)
non può esplorare lo spazio di ricerca.

- File: `src/models/autoencoder.py` (Encoder, Decoder, Autoencoder).
- File: `src/models/train_utils.py` (train_one_epoch, validate, EarlyStopping).
- File: `src/validation/run_experiment_ae.py` (`train_and_evaluate_ae(config)`).
- File: `src/validation/run_search_ae.py` (CLI).
- Test: 1 run di 3-5 epoche per verificare la pipeline end-to-end.

> **Perché il costruttore è parametrico dall'inizio**: il validatore
> (Fase 3.1) deve poter istanziare 20 modelli con 20 combinazioni di HP senza
> toccare il codice. Costruttore rigido → riscrittura ad ogni run.

#### **Fase 3.1 — Validazione AE**
**Obiettivo**: identificare `HP_AE_best` tramite random search vincolato
(spazio in §4.8).

- Esecuzione: `python -m src.validation.run_search_ae --n-iter 20 --seed 42`.
- Output: `reports/tables/validation_results_ae.csv`.
- Analisi: `python -m src.validation.analyze_results` → 3 grafici di
  sensitività (`sensitivity_ae_*.png`).
- Selezione: riga con `best_val_pr_auc` massimo (vedi §4.8.8).
- Output finale: `config/params_validated_ae.yaml`.

> **Nota sul flusso validation**: Ogni run esegue:
> 1. Training su `train` (70% normali) con early stopping su `val_normal` (15% normali)
> 2. HP selection: soglia (99° percentile su `val_normal`) + PR-AUC/ROC-AUC/F1 su `val_normal` + `val_anomaly` (15% normali + 50% slow)
> 3. Test finale: metriche su `test_normal` + `test_anomaly` (15% normali + 50% slow) — **solo per report, mai per HP selection**
> 4. `best_val_pr_auc` proviene dal validation set (NON dal test set come in versioni precedenti)

#### **Fase 3.2 — Training finale AE**
**Obiettivo**: addestrare il modello definitivo con `HP_AE_best`.

- Esecuzione: `python -m src.models.train_ae --config params_validated_ae.yaml`.
- Nested validation: 3 run con seed diversi sulla stessa config → media ± std.
- Output: `reports/checkpoints/ae_baseline.pth` + `ae_final_metrics.csv`.

#### **Fase 4.0 — Costruzione modello AAE parametrico**
**Obiettivo**: aggiungere il discriminatore e l'addestramento alternato
(Makhzani et al., 2015), riusando Encoder/Decoder da 3.0.

- File: `src/models/adversarial_ae.py` (Discriminator, AdversarialAE).
- File: `src/validation/run_experiment_aae.py` (`train_and_evaluate_aae`).
- File: `src/validation/run_search_aae.py` (CLI).
- Test: 1 run di 3-5 epoche.

#### **Fase 4.1 — Validazione AAE**
**Obiettivo**: identificare `HP_AAE_best` validando i 2 HP AAE-specifici
(`reconstruction_weight`, `adversarial_weight`), con HP AE-derivati fissati a
`HP_AE_best`.

- Esecuzione: `python -m src.validation.run_search_aae --n-iter 15 --seed 42`.
- Output: `reports/tables/validation_results_aae.csv` (15 righe).
- Analisi: 2 grafici di sensitività (`sensitivity_aae_*.png`).
- Selezione: vedi §4.8.3.
- **Sanity check**: AAE vs AE → recovery condizionale se `PR-AUC_AAE <
  PR-AUC_AE - 0.05` (vedi §4.8.7).
- Output finale: `config/params_validated_aae.yaml`.

#### **Fase 4.2 — Training finale AAE**
**Obiettivo**: addestrare il modello definitivo con `HP_AAE_best`.

- Esecuzione: `python -m src.models.train_aae --config params_validated_aae.yaml`.
- Nested validation: 3 run con seed diversi.
- Output: `reports/checkpoints/aae_final.pth` + `aae_final_metrics.csv`.

#### **Fase 5 — Valutazione e confronto** (`src/models/compare_models.py` + `src/utils/metrics.py`)
**Obiettivo**: rispondere *quantitativamente* alla domanda della traccia.

- **Per ogni modello** (AE e AAE):
  - Calcolo errore di ricostruzione per ogni sample.
  - Scelta soglia: percentile 95–99 sul validation set Normal (per garantire
    ~5% FPR baseline) + alternativa con massimizzazione F1 su validation.
  - Metriche su test set:
    - Accuracy, Precision, Recall, F1
    - **PR-AUC** (più informativo di ROC-AUC su dati sbilanciati)
    - **ROC-AUC**
    - Confusion matrix
  - Distribuzione errori per classe (visualizzata).

- **Confronto**:
  - Tabella affiancata AE vs AAE.
  - Curve ROC e PR sovrapposte.
  - Test statistico (McNemar o paired bootstrap) per dire se la differenza è
    significativa.

**Output**: `reports/tables/metrics_comparison.{csv,tex}`,
`reports/figures/{roc,pr,error_dist,confusion}_*.png`.

#### **Fase 6 — Test, pulizia, riproducibilità**
- Test `pytest` in `tests/`:
  - `test_dataset.py`: shape, tipi, split deterministico.
  - `test_models.py`: forward pass, count parametri, output range.
  - `test_metrics.py`: valori noti (es. AUC=1 su predizione perfetta, AUC=0.5
    su random).
- `src/main.py` come entry point unico (CLI con argomenti).
- README aggiornato con istruzioni di riproducibilità.

---

## 4. Scelte di design (motivazioni)

### 4.1 Input: sequence-aware (finestra temporale)

**Scelta**: **sequence-aware con finestra scorrevole di W timestep consecutivi.**

Ogni sample non è più una singola riga `(86,)` ma un blocco di `W` righe
consecutive, organizzato come tensore `(W, 86)`. La rete vede quindi un piccolo
spezzone di storia recente e può cogliere pattern temporali — trend, derive
lente, periodicità, micro-oscillazioni.

#### Confronto Point-wise vs Sequence-aware

| Aspetto | Point-wise | Sequence-aware (finestra W) |
|---|---|---|
| Input alla rete | `(86,)` (un solo istante) | `(W, 86)` (W istanti consecutivi) |
| Architettura | MLP (Linear + ReLU) | 1D-Conv (encoder temporale) |
| Cattura pattern temporali | ❌ no | ✅ sì |
| Riferimento alla traccia | non citato | ✅ **in linea con Kim S. et al.** (rif. obbligatorio) |
| Tipo di anomalie rilevate | solo statiche ("qui e ora è strano") | anche dinamiche ("negli ultimi W passi è andato su lentamente") |
| Velocità training | 🚀 veloce | 🐢 più lento (più parametri, più passi) |
| Complessità codice | bassa | media (gestione finestra + padding) |

#### Esempio concreto

Supponiamo `W = 4` e di avere al tempo `t` i 4 istanti consecutivi
`x_{t-3}, x_{t-2}, x_{t-1}, x_t`, ciascuno di 86 feature.

- **Point-wise** analizza solo `x_t` da solo. Non sa che la temperatura è
  salita nei 3 istanti precedenti: vede solo "temperatura = 50°C" e decide
  che è normale.
- **Sequence-aware** analizza l'intera sequenza `[x_{t-3}, x_{t-2}, x_{t-1}, x_t]`.
  Può riconoscere il *trend* (sale di 5°C a ogni passo) e segnalarlo come
  anomalia, anche se ogni singolo istante preso da solo sembra normale.

Questo è esattamente il caso del nostro problema: la traccia parla di robot
che si muove "**more slowly or less precisely than normal**". Sono
caratteristiche **dinamiche** — un robot rallentato a ogni timestep può
avere valori istantanei dei singoli sensori del tutto plausibili, e solo
l'evoluzione temporale lo tradisce.

#### Perché sequence-aware e non point-wise

1. **È nel paper di riferimento**. Kim S. et al., "Towards a Rigorous
   Evaluation of Time-series Anomaly Detection" (2022) — citato
   esplicitamente nella traccia a p. 20 — lavora su approcci sequence-aware
   per time-series AD. Adottare point-wise significherebbe ignorare il
   riferimento bibliografico indicato dal PO.
2. **Il problema è dinamico per natura**. "Moving more slowly" e "less
   precisely" sono descrizioni di *andamento nel tempo*, non di valori
   puntuali. Un approccio che guarda un solo istante per volta non può
   cogliere la differenza tra "va piano perché sta decelerando" e "va piano
   perché è in fase di riposo".
3. **Migliore espressività → risultati migliori**. A parità di modello, dare
   in input W istanti consecutivi fornisce alla rete più informazione utile
   (più varianza spiegabile nel latent). La letteratura su time-series AD
   mostra che i modelli sequence-aware (USAD, Anomaly Transformer, LSTM-VAE)
   battono costantemente i modelli point-wise sugli stessi benchmark.
4. **Costo accettabile**. Il rallentamento in training è reale ma non
   drammatico: 1D-Conv è computazionalmente efficiente, e con `W` piccolo
   (es. 8–32) il carico resta gestibile su GPU anche medio-bassa. Il
   vantaggio in espressività compensa largamente il costo.

#### Svantaggi di sequence-aware (per onestà)

- **Più lento in training** rispetto a point-wise: ogni forward pass elabora
  `W × 86` input invece di `86`. Con `W=16` e batch 256 sono ~350k valori per
  batch contro i 22k point-wise.
- **Gestione del bordo**: i primi `W-1` istanti di ogni sequenza non hanno
  abbastanza "storia a sinistra". Soluzioni: padding (zeri o repliche),
  trimming (scartare i primi `W-1`), o inizio a `t=W-1`. Scelta →
  trim + scarto (i primi `W-1` sample sono pochi rispetto ai 233k).
- **Iperparametro `W` in più**: va scelto. Troppo piccolo perde informazione
  temporale, troppo grande diluisce il segnale in rumore (e rallenta).
  Vedi §4.1.1 sotto per come sceglierlo.
- **Architettura leggermente più complessa**: serve 1D-Conv (o LSTM) al posto
  di un MLP semplice.

#### 4.1.1 Scelta della dimensione della finestra `W`

`W` è il numero di timestep consecutivi che diamo in pasto alla rete ad ogni
sample. È l'iperparametro più importante di questa sezione.

**Criteri per sceglierlo**:

1. **Frequenza di campionamento** (se nota). Se i dati sono a 100 Hz e un
   "comportamento lento" emerge in 1–2 secondi, allora `W` deve coprire
   almeno quel range: `W ≥ 100` per 1s, `W ≥ 200` per 2s. Se la frequenza
   non è nota o è irregolare, va dedotta in Fase 1.
2. **Tipo di pattern da catturare**. Il problema parla di "rallentamento" e
   "perdita di precisione": sono fenomeni a bassa frequenza (trend lenti,
   non spike impulsivi). `W` non deve essere né troppo corto (perderebbe
   il trend) né troppo lungo (includerebbe troppe oscillazioni fisiologiche
   come "rumore").
3. **Complessità accettabile**. `W` grande = più parametri nella prima
   conv, batch più pesanti, training più lento. `W` ragionevole: 8–64.
4. **Validazione empirica**. In Fase 3 confrontiamo `W ∈ {8, 16, 32, 64}` su
   validation set → scegliamo quello con miglior trade-off errore di
   ricostruzione / tempo di training.

**Default iniziale**: **`W = 16`** (salvo indicazioni dalla Fase 1).

Da aggiungere a `config/params.yaml`:
```yaml
data:
  window_size: 16
  window_stride: 1   # finestre scorrevoli di 1 timestep
```

#### 4.1.2 Architettura del sequence-aware Autoencoder

L'encoder non è più un MLP piatto: deve ridurre un tensore `(W, 86)` a un
vettore latente `(latent_dim,)` **riassumendo l'evoluzione temporale**.

**Architettura scelta: Encoder convoluzionale 1D (3 layer, 2 pooling) + Decoder speculare auto-derivato.**

```
NEW ENCODER (3 conv layers, 2 pools)          NEW DECODER (auto-derived)
─────────────────────────────────────────     ─────────────────────────────────────────
Input  (B, 82, W)                              Input  (B, latent_dim)
   ↓                                               ↓
Conv1d(82→64, kernel=5, pad=2) + ReLU            Linear(latent → 16·W/4) + ReLU
   ↓                                               ↓
MaxPool1d(2)  → (B, 64, W/2)                   Reshape → (B, 16, W/4)
   ↓                                               ↓
Conv1d(64→32, kernel=3, pad=1) + ReLU          ConvTranspose1d(16→32, k=4,s=2,p=1) + ReLU
   ↓                                               ↓
MaxPool1d(2)  → (B, 32, W/4)                   ConvTranspose1d(32→64, k=4,s=2,p=1) + ReLU
   ↓                                               ↓
Conv1d(32→16, kernel=3, pad=1) + ReLU          Conv1d(64→82, kernel=3, pad=1)
   ↓                                               ↓
Flatten → (B, 16·W/4)                          Output (B, 82, W)
   ↓
Linear(16·W/4 → latent_dim)
   ↓
z  (B, latent_dim)
```

**Note sull'architettura**:

- **Input layout**: PyTorch `Conv1d` vuole `(batch, channels, length)`,
  quindi passiamo `(B, 82, W)`: gli 82 sensori sono i "canali", le `W`
  posizioni temporali sono la "lunghezza". È un ribaltamento del layout
  `(B, W, 82)` che useremo nel `Dataset` per comodità.
- **Perché 1D-Conv e non LSTM**: la 1D-Conv è molto più veloce da
  addestrare (parallelizzabile, niente stato ricorrente) e cattura pattern
  locali (brevi trend, oscillazioni) che sono esattamente ciò che ci
  interessa in finestre corte. LSTM avrebbe senso per finestre molto
  lunghe (W ≥ 200) o se volessimo modellare dipendenze a lungo raggio, ma
  qui non serve.
- **3 layer convoluzionali a imbuto (64 → 32 → 16)**: ogni strato comprime
  gradualmente l'informazione, costruendo rappresentazioni gerarchiche.
  I primi layer catturano dettagli locali a grana fine, gli ultimi combinano
  quei dettagli in pattern più ampi.
- **Due stadi di pooling (MaxPool1d dopo layer 1 e 2)**: riducono la
  dimensione temporale di fattore 4 complessivo (W → W/2 → W/4). Questo
  richiede che W sia multiplo di 4. Il terzo layer conv NON ha pooling,
  preservando la risoluzione temporale residua per il flatten.
- **Niente AdaptiveAvgPool1d**: il flatten preserva l'intera sequenza
  temporale residua (W/4 posizioni × 16 canali = 4W elementi), dando
  allo strato denso finale accesso a tutta l'informazione spaziotemporale.
- **Decoder auto-derivato**: i canali del decoder sono l'inverso di quelli
  dell'encoder (16 → 32 → 64 → 82), con due ConvTranspose1d per i due
  stadi di upsampling. È strutturalmente impossibile disallineare encoder
  e decoder.

**Addestramento**: MAE tra input e ricostruzione, ottimizzatore Adam, batch
size 256 (da `params.yaml`).

#### 4.1.3 Estensione all'AAE

Per l'AAE, l'encoder è lo stesso del baseline. Il discriminatore riceve `z`
(latent vector di dim `latent_dim`) e deve distinguere `z ~ N(0,I)` da
`z = E(x)`. Il decoder è lo stesso del baseline. **Solo l'encoder cambia
rispetto al punto 4.1.2** (parte convoluzionale); tutto il resto della
pipeline AAE rimane identico al piano originale.

### 4.2 Normalizzazione: `StandardScaler` (z-score)
**Scelta**: media 0, std 1, fit solo sul train.

**Motivazione**:
- Feature eterogenee (potenze, angoli, temperature) hanno range incompatibili
  → senza normalizzazione il latent space viene dominato dalle feature a
  varianza maggiore.
- StandardScaler è lo standard per autoencoder (più stabile di MinMax se ci
  sono outlier, e i nostri outlier sono attesi).
- Fit solo sul train per evitare data leakage.

**Alternative considerate**:
- `MinMaxScaler` → scartato: sensibile a outlier.
- `RobustScaler` → da valutare in Fase 1 se gli outlier sono molti.
- Normalizzazione per-feature con statistiche di dominio (angoli → [-π,π]) →
  da valutare caso per caso se StandardScaler fallisce.

### 4.3 Loss di ricostruzione: MAE (scelta)
**Scelta**: MAE per il baseline. Inizialmente si era scelto MSE, ma è stata
cambiata in MAE perché i valori saturo (Gyro ±2000, Acc ±16) producevano
errori di ricostruzione al quadrato eccessivi, distorti il training e
sovrastavano le anomalie vere.

**Motivazione**:
- MAE penalizza linearmente gli errori → robusta a outlier e feature saturo.
- MSE (scomodatazione valutata) amplifica al quadrato gli errori grandi,
  causando errori di ricostruzione proibitivi quando i sensori saturano.

### 4.4 Dimensione latente: 16 (iniziale) → tuning
**Scelta iniziale**: `latent_dim=16` (da `params.yaml`).

**Motivazione**:
- Compressione 86 → 16 = fattore ~5×, sufficiente per estrarre pattern.
- Spazio latente gaussiano gestibile dal discriminatore.
- Da confrontare con 8 e 32 in una piccola grid search.

### 4.5 Prior latente: `N(0, I)` gaussiana
**Scelta**: prior standard gaussiana.

**Motivazione**:
- È il default dell'AAE originale (Makhzani 2015).
- Permette di usare KL-divergence implicita nel discriminatore.
- Alternativa: mistura di gaussiane → più espressiva ma più complessa; non
  necessaria al baseline.

### 4.6 Metriche di valutazione
**Scelte primarie**:
1. **PR-AUC** — gold standard per anomaly detection con sbilanciamento.
2. **ROC-AUC** — per completezza, ma da interpretare con cautela.
3. **F1-score** — con soglia scelta su validation, non test.
4. **Precision/Recall** a soglia operativa — perché in produzione serve
   scegliere un trade-off.

**Scelte secondarie** (per arricchire l'analisi):
- Detection latency (rilevante per time-series).
- False positive rate a soglia target.
- Distribuzione dell'errore (mean, std, percentili per classe).

> **Nota**: non usiamo solo l'accuracy perché su un dataset sbilanciato è
> fuorviante (un classificatore banale che dice sempre "normale" avrebbe
> accuracy altissima ma F1=0).

### 4.7 Scelta della soglia
- **Soglia primaria**: 99° percentile dell'errore di ricostruzione sul
  validation set (garantisce FPR ≈ 1% in condizioni normali).
- **Soglia alternativa**: soglia che massimizza F1 sul validation set
  (ottimistica, da usare come upper bound).
- **Soglie da confrontare** per mostrare il trade-off operazionale.

### 4.8 Validazione iperparametri

La validazione riduce l'incertezza sulle scelte che impattano le metriche
finali. Senza, i valori in `params.yaml` sarebbero assunzioni non verificate.

**4.8.1 Metodo di validazione**

Si usa random search vincolato (Bergstra & Bengio, 2012) con spazio di ricerca
definito a priori. N=20 run per AE, N=15 per AAE. Le metriche di validation
sono stimate su **singola fold temporale** (validation set esistente, ~47k
sample, errore standard ~0.5% per la regola `1/√N`).

**Non si usa k-fold**: k-fold temporale introduce rischio di concept drift;
k-fold random romperebbe la continuità temporale (lag-1 AC = 0.99+, vedi §1.3)
causando data leakage. La stabilità della scelta è garantita da **nested
validation**: 3 run con seed diversi sulla configurazione migliore, con
report di media ± deviazione standard.

**4.8.2 Spazio di ricerca AE**

| HP | Range | Tipo | Giustificazione del range |
|----|-------|------|---------------------------|
| `W` | {16, 32, 64, 128} | discreto | Potenze di 2 da 16 a 128 (tutti multipli di 4), default 16. Copre finestre brevi (pattern locali) a lunghe (trend lenti). Vincolo: W deve essere multiplo di 4 per i 2 stadi di pooling. |
| `latent_dim` | {8, 12, 16, 24, 32} | discreto | Compromesso compressione: input al Linear = 4×W (es. 64 per W=16, 512 per W=128) → latent_dim. |

**Architettura encoder FISSATA (non validata):**
- 3 layer convoluzionali con canali: `(64, 32, 16)` (progressione a imbuto)
- Kernel sizes: `(5, 3, 3)`
- 2 stadi di MaxPool1d(2) dopo layer 1 e 2 → riduzione temporale fattore 4
- Nessun AdaptiveAvgPool1d → Flatten preserva sequenza temporale (W/4 × 16 = 4W elementi)
- Decoder auto-derivato: canali inversi `(16, 32, 64)` + 2 ConvTranspose1d + Conv1d finale

Spazio totale: **20 combinazioni** (4 × 5), tutte campionabili esaustivamente con
`sklearn.model_selection.ParameterGrid` o `ParameterSampler(seed=42)`.

Esempio delle 20 combinazioni (griglia completa):

| run_id | W | latent_dim |
|--------|---|------------|
| 1      | 16 | 8          |
| 2      | 16 | 12         |
| 3      | 16 | 16         |
| 4      | 16 | 24         |
| 5      | 16 | 32         |
| 6      | 32 | 8          |
| 7      | 32 | 12         |
| 8      | 32 | 16         |
| 9      | 32 | 24         |
| 10     | 32 | 32         |
| 11     | 64 | 8          |
| 12     | 64 | 12         |
| 13     | 64 | 16         |
| 14     | 64 | 24         |
| 15     | 64 | 32         |
| 16     | 128 | 8         |
| 17     | 128 | 12         |
| 18     | 128 | 16         |
| 19     | 128 | 24         |
| 20     | 128 | 32         |

(Le combinazioni effettive dipendono dal seed se si usa random sampling; sopra è la griglia esaustiva.)

**4.8.3 Spazio di ricerca AAE**

HP AE-derivati (`W`, `latent_dim`) fissati a `HP_AE_best`.
L'architettura encoder (3 layer, canali 64/32/16) è fissa e non è più un iperparametro.
HP validati:

| HP | Range | Tipo | Giustificazione |
|----|-------|------|----------------|
| `reconstruction_weight` | loguniform(0.5, 2.0) | continuo | Bilanciamento ricostruzione/adv, default Makhzani 2015 non garantito per time-series |
| `adversarial_weight` | loguniform(0.01, 0.5) | continuo | Cuore AAE, range tipico 0.001–1.0 in letteratura |

15 combinazioni campionate con `ParameterSampler(seed=42)`. Esempio delle prime
5:

| run_id | reconstruction_weight | adversarial_weight |
|--------|----------------------|---------------------|
| 1      | 1.43                 | 0.087               |
| 2      | 0.71                 | 0.213               |
| 3      | 1.85                 | 0.034               |
| 4      | 0.92                 | 0.156               |
| 5      | 1.27                 | 0.412               |

**4.8.4 Iperparametri fissati senza validazione**

I seguenti iperparametri sono fissati a default di letteratura per ridurre la
dimensionalità dello spazio di ricerca. **Compromesso accettato**: potrebbero
non essere ottimali per il problema specifico; in caso di metriche deludenti
in Fase 5, diventano candidati per analisi di sensitività post-hoc.

| HP | Default | Fonte / motivazione |
|----|---------|---------------------|
| `optimizer` | Adam | Kingma & Ba, 2014 (standard de facto) |
| `learning_rate` | 1e-3 | Default Adam (Kingma & Ba, 2014) |
| `batch_size` | 256 | Compromesso standard su GPU moderne per dataset 10⁵-10⁶ |
| `loss` (reconstruction) | MAE | Robusta a feature saturo (±2000), non amplifica errori grandi al quadrato |
| `weight_decay` | 0 | Non critico per AE brevi (Goodfellow et al., 2016, §6.2) |
| `early_stopping_patience` | 10 | Standard (Goodfellow et al., 2016, §7.8) |
| `early_stopping_min_delta` | 1e-4 | EarlyStopping: stop se miglioramento < min_delta |
| `best_epoch` | tracciato in `EarlyStopping` | Ora traccia l'epoca reale del miglior val_loss (non `len(history)`) |
| `discriminator_updates_per_gen` | 1 | Default GAN (Goodfellow et al., 2014) |
| `discriminator_lr` | 1e-3 | Stesso di E+D, default comune nelle implementazioni AAE |
| **Architettura Encoder (FIXA)** | | |
| `encoder_num_layers` | 3 | Scelto per rappresentazioni gerarchiche più ricche (vedi §4.1.2) |
| `encoder_conv_channels` | (64, 32, 16) | Progressione a imbuto: compressione graduale da 82 sensori verso 16 |
| `encoder_conv_kernels` | (5, 3, 3) | Kernel ampio all'inizio per contesto largo, poi stretti per dettagli |
| `encoder_pool_stages` | 2 | MaxPool1d(2) dopo layer 1 e 2 → riduzione temporale ×4 |
| `encoder_pool_size` | 2 | Fattore di pooling per stadio |
| `adaptive_avg_pool` | False | Rimosso: Flatten preserva informazione temporale per Linear finale |
| **Architettura Decoder (AUTO-DERIVATA)** | | |
| `decoder_channels` | (16, 32, 64, 82) | Inverso di encoder_channels + input_dim finale |
| `decoder_upsample_stages` | 2 | ConvTranspose1d per ogni stadio di pooling dell'encoder |
| `decoder_final_kernel` | 3 | Conv1d finale per proiezione a input_dim |

**4.8.5 Workflow CLI**

Tutta la validazione è eseguita via script Python (`.py`), non notebook, per
compatibilità con HPC e parallelizzazione. I notebook sono usati solo per
ispezione visiva dei CSV.

```bash
# AE
python -m src.validation.run_search_ae --n-iter 20 --seed 42 --no-resume
python -m src.validation.analyze_results --input validation_results_ae.csv
python -m src.models.train_ae --config params_validated_ae.yaml

# AAE
python -m src.validation.run_search_aae --n-iter 15 --seed 42 --no-resume
python -m src.validation.analyze_results --input validation_results_aae.csv
python -m src.models.train_aae --config params_validated_aae.yaml
```

> **Importante `--no-resume`**: il default di `run_search_ae.py` è
> `resume=False` (cambiato da `True` in Fase 3.1 per evitare che run_id già
> presenti nel CSV vengano saltati). Lo script SLURM `slurm_ae_search.sh`
> passa sempre `--no-resume` per garantire che le 50 epoche vengano eseguite.

**4.8.6 Parallelizzazione**

Ogni run è un processo Python indipendente. Su HPC, job array SLURM con
`CUDA_VISIBLE_DEVICES=$((SLURM_ARRAY_TASK_ID % num_gpus))`. Il numero di GPU
è determinato a runtime via `nvidia-smi`. In locale, `ProcessPoolExecutor` o
`&`. Il CSV finale è scritto in append (`mode='a'`) per garantire robustezza
a crash.

Sul cluster, usa `--no-resume` per forzare il complete rewrite del CSV:
```bash
cd hpc && N_ITER=20 MAX_VAL_EPOCHS=50 ./hpc_connect.sh batch slurm_ae_search.sh
```


**4.8.7 Recovery condizionale post-validazione AAE**

Dopo aver selezionato `HP_AAE_best`, si confronta `PR-AUC_AAE` vs
`PR-AUC_AE`:

- Se `PR-AUC_AAE ≥ PR-AUC_AE - 0.05`: la stima "HP AE trasferiti" è
  sufficiente, procedere a training finale (4.2).
- Se `PR-AUC_AAE < PR-AUC_AE - 0.05`: recovery con 10-15 run AAE con
  **tutti** gli HP AE-derivati liberi (non solo AAE-specifici), per
  identificare se qualche HP AE non è adatto all'AAE.

**4.8.8 Formato degli output e criterio di selezione**

`reports/tables/validation_results_ae.csv`:

```csv
run_id,W,latent_dim,encoder_channels,best_val_pr_auc,best_val_f1,best_epoch,train_time_sec,seed
1,24,16,[128, 64],0.852,0.781,12,341,42
2,12,32,[64, 32],0.834,0.762,15,298,42
...
```

`HP_AE_best` = riga con `best_val_pr_auc` massimo. Tie-break su
`best_val_f1`, poi `best_epoch` (preferenza per convergenza più rapida).

Stesso formato per `validation_results_aae.csv`, con colonne
`reconstruction_weight` e `adversarial_weight` al posto di quelle AE.

Output finali:
- `reports/tables/validation_results_ae.csv`
- `reports/tables/validation_results_aae.csv`
- `reports/figures/sensitivity_ae_W.png`, `sensitivity_ae_latent_dim.png`,
  `sensitivity_ae_encoder_channels.png`
- `reports/figures/sensitivity_aae_reconstruction_weight.png`,
  `sensitivity_aae_adversarial_weight.png`
- `config/params_validated_ae.yaml`
- `config/params_validated_aae.yaml`

---

## 5. Struttura del repository

```
AM01-.../
├── config/                          # Configurazioni YAML
│   ├── config.yaml                  # Setup generale (device, paths, logging)
│   ├── params.yaml                  # Iperparametri default (Fase 2)
│   ├── params_validated_ae.yaml     # [output Fase 3.1] HP AE validati
│   └── params_validated_aae.yaml    # [output Fase 4.1] HP AAE validati
│
├── data/
│   ├── raw/                         # Dati originali (versionati o via DVC)
│   │   └── KukaVelocityDataset/
│   └── processed/                   # Output del preprocessing
│       ├── train.npy                # 70% KukaNormal (~163k, 82 features)
│       ├── val_normal.npy           # 15% KukaNormal (~35k, early stopping + threshold)
│       ├── val_anomaly.npy          # 50% KukaSlow (~21k, HP selection)
│       ├── test_normal.npy          # 15% KukaNormal (~35k, final report)
│       ├── test_anomaly.npy         # 50% KukaSlow (~21k, final report)
│       ├── scaler.pkl               # StandardScaler fitted on train
│       └── selected_columns.npy     # 82 column names
│
├── src/                             # Codice di produzione (importabile, testabile)
│   ├── data/
│   │   ├── dataset.py               # torch.utils.data.Dataset
│   │   └── preprocessing.py         # Pipeline preprocessing
│   ├── models/
│   │   ├── autoencoder.py           # Baseline AE (costruttore parametrico)
│   │   ├── adversarial_ae.py        # AAE (costruttore parametrico)
│   │   ├── train_utils.py           # train/validate/EarlyStopping
│   │   ├── train_ae.py              # CLI: training finale AE
│   │   ├── train_aae.py             # CLI: training finale AAE
│   │   └── compare_models.py        # Logica di confronto
│   ├── validation/                  # Pipeline di validazione iperparametri
│   │   ├── search_space.py          # define_search_space_ae/aae
│   │   ├── run_experiment_ae.py     # train_and_evaluate_ae(config)
│   │   ├── run_experiment_aae.py    # train_and_evaluate_aae(config)
│   │   ├── run_search_ae.py         # CLI N run random AE
│   │   ├── run_search_aae.py        # CLI N run random AAE
│   │   └── analyze_results.py       # CLI: tabelle + grafici da CSV
│   ├── utils/
│   │   ├── metrics.py               # PR-AUC, ROC-AUC, F1, ...
│   │   ├── visualization.py         # Plot ROC, PR, distribuzioni
│   │   └── config.py                # Loader YAML
│   └── main.py                      # Entry point CLI
│
├── notebooks/                       # SOLO esplorazione + ispezione visiva CSV
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_inspect_validation_results.ipynb   # ispezione CSV validazione
│
├── tests/                           # pytest
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── reports/                         # Output finali
│   ├── checkpoints/                 # .pth serializzati
│   ├── figures/                     # PNG per il report
│   └── tables/                      # CSV/TeX per il report
│
├── docs/
│   ├── Projects Topics Presentation.pdf
│   ├── Project_proposal_template_2026.docx
│   ├── methodology.md               # Idea iniziale (mantenere per storia)
│   └── project_plan.md              # ← QUESTO FILE
│
├── main.py                       # Wrapper sottile che chiama src/main.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

### 5.1 Regola: notebook ≠ codice di produzione
- **Notebook** (`notebooks/`) → esplorazione, visualizzazione, prototipazione.
  Cosa si impara, non cosa si consegna.
- **Moduli** (`src/`) → codice di produzione, importato da notebook, da test e
  da `main.py`. Cosa si consegna.

Regola pratica: se una cella di notebook inizia a essere chiamata da altre
celle, va spostata in un modulo `.py`.

---

## 6. Stato di avanzamento (checklist)

> Da aggiornare a ogni sessione. Le checkbox riflettono lo stato reale.

### Fase 1 — Esplorazione (COMPLETATA)
- [x] Caricamento 3 `.npy` e shape/dtype check
- [x] Identificazione 87ª colonna di `KukaSlow` (`anomaly`, label binaria sempre 1)
- [x] Statistiche descrittive per feature
- [x] Confronto distribuzioni Normal vs Slow
- [x] Heatmap correlazioni (Spearman, max 0.9576)
- [x] PCA / t-SNE 2D (PC1=10.4%, PC2=10.2%, 10 PC≈55% varianza cumulativa)
- [x] Verifica ordinamento temporale → **split temporale** (lag-1 AC=0.99+)
- [x] Decisioni scritte in fondo al notebook 01 (vedi §7 per risoluzioni)
- [x] Correzione path assoluto `DATA_PATH` (nbconvert CWD ≠ project root)
- [x] Correzione `boxplot()` API `labels` → `tick_labels` (matplotlib ≥3.6)
- [x] Correzione directory `reports/figures/` creata con `os.makedirs`
- [x] 4 figure generate in `reports/figures/fase1_*.png`

### Fase 2 — Preprocessing
- [x] Gestione colonna extra (`anomaly` rimossa da KukaSlow: 87→86)
- [x] Pulizia NaN/inf/outlier (nessun NaN/inf nel raw; saturazione sensore gestita da scaler)
- [x] Split deterministico temporale (70/15/15 normal, 50/50 slow, no shuffle)
- [x] Normalizzazione (StandardScaler, fit su train only)
- [x] `src/data/preprocessing.py` implementato (load → split → normalize → save)
- [x] `src/data/dataset.py` implementato (KukaDataset, windowing on-the-fly)
- [x] Test `tests/test_dataset.py` (33 test, tutti passanti)
- [x] `notebooks/02_preprocessing.ipynb` popolato (6 celle)
- [x] `src/utils/config.py` creato (loader YAML unificato)
- [x] `src/main.py` refactorato (usa config.py, flag --phase)
- [x] Output generati in `data/processed/` (5 .npy + scaler.pkl + selected_columns.npy)
- [x] Verifica locale: pipeline end-to-end ✅, pytest 33/33 ✅

### Fase 3.0 — Costruzione AE parametrico (COMPLETATA)
- [x] `src/models/autoencoder.py` (Encoder, Decoder, Autoencoder — costruttore parametrico, `EarlyStopping` con `best_epoch`)
- [x] `src/models/train_utils.py` (train_one_epoch, validate, EarlyStopping)
- [x] `src/validation/run_experiment_ae.py` (`train_and_evaluate_ae(config)`)
- [x] `src/validation/run_search_ae.py` (CLI per N run random, default `--no-resume`)
- [x] Test 1 run (3-5 epoche) verificata

### Fase 3.1 — Validazione AE (COMPLETATA)
- [x] `python -m src.validation.run_search_ae --n-iter 20 --seed 42 --no-resume`
- [x] `reports/tables/validation_results_ae.csv` (20 righe, 50 epoche garantite)
- [x] `python -m src.validation.analyze_results` → 3 grafici sensitività
- [x] `config/params_validated_ae.yaml`
- [x] Selezione `HP_AE_best` (riga con PR-AUC max, vedi §4.8.8)

### Fase 3.2 — Training finale AE (COMPLETATA)
- [x] `python -m src.models.train_ae --config params_validated_ae.yaml`
- [x] Nested validation (3 run con seed diversi) → media ± std
- [x] `reports/checkpoints/ae_baseline.pth`
- [x] `reports/tables/ae_final_metrics.csv`

### Fase 4.0 — Costruzione AAE parametrico
- [ ] `src/models/adversarial_ae.py` (Discriminator, AdversarialAE)
- [ ] `src/validation/run_experiment_aae.py`
- [ ] `src/validation/run_search_aae.py` (CLI)
- [ ] Test 1 run (3-5 epoche) per verificare pipeline

### Fase 4.1 — Validazione AAE
- [ ] `python -m src.validation.run_search_aae --n-iter 15 --seed 42`
- [ ] `reports/tables/validation_results_aae.csv` (15 righe)
- [ ] `python -m src.validation.analyze_results --aae` → 2 grafici sensitività
- [ ] Sanity check AAE vs AE → recovery condizionale se PR-AUC_AAE < PR-AUC_AE - 0.05
- [ ] `config/params_validated_aae.yaml`
- [ ] Selezione `HP_AAE_best`

### Fase 4.2 — Training finale AAE
- [ ] `python -m src.models.train_aae --config params_validated_aae.yaml`
- [ ] Nested validation (3 run con seed diversi)
- [ ] `reports/checkpoints/aae_final.pth`
- [ ] `reports/tables/aae_final_metrics.csv`

### Fase 5 — Valutazione
- [ ] `src/utils/metrics.py` completo
- [ ] Distribuzione errori per classe (AE e AAE)
- [ ] Scelta soglia su validation
- [ ] PR-AUC, ROC-AUC, F1, Precision, Recall
- [ ] Confronto tabellare AE vs AAE
- [ ] Test statistico (McNemar / paired bootstrap)
- [ ] Visualizzazioni comparative
- [ ] `reports/tables/metrics_comparison.csv` e `.tex`
- [ ] `reports/figures/*.png`

### Fase 6 — Riproducibilità
- [ ] `tests/test_models.py` (forward pass, count param)
- [ ] `tests/test_metrics.py` (valori noti)
- [ ] `src/main.py` CLI funzionante
- [ ] README con istruzioni di riproduzione
- [ ] Seed fissati ovunque

---

## 7. HPC workflow

Il deployment e il training avvengono sul cluster Legion (SLURM). Il workflow è diviso in:

| Fase | Comando locale | Script SLURM | Output |
|---|---|---|---|
| Setup | `./hpc_connect.sh deploy` | `setup_env.sh` | `~/am01_project/.venv/`, kernel Jupyter |
| Data upload | `./hpc_connect.sh upload data/raw/ ~/am01_project/data/raw/` | — | `data/raw/*.npy` su HPC |
| Search | `./hpc_connect.sh batch slurm_ae_search.sh` | `slurm_ae_search.sh` | `validation_results_ae.csv`, 3 plots, `params_validated_ae.yaml` |
| Training | `./hpc_connect.sh batch slurm_ae_train.sh` | `slurm_ae_train.sh` | `ae_baseline.pth`, `ae_final_metrics.csv` |

**Dove esegue cosa**: il preprocessing e il training girano su **compute node** (GPU A40),
nessun carico sul login node. `hpc_connect.sh batch` esegue deploy (rsync) → sbatch →
stream log → auto-fetch risultati. Il preprocessing è triggerato automaticamente
se `data/processed/train.npy` è mancante (sui script SLURM).

## Decisioni aperte (backlog)

> Domande la cui risposta cambierà il codice. Da affrontare nell'ordine in cui
> emergono durante l'esecuzione.

1. ✅ **87ª colonna di `KukaSlow` — che cos'è?** È `anomaly`, una label binaria
   (valore sempre 1) aggiunta SOLO in KukaSlow per marcare la classe anomala.
   Non è una feature: contiene informazione ridondante (sappiamo già che
   KukaSlow = anomalo). Rimossa in Fase 2 per allineare le matrici a 86 colonne.
   - **Alternativa**: tenerla e usarla come feature → rifiutata: fornirebbe
     l'etichetta al modello, un data leakage insorto.
2. ✅ **Feature costanti — che cos'sono?** Colonne il cui valore non varia mai
   nel dataset (std ≈ 0). Identificate in Fase 1: `sensor_id{2,5,6,7}_temp`
   (std ~1e-10). Rimosse in Fase 2 perché non portano informazione
   (un sensore rotto o un valore fisso). Verificate costanti in ENTRAMBI i
   dataset. Correlazione > 0.95 → valutare rimozione feature correlate (Fase 2).
3. ✅ **Split temporale vs random — definizione**:
   - **Temporale**: i dati vengono divisi mantenendo l'ordine cronologico
     (primi 70% → train, successivi 15% → val_normal, ultimi 15% → test_normal).
     Per KukaSlow: primi 50% → val_anomaly, ultimi 50% → test_anomaly.
   - **Random**: mescola i dati con un seed prima dello split.
   Scelta: **temporale**. Confermato in Fase 1: lag-1 AC = 0.99+ su tutte le
   feature → i dati sono una sessione continua. Lo shuffle romperebbe la
   continuità e causerebbe leakage. Split normal 70/15/15, slow 50/50.
4. **Dimensione della finestra W — definizione**: numero di timestep consecutivi
   dati in pasto alla rete ad ogni sample (slide con stride=1). Default `W=16`.
   In Fase 3 si confrontano `W ∈ {8, 16, 32, 64}` su validation set → si
   conferma il valore migliore per trade-off errore di ricostruzione / costo
   computazionale. Vedi §4.1.1.
5. **MAE vs MSE — definizione** (MAE scelta, MSE valutata):
   - **MAE** (Mean Absolute Error): penalizza linearmente, robusto a outlier e
     satura — **loss scelta** (config: `training.loss: "mae"`).
   - **MSE** (Mean Squared Error): penalizza quadraticamente gli errori grandi.
   Confronto empirico in Fase 5.
6. **Dimensione latente ottimale — definizione**: numero di dimensioni del
   vettore latente z. Grid search su {8, 16, 32} in Fase 5.
7. **Quanti training run — definizione**: numero di volte si addestra il modello.
   Almeno 1 AE + 1 AAE, idealmente 3+ run ciascuno con seed diversi per stima
   varianza e test statistico.
8. **Modelli di confronto extra — definizione**: baseline non-deep (Isolation
   Forest, OC-SVM) da confrontare con AE/AAE. → facoltativo, da valutare se
   avanza tempo.
9. ✅ **Clipping outlier — che cos'è?** Windowing che "raddrizzà" i valori
   estremi a un percentile (es. p1 e p99): ogni valore sotto p1 diventa p1,
   ogni valore sopra p99 diventa p99. Viene fatto **dopo lo split** e **i
   bound sono calcolati su train only** (nessun leakage).
   **Deciso: NON applicare clipping.** I valori di saturazione sensore (Gyro
   ±2000, Acc ±16) sono artefatti hardware, non anomalie da rilevare. Lo
   StandardScaler li assorbe (z-score alti ma finiti) e la MAE loss gestisce
   linearmente le finestre con spike saturati senza amplificarli al quadrato.
   La classe anomala si caratterizza da drift lenti, non da spike — il clipping
   non aiuterebbe e potrebbe nascondere pattern utili.
   → Pipeline: load → split → normalize → save (NO clipping step).
10. ✅ **Validare gli iperparametri dei modelli?** Sì, con random search
    vincolato (Bergstra & Bengio, 2012) su 3 HP AE (`W`, `latent_dim`,
    `encoder_channels`) e 2 HP AAE-specifici (`reconstruction_weight`,
    `adversarial_weight`). Fissati senza validazione per letteratura:
    optimizer, learning_rate, batch_size, loss, weight_decay, early stopping
    patience, discriminator_updates_per_gen, discriminator_lr
    (vedi §4.8.4 per fonti). Compromesso accettato: questi HP potrebbero non
    essere ottimali, in caso di metriche deludenti in Fase 5 diventano
    candidati per analisi di sensitività post-hoc.
11. ✅ **CLI-first vs notebook per la validazione?** CLI-first. Cluster HPC
    esegue script `.py`; notebook usati solo per ispezione visiva dei CSV.
    Ogni run è un processo indipendente → parallelizzabile con job array
    SLURM su GPU multiple (vedi §4.8.6). CSV scritto in append per
    robustezza a crash.
12. ✅ **k-fold vs single fold per la validazione degli HP?** Single fold
    temporale sul validation set esistente (~47k sample, errore standard
    ~0.5% per la regola `1/√N`). k-fold temporale introdurrebbe rischio di
    concept drift; k-fold random romperebbe la continuità temporale (lag-1
    AC = 0.99+, vedi §1.3) causando data leakage. La stabilità della scelta
    è garantita dalla nested validation (3 run con seed diversi sulla
    configurazione selezionata).
13. ✅ **Freeze parziale HP AE → AAE?** Sì. HP AE-derivati (`W`, `latent_dim`,
    `encoder_channels`) vengono ereditati dall'AE senza ri-validazione. Solo
    2 HP AAE-specifici validati ex novo. Recovery condizionale se
    `PR-AUC_AAE < PR-AUC_AE - 0.05`: 10-15 run AAE con tutti gli HP AE
    liberi per identificare interazioni avverse (vedi §4.8.7).

---

## 8. Riferimenti e materiali

- **Traccia del corso**: `docs/Projects Topics Presentation.pdf`, pp. 19–20.
- **Riferimento chiave**: Kim S. et al., "Towards a Rigorous Evaluation of
  Time-series Anomaly Detection", 2022.
- **AAE originale**: Makhzani, A. et al., "Adversarial Autoencoders", 2015.
- **Dataset**: Kuka Velocity Dataset (incluso).

---

## 9. Note di sessione (logbook)

> Sezione libera per annotare *cosa abbiamo fatto oggi*, *cosa non ha
> funzionato*, *idee emerse*. Da consultare a inizio sessione per orientarsi.

### Sessione 1 — Fase 1: Esplorazione dati (25–30ago2026)

**Processo (ordine di lavoro):**
1. Caricamento 3 .npy → identificata la 87ª colonna di `KukaSlow` come `anomaly`
   (label binaria, sempre 1) → non è una feature.
2. Statistiche descrittive → identificate 4 feature costanti
   (`sensor_id{2,5,6,7}_temp`, std ~1e-10) → da rimuovere in Fase 2.
3. Lag-1 autocorrelation = 0.99+ su tutte le feature → dati ordinati temporalmente
   → scelta dello **split temporale** (no shuffle).
4. Distribuzioni Normal vs Slow → confermato StandardScaler (std range 0.01–51.8);
   nessun NaN/inf nel raw.
5. PCA (PC1=10.4%) + t-SNE → nessuna feature domina linearmente.

**Decisioni chiuse:**
- 87ª colonna = `anomaly` → rimossa (etichetta, non feature).
- 4 feature costanti → rimosse in Fase 2.
- Split = temporale (60/20/20).
- Normalizzazione = StandardScaler (fit su train only).
- W = 16 (default, validazione in Fase 3). Input dim = 82.
- Correlazione max Spearman = 0.9576 → valutare rimozione feature correlate (Fase 2).

**Output:** 4 figure in `reports/figures/fase1_*.png`.

**Prossima fase:** Fase 2 — preprocessing.

### Sessione 2 — Fase 2: Preprocessing (31ago2026 – 1set2026)

**Processo (ordine di lavoro):**
1. **Scelta clipping**: valutata la possibilità di clippare i valori di saturazione
   sensore (Gyro ±2000, Acc ±16). **Deciso: NO** — artefatti hardware, la classe
   anomala si caratterizza da drift lenti non da spike, StandardScaler + MAE li
   gestiscono. Pipeline = load → split → normalize → save.
2. **`src/utils/config.py`**: creato il loader YAML unificato (config.yaml +
   params.yaml). Path root con `parents[2]` (utils/ → src/ → root).
3. **`src/data/preprocessing.py`**: 6 funzioni. `load_kuka_data` elimina
   `anomaly` (87→86) e le 4 feature costanti (86→82), verificando su ENTRAMBI i
   dataset. `normalize_data` usa StandardScaler fit su train only + asserzioni
   anti-leakage.
4. **`src/data/dataset.py`**: `KukaDataset` con windowing on-the-fly (2.9 GB →
   0.18 GB). Windowing separato per split → nessun crossover normal/anomaly.
5. **`tests/test_dataset.py`**: 33 test, tutti passanti. `sensor_id3_temp`
   costante in KukaSlow ma non in KukaNormal → tenuta (potenzialmente
   discriminante). `sensor_id4_temp` costante nel train split → StandardScaler
   gestisce (scale=1).
6. **Refactoring**: `main.py` usa `config.py` + flag `--phase`; docstring
   snelliti; `params.yaml` commenti in inglese; notebook con commenti guidati.
7. **Documentazione**: §7 #9 e §9 riscritti in linguaggio neutro ("deciso
   perché" invece di "rifiutato dall'utente").

**Output:** 4 .npy + scaler.pkl + selected_columns.npy in `data/processed/`.
**Verifica:** pytest 33/33 ✅; pipeline end-to-end ~1s.

**Prossima fase:** Fase 3 — baseline Autoencoder (1D-Conv encoder + decoder speculare).

### Sessione 3 — Fase 3: Costruzione AE + validation + training + HPC fixes (4–5set2026)

**Processo:**
1. **Costruzione AE parametrico**: implementati `autoencoder.py` (SequenceAutoencoder con
   costruttore parametrico `from_config`), `train_utils.py` (EarlyStopping, train_one_epoch),
   `run_experiment_ae.py` (`train_and_evaluate_ae`), `run_search_ae.py` (CLI per random search).

2. **Bug `best_epoch`**: in `run_experiment_ae.py` e `train_ae.py`, `best_epoch` era
   `len(history["val_loss"])` (totale epoche). Fix: `EarlyStopping` traccia
   `self.best_epoch` (l'epoca del miglior val_loss), passata via `history["best_epoch"]`.

3. **Bug resume + 50 epoche**: `run_search_ae.py` aveva `resume=True` per default,
   saltando run_id già nel CSV. Fix: default `resume=False`, flag `--no-resume`
   nello script SLURM. Il CSV obsoleto (34 righe, 2 epoche) è stato rimosso da `reports/`
   e `hpc/reports/`.

4. **Bug patience hardcoded**: `run_experiment_ae.py` aveva `patience=5` hardcoded.
   Fix: letto da `config["training"]["early_stopping"]["patience"]` (valore 10).
   Aggiunto `min_delta` da config.

5. **HPC execution fixes (7 problemi)**:
   - **Data loading**: `data/raw/` ora incluso nel deploy rsync (era escluso).
   - **Preprocessing location**: eseguito su compute node (HPC-side) dopo sync
     a `$SCRATCH`, triggerato automaticamente se `data/processed/` mancante.
   - **Config errata**: 50 epoche garantite con `--no-resume` + `resume=False` default.
   - **Overwrite**: rimosso `--ignore-existing` da tutti i rsync; fixato filtro
     `--include='*' --exclude='*'` (no-op) in `slurm_ae_search.sh`.
   - **"no log file found"**: SLURM log va in `~/jobs/logs/`, non in scratch.
     Fixato `cp` path negli script SLURM + fallback in `cmd_batch`.
   - **Results scattered**: eliminate `hpc/reports/` e `hpc/data/`, aggiunte a `.gitignore`.
   - **Data scattered**: `hpc/data/` eliminato, dati solo in `data/`.

6. **Verifica locale**: search 5-iter, 10-epoch, `--no-resume` → CSV 5 righe pulite,
   `best_epoch` > 2. Training 1 seed → checkpoint + CSV metriche.

**Outputs:** `reports/tables/validation_results_ae.csv`, `reports/figures/sensitivity_ae_*.png`,
`config/params_validated_ae.yaml`, `reports/tables/ae_final_metrics.csv`, `reports/checkpoints/ae_baseline.pth`.

**Prossima fase:** Fase 4 — AAE (adversarial autoencoder).

### Sessione 5 — Fix validation split & HP selection leakage (30set2026)

**Problema identificato**: Il flusso di validazione precedente aveva due problemi critici:
1. **HP selection usava test set** → `best_val_pr_auc` proveniva dal test set (data leakage totale)
2. **Validation solo normali** → impossibile calcolare PR-AUC/ROC-AUC senza anomalie
3. **Split 60/20/20 non conforme** → specifica richiedeva 70/15/15 normal + 50/50 slow

**Soluzione implementata**:
- **Nuovi split** (`config/params.yaml`): `train_split=0.70`, `val_split=0.15`, `slow_val_split=0.50`
- **Preprocessing** (`src/data/preprocessing.py`): split normal 70/15/15, slow 50/50 → 5 file .npy separati
- **Validation experiment** (`src/validation/run_experiment_ae.py`):
  - 5 DataLoader separati (train, val_normal, val_anomaly, test_normal, test_anomaly)
  - Early stopping su `val_normal` (reconstruction loss)
  - HP selection: threshold (99° percentile su val_normal) + PR-AUC/ROC-AUC/F1 su val_normal + val_anomaly
  - Test finale: metriche su test_normal + test_anomaly (solo report)
  - `best_val_pr_auc` ora proviene da validation set (non test!)
- **Leakage accettato**: `val_normal` serve per early stopping E threshold/PR-AUC (standard practice ML); test set rimane unseen

**File modificati**:
- `config/params.yaml`
- `src/data/preprocessing.py` (split_temporal_data, normalize_data)
- `src/validation/run_experiment_ae.py` (5 loader, _compute_errors_for_loader, train_and_evaluate_ae)

**Prossima fase**: Rigenerare processed data e lanciare HP search con nuova logica.
