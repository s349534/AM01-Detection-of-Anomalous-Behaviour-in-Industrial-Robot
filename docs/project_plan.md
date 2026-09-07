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
  (60/20/20), più tutto `KukaSlow` aggiunto al test set come classe anomala.
- **Training (60% di KukaNormal, ~140k)**: il modello impara la distribuzione
  normale.
- **Validation (20% di KukaNormal, ~47k)**: per early stopping + per calibrare
  la soglia di decisione (es. 99° percentile dell'errore di ricostruzione).
- **Test set finale**:
  - 20% di KukaNormal (~47k) → classe "normale" per le metriche.
  - Tutto KukaSlow (~41k) → classe "anomala" per le metriche.
  - Totale ~88k campioni etichettati.

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
  StandardScaler + MSE loss (vedi §7 #9).
- **Split**: solo `KukaNormal` per train+val (60/20/20, no shuffle); test =
  Normal hold-out + tutto `KukaSlow` (classe anomala).
- **Normalizzazione**: `StandardScaler` (fit **solo** sul train) — motivazione §4.
- Windowing on-the-fly in `KukaDataset` (non pre-computato) — motivazione §4.1.
- Salvataggio:
  - `data/processed/{train, val, test_normal, test_anomaly}.npy`
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
- Output: `reports/tables/validation_results_ae.csv` (20 righe).
- Analisi: `python -m src.validation.analyze_results` → 3 grafici di
  sensitività (`sensitivity_ae_*.png`).
- Selezione: riga con `best_val_pr_auc` massimo (vedi §4.8.8).
- Output finale: `config/params_validated_ae.yaml`.

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

**Architettura scelta: Encoder convoluzionale 1D + Decoder speculare.**

```
ENCODER                                    DECODER
─────────────────────────                 ─────────────────────────
Input  (B, 86, W)                          Input  (B, latent_dim)
   ↓                                          ↓
Conv1d(86→128, kernel=5, pad=2)            Linear(latent → 64·W')
   ↓ ReLU                                    ↓ ReLU
MaxPool1d(2)  → (B, 128, W/2)             Reshape → (B, 64, W')
   ↓                                          ↓
Conv1d(128→64, kernel=3, pad=1)           ConvTranspose1d(64→128, kernel=4, stride=2)
   ↓ ReLU                                    ↓ ReLU
AdaptiveAvgPool1d(1) → (B, 64, 1)         Conv1d(128→86, kernel=3, pad=1)
   ↓                                          ↓
Flatten → (B, 64)                          Output (B, 86, W)
   ↓                                          ↓
Linear(64 → latent_dim)                    → ricostruzione della sequenza
   ↓                                          → errore = MSE(x, x̂)
z  (B, latent_dim)
```

**Note sull'architettura**:

- **Input layout**: PyTorch `Conv1d` vuole `(batch, channels, length)`,
  quindi passiamo `(B, 86, W)`: gli 86 sensori sono i "canali", le `W`
  posizioni temporali sono la "lunghezza". È un ribaltamento del layout
  `(B, W, 86)` che useremo nel `Dataset` per comodità.
- **Perché 1D-Conv e non LSTM**: la 1D-Conv è molto più veloce da
  addestrare (parallelizzabile, niente stato ricorrente) e cattura pattern
  locali (brevi trend, oscillazioni) che sono esattamente ciò che ci
  interessa in finestre corte. LSTM avrebbe senso per finestre molto
  lunghe (W ≥ 200) o se volessimo modellare dipendenze a lungo raggio, ma
  qui non serve.
- **AdaptiveAvgPool1d(1)**: comprime la dimensione temporale a 1 dopo le
  conv, così il Linear finale riceve un vettore di lunghezza fissa
  indipendente da W. Vantaggio: posso cambiare W senza ridisegnare
  l'encoder.
- **Decoder speculare**: ConvTranspose1d fa l'upsampling temporale. Per
  finestre piccole (W=16) la ricostruzione è praticamente perfetta sui
  sample normali → l'errore è guidato quasi solo dalle anomalie.

**Addestramento**: MSE tra input e ricostruzione, ottimizzatore Adam, batch
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

### 4.3 Loss di ricostruzione: MSE (poi MAE da confrontare)
**Scelta iniziale**: MSE per il baseline; MAE come confronto se MSE produce
code troppo pesanti per le anomalie.

**Motivazione**:
- MSE penalizza di più gli errori grandi → enfatizza le anomalie, che è
  *esattamente* ciò che vogliamo in anomaly detection.
- MAE è più robusto a outlier → utile in scenari industriali rumorosi.

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
| `W` | {8, 12, 16, 24, 32} | discreto | Limiti inferiore/superiore vincolati da Fase 1 (frequenza di campionamento, costo computazionale) |
| `latent_dim` | {8, 12, 16, 24, 32} | discreto | Compromesso compressione 82→{8..32} = 2.5×–10× |
| `encoder_channels` | {[64,32], [128,64], [128,64,32]} | categorico | 2 o 3 layer, range standard per AE su dati 1D |

Spazio totale: 75 combinazioni, 20 campionate con
`sklearn.model_selection.ParameterSampler(seed=42)`. Esempio delle prime 10
combinazioni campionate:

| run_id | W | latent_dim | encoder_channels |
|--------|---|------------|------------------|
| 1      | 24 | 16         | [128, 64]        |
| 2      | 12 | 32         | [64, 32]         |
| 3      | 16 | 8          | [128, 64, 32]    |
| 4      | 8  | 24         | [128, 64]        |
| 5      | 32 | 12         | [64, 32]         |
| 6      | 24 | 32         | [128, 64, 32]    |
| 7      | 16 | 16         | [64, 32]         |
| 8      | 12 | 8          | [128, 64]        |
| 9      | 32 | 24         | [128, 64]        |
| 10     | 8  | 12         | [128, 64, 32]    |

(Le combinazioni effettive dipendono dal seed; sopra è un'illustrazione del
formato. Le 20 run prodotte andranno a popolare il CSV finale.)

**4.8.3 Spazio di ricerca AAE**

HP AE-derivati (`W`, `latent_dim`, `encoder_channels`) fissati a `HP_AE_best`.
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
| `loss` (reconstruction) | MSE | Massima verosimiglianza gaussiana, enfatizza outlier (anomalie) |
| `weight_decay` | 0 | Non critico per AE brevi (Goodfellow et al., 2016, §6.2) |
| `early_stopping_patience` | 5 | Standard (Goodfellow et al., 2016, §7.8) |
| `discriminator_updates_per_gen` | 1 | Default GAN (Goodfellow et al., 2014) |
| `discriminator_lr` | 1e-3 | Stesso di E+D, default comune nelle implementazioni AAE |

**4.8.5 Workflow CLI**

Tutta la validazione è eseguita via script Python (`.py`), non notebook, per
compatibilità con HPC e parallelizzazione. I notebook sono usati solo per
ispezione visiva dei CSV.

```bash
# AE
python -m src.validation.run_search_ae --n-iter 20 --seed 42
python -m src.validation.analyze_results --input validation_results_ae.csv
python -m src.models.train_ae --config params_validated_ae.yaml

# AAE
python -m src.validation.run_search_aae --n-iter 15 --seed 42
python -m src.validation.analyze_results --input validation_results_aae.csv
python -m src.models.train_aae --config params_validated_aae.yaml
```

**4.8.6 Parallelizzazione**

Ogni run è un processo Python indipendente. Su HPC, job array SLURM con
`CUDA_VISIBLE_DEVICES=$((SLURM_ARRAY_TASK_ID % num_gpus))`. Il numero di GPU
è determinato a runtime via `nvidia-smi`. In locale, `ProcessPoolExecutor` o
`&`. Il CSV finale è scritto in append (`mode='a'`) per garantire robustezza
a crash.

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
- [x] Split deterministico temporale (60/20/20, no shuffle)
- [x] Normalizzazione (StandardScaler, fit su train only)
- [x] `src/data/preprocessing.py` implementato (load → split → normalize → save)
- [x] `src/data/dataset.py` implementato (KukaDataset, windowing on-the-fly)
- [x] Test `tests/test_dataset.py` (33 test, tutti passanti)
- [x] `notebooks/02_preprocessing.ipynb` popolato (6 celle)
- [x] `src/utils/config.py` creato (loader YAML unificato)
- [x] `src/main.py` refactorato (usa config.py, flag --phase)
- [x] Output generati in `data/processed/` (4 .npy + scaler.pkl + selected_columns.npy)
- [x] Verifica locale: pipeline end-to-end ✅, pytest 33/33 ✅

### Fase 3.0 — Costruzione AE parametrico
- [ ] `src/models/autoencoder.py` (Encoder, Decoder, Autoencoder — costruttore parametrico)
- [ ] `src/models/train_utils.py` (train_one_epoch, validate, EarlyStopping)
- [ ] `src/validation/run_experiment_ae.py` (`train_and_evaluate_ae(config)`)
- [ ] `src/validation/run_search_ae.py` (CLI per N run random)
- [ ] Test 1 run (3-5 epoche) su 3-5 epoche per verificare pipeline

### Fase 3.1 — Validazione AE
- [ ] `python -m src.validation.run_search_ae --n-iter 20 --seed 42`
- [ ] `reports/tables/validation_results_ae.csv` (20 righe)
- [ ] `python -m src.validation.analyze_results` → 3 grafici sensitività
- [ ] `config/params_validated_ae.yaml`
- [ ] Selezione `HP_AE_best` (riga con PR-AUC max, vedi §4.8.8)

### Fase 3.2 — Training finale AE
- [ ] `python -m src.models.train_ae --config params_validated_ae.yaml`
- [ ] Nested validation (3 run con seed diversi) → media ± std
- [ ] `reports/checkpoints/ae_baseline.pth`
- [ ] `reports/tables/ae_final_metrics.csv`

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

## 7. Decisioni aperte (backlog)

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
     (primi 60% → train, successivi 20% → val, ultimi 20% → test).
   - **Random**: mescola i dati con un seed prima dello split.
   Scelta: **temporale**. Confermato in Fase 1: lag-1 AC = 0.99+ su tutte le
   feature → i dati sono una sessione continua. Lo shuffle romperebbe la
   continuità e causerebbe leakage. Split 60/20/20 su KukaNormal, test =
   Normal hold-out + Slow.
4. **Dimensione della finestra W — definizione**: numero di timestep consecutivi
   dati in pasto alla rete ad ogni sample (slide con stride=1). Default `W=16`.
   In Fase 3 si confrontano `W ∈ {8, 16, 32, 64}` su validation set → si
   conferma il valore migliore per trade-off errore di ricostruzione / costo
   computazionale. Vedi §4.1.1.
5. **MAE vs MSE — definizione**:
   - **MSE** (Mean Squared Error): penalizza quadraticamente gli errori grandi.
   - **MAE** (Mean Absolute Error): penalizza linearmente, più robusto a outlier.
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
   StandardScaler li assorbe (z-score alti ma finiti) e la MSE loss penalizza
   naturalmente le finestre con spike saturati. La classe anomala si
   caratterizza da drift lenti, non da spike — il clipping non aiuterebbe e
   potrebbe nascondere pattern utili.
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
   anomala si caratterizza da drift lenti non da spike, StandardScaler + MSE li
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
