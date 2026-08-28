# Project Plan — AM01: Anomalous Behaviour Detection in Industrial Robot

> Documento di lavoro condiviso. Da aggiornare a ogni sessione per ricordare
> *dove siamo*, *dove stiamo andando* e *perché* abbiamo fatto certe scelte.

---

## 0. TL;DR

- **Obiettivo**: implementare un **Adversarial Autoencoder (AAE)** per anomaly
  detection su time-series di un robot Kuka, e dimostrare se la componente
  avversariale migliora un autoencoder tradizionale.
- **Architettura**: **sequence-aware con finestra di W timestep**, encoder
  1D-Conv + decoder speculare. Scelta motivata dal paper di riferimento
  (Kim S. et al.) e dalla natura dinamica del problema.
- **Approccio**: bottom-up — esplorazione dati → preprocessing → baseline
  AE → AAE → valutazione comparativa rigorosa.
- **Domanda di ricerca**: *La componente avversariale di un AAE migliora le
  performance di anomaly detection rispetto a un autoencoder vanilla su
  time-series di un robot industriale?*

---

## 1. Specifica del progetto

### Cosa chiede la traccia
Implementare un AAE per anomaly detection sul dataset Kuka (time-series di
86 sensori: posizioni giunti, velocità, corrente, potenza, accelerometro,
giroscopio, temperatura). Il modello deve:

1. Essere addestrato su dati **normali** (semi-supervised).
2. Rilevare deviazioni → flag anomalie.
3. Essere **confrontato** con un autoencoder tradizionale.
4. Essere valutato con **metriche appropriate, motivate esplicitamente**
   (la scelta delle metriche è parte della valutazione).

### Riferimento chiave
Kim S. et al., *"Towards a Rigorous Evaluation of Time-series Anomaly
Detection"*, 2022 — citato esplicitamente nella traccia.

### Sfide dichiarate
- Adversarial training
- Time-series (non iid)
- Model comparison

---

## 2. Dataset

### File disponibili (`data/raw/KukaVelocityDataset/`)

| File | Shape | Note |
|---|---|---|
| `KukaNormal.npy` | `(233792, 86)` | Movimenti normali |
| `KukaSlow.npy` | `(41538, 87)` | Movimenti anomali/lenti |
| `KukaColumnNames.npy` | `(87,)` | Nomi delle 87 colonne |

### Anomalie da chiarire (Fase 1)
- `KukaSlow` ha 87 colonne vs 86 di `KukaNormal` → la colonna in più va
  identificata (timestamp? label sintetica?).
- Distribuzione temporale: ordinati? campionamento regolare? ci sono gap?

### Strategia di split
- **KukaNormal NON va nel training per intero.** Va diviso 60/20/20.
- **Training (60% di KukaNormal, ~140k)**: il modello impara la distribuzione
  normale.
- **Validation (20% di KukaNormal, ~47k)**: per early stopping + scelta soglia.
- **Test**: 20% di KukaNormal (~47k) + tutto KukaSlow (~41k) = ~88k etichettati.
- **Modalità**: split temporale (no shuffle) come default — simula il
  deployment reale, in linea con Kim S. et al. Da confermare in Fase 1.

> **Perché solo normali in training**: l'AAE impara la distribuzione del
> "comportamento normale". In inference, input con alto errore di
> ricostruzione = anomalia. Mescolare anomale in training "avvelena" il modello.
>
> **Perché non usare KukaNormal al 100%**: senza val non si fa early stopping
> né si calibra la soglia; senza test_norm non abbiamo veri negativi puliti.

---

## 3. Roadmap a fasi

### Fase 1 — Esplorazione dati (Notebook 01)
- Caricare i 3 `.npy`, shape/dtype/memoria.
- Identificare la 87ª colonna di KukaSlow.
- Statistiche descrittive per feature.
- Distribuzioni Normal vs Slow (istogrammi, boxplot).
- Heatmap correlazioni.
- PCA 2D e t-SNE 2D (separabilità visiva).
- Verifica ordinamento temporale → decide modalità di split.
- **Output**: decisioni documentate (feature da scartare, normalizzazione,
  W, split).

### Fase 2 — Preprocessing (Notebook 02 + `src/data/`)
- Gestione colonna extra in KukaSlow.
- Pulizia: NaN, inf, outlier.
- Split deterministico (temporale o random, da Fase 1).
- Normalizzazione `StandardScaler`, fit solo sul train.
- Salvataggio in `data/processed/` + scaler.
- `src/data/preprocessing.py` e `src/data/dataset.py` modularizzati.
- `tests/test_dataset.py` (almeno shape).

### Fase 3 — Baseline Autoencoder sequence-aware (Notebook 03 + `src/models/autoencoder.py`)
- Encoder 1D-Conv + decoder speculare (vedi §4.1).
- Training su **solo dati normali**, validation per early stopping.
- Confronto `W ∈ {8, 16, 32, 64}` su validation → scelta finale.
- Output: `reports/checkpoints/ae_baseline.pth`.

> Senza baseline non possiamo rispondere alla domanda della traccia.

### Fase 4 — Adversarial Autoencoder (`src/models/adversarial_ae.py`)
- Encoder E(x)→z, Decoder D(z)→x̂, Discriminatore C(z)→{0,1}.
- Prior: `z ~ N(0, I)`.
- Training alternato:
  1. Phase 1: aggiorna C per distinguere `z~prior` da `z=E(x)`.
  2. Phase 2: aggiorna E+D per (a) ricostruire bene, (b) ingannare C.
- Pesi da `params.yaml`: `reconstruction_weight: 1.0`, `adversarial_weight: 0.1`.
- Output: `reports/checkpoints/aae_final.pth`.

### Fase 5 — Valutazione e confronto (`src/models/compare_models.py` + `src/utils/metrics.py`)
- Per ogni modello (AE, AAE):
  - Errore di ricostruzione per sample.
  - Soglia: 99° percentile sul validation (FPR ≈ 1%) + variante F1-max.
  - PR-AUC, ROC-AUC, F1, Precision, Recall, confusion matrix.
- Confronto:
  - Tabella affiancata AE vs AAE.
  - Curve ROC e PR sovrapposte.
  - Test statistico (McNemar) per dire se la differenza è significativa.
- Output: `reports/tables/metrics_comparison.{csv,tex}`,
  `reports/figures/*.png`.

### Fase 6 — Test, pulizia, riproducibilità
- `tests/test_models.py` (forward pass, count param).
- `tests/test_metrics.py` (valori noti).
- `src/main.py` CLI funzionante.
- README aggiornato.

---

## 4. Scelte di design

### 4.1 Architettura: sequence-aware con 1D-Conv

**Ogni sample = finestra di W timestep consecutivi, shape `(W, 86)`.**

| | Point-wise | Sequence-aware |
|---|---|---|
| Input | `(86,)` | `(W, 86)` |
| Architettura | MLP | 1D-Conv |
| Pattern temporali | ❌ | ✅ |
| Rif. Kim S. et al. | non citato | ✅ in linea |
| Anomalie rilevate | statiche | anche dinamiche |

**Scelta**: sequence-aware, perché il problema è dinamico per natura
("moving more slowly", "less precisely") e il paper di riferimento
lavorava su time-series sequence-aware.

**Costo accettabile**: 1D-Conv è veloce, finestre piccole (W ≤ 32) restano
gestibili.

#### 4.1.1 Scelta della finestra W
- `W` è il numero di timestep consecutivi per sample.
- **Default iniziale: W=16.**
- Criteri:
  1. Frequenza di campionamento (da Fase 1): se 100Hz, W=16 = 0.16s, forse
     poco. Se pochi Hz, W=16 = secondi, abbondante.
  2. Pattern da catturare: trend lenti, non spike.
  3. Costo: W grande = più memoria e tempo.
  4. Validazione empirica in Fase 3: confronto `W ∈ {8, 16, 32, 64}`.

#### 4.1.2 Architettura concreta

```
ENCODER                                  DECODER (speculare)
─────────────────────────                ─────────────────────────
Input  (B, 86, W)                        Input  (B, latent_dim)
   ↓                                        ↓
Conv1d(86→128, k=5)                      Linear(latent → 64·W')
   ↓ ReLU                                  ↓ ReLU
MaxPool1d(2)                             Reshape → (B, 64, W')
   ↓                                        ↓
Conv1d(128→64, k=3)                      ConvTranspose1d(64→128, k=4, stride=2)
   ↓ ReLU                                  ↓ ReLU
AdaptiveAvgPool1d(1) → (B, 64, 1)       Conv1d(128→86, k=3)
   ↓                                        ↓
Linear(64 → 16)                          Output (B, 86, W)
   ↓
z  (B, 16)  →  vettore latente
```

- 2 strati Conv perché i dati del robot hanno complessità media (non
  immagini, non audio).
- Filtri 128→64 (piramide decrescente, standard).
- Kernel 5 poi 3 (dispari, decrescente, per pattern locali).
- `AdaptiveAvgPool1d(1)` → indipendente da W (cambio W senza ridisegnare).
- Decoder speculare: fa il percorso inverso, ricostruisce la sequenza.
- Loss: MSE(input, output).

#### 4.1.3 Estensione all'AAE
Encoder e decoder identici al baseline. L'AAE aggiunge solo il
**discriminatore**: una rete piccola (32→16→1) che guarda `z` e dice
"questo `z` è da prior gaussiano o da encoder?". Il training diventa un
gioco a due: il discriminatore cerca di non farsi ingannare, l'encoder
impara a produrre `z` indistinguibili dal prior. Risultato: spazio latente
più regolare.

### 4.2 Normalizzazione: StandardScaler
Media 0, std 1, fit solo sul train. Motivo: feature eterogenee (potenze,
angoli, temperature) hanno range incompatibili; senza normalizzazione il
latent space viene dominato dalle feature a varianza maggiore.

### 4.3 Loss: MSE (poi MAE)
- MSE penalizza errori grandi → enfatizza le anomalie. Standard per AE.
- MAE è più robusto a outlier → da confrontare se i dati sono rumorosi.

### 4.4 Dimensione latente: 16
- Compressione ~5×, sufficiente per estrarre pattern.
- Spazio gestibile dal discriminatore.
- Da confrontare con 8 e 32 in Fase 5 (grid search leggera).

### 4.5 Prior latente: N(0, I)
Default dell'AAE originale (Makhzani 2015).

### 4.6 Metriche
**Primarie** (per il confronto AE vs AAE):
1. **PR-AUC** — gold standard su dati sbilanciati.
2. **ROC-AUC** — per completezza.
3. **F1** a soglia fissata su validation.
4. **Precision/Recall** a soglia operativa.

**Secondarie**:
- Distribuzione errore per classe.
- Confusion matrix.
- Detection latency (rilevante per time-series).

> Niente accuracy come metrica principale: su dati sbilanciati un
> classificatore banale che dice sempre "normale" avrebbe accuracy altissima
> ma F1=0.

### 4.7 Scelta della soglia
- **Primaria**: 99° percentile dell'errore di ricostruzione sul validation
  set (FPR ≈ 1% in condizioni normali).
- **Alternativa**: soglia che massimizza F1 su validation (upper bound
  ottimistico).
- Entrambe riportate → mostra il trade-off operazionale.

---

## 5. Struttura del repository

```
config/         config.yaml, params.yaml
data/           raw/ (originali), processed/ (output preprocessing)
src/
  data/         dataset.py, preprocessing.py
  models/       autoencoder.py, adversarial_ae.py, compare_models.py
  utils/        metrics.py, visualization.py, config.py
  main.py       entry point CLI
notebooks/      01_data_exploration.ipynb, 02_preprocessing.ipynb, 03_model_training.ipynb
tests/          test_dataset.py, test_models.py, test_metrics.py
reports/        checkpoints/, figures/, tables/
docs/           presentation.pdf, project_plan.md (questo file)
```

**Regola**: il codice in `src/` è di produzione, importato da notebook, test
e `main.py`. Il codice nei notebook è esplorativo: se viene usato in
altre celle, va spostato in `src/`.

---

## 6. Stato di avanzamento

### Fase 1 — Esplorazione
- [ ] Caricamento 3 `.npy` e shape/dtype
- [ ] Identificazione 87ª colonna di KukaSlow
- [ ] Statistiche descrittive per feature
- [ ] Confronto distribuzioni Normal vs Slow
- [ ] Heatmap correlazioni
- [ ] PCA / t-SNE 2D
- [ ] Verifica ordinamento temporale → decide split
- [ ] Decisioni documentate

### Fase 2 — Preprocessing
- [ ] Gestione colonna extra
- [ ] Pulizia NaN/inf/outlier
- [ ] Split deterministico
- [ ] Normalizzazione
- [ ] `src/data/preprocessing.py`
- [ ] `src/data/dataset.py`
- [ ] `tests/test_dataset.py`

### Fase 3 — Baseline AE sequence-aware
- [ ] `src/models/autoencoder.py` (Encoder 1D-Conv + Decoder speculare)
- [ ] Training loop in notebook 03
- [ ] Early stopping su validation
- [ ] Checkpoint salvato
- [ ] Forward pass verificato
- [ ] Confronto W ∈ {8, 16, 32, 64} → scelta finale

### Fase 4 — AAE
- [ ] `src/models/adversarial_ae.py` (E + D + C)
- [ ] Training loop alternato
- [ ] Bilanciamento pesi reconstruction/adversarial
- [ ] Checkpoint salvato
- [ ] Sanity check: distribuzione `z` simile a N(0, I)

### Fase 5 — Valutazione
- [ ] `src/utils/metrics.py`
- [ ] Distribuzione errori per classe (AE e AAE)
- [ ] Scelta soglia su validation
- [ ] PR-AUC, ROC-AUC, F1, Precision, Recall
- [ ] Confronto tabellare AE vs AAE
- [ ] Test statistico (McNemar)
- [ ] Visualizzazioni comparative
- [ ] `reports/tables/metrics_comparison.csv` + `.tex`
- [ ] `reports/figures/*.png`

### Fase 6 — Riproducibilità
- [ ] `tests/test_models.py`
- [ ] `tests/test_metrics.py`
- [ ] `src/main.py` CLI
- [ ] README aggiornato
- [ ] Seed fissati ovunque

---

## 7. Decisioni aperte

1. **87ª colonna di KukaSlow**: timestamp, label o errore? → Fase 1.
2. **Feature da scartare**: varianza zero, correlazione > 0.95, dominio. → Fase 1.
3. **Split temporale o random**: default temporale, da confermare in Fase 1.
4. **W**: default 16, validato in Fase 3.
5. **MAE vs MSE**: confronto in Fase 5.
6. **Latent dim**: grid {8, 16, 32} in Fase 5.
7. **Quanti run**: ≥1 AE + ≥1 AAE, idealmente 3+ con seed diversi.
8. **Modelli extra** (Isolation Forest, OC-SVM): facoltativi, se avanza tempo.

---

## 8. Riferimenti

- Traccia: `docs/Projects Topics Presentation.pdf`, pp. 19–20.
- Kim S. et al., *"Towards a Rigorous Evaluation of Time-series Anomaly
  Detection"*, 2022.
- Makhzani, A. et al., *"Adversarial Autoencoders"*, 2015.

---

## 9. Note di sessione

> Sezione libera per annotare *cosa abbiamo fatto*, *cosa non ha funzionato*,
> *idee emerse*. Da consultare a inizio sessione.

- *(da compilare)*
