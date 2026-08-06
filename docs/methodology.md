# Methodología del Progetto

## Approccio Metodologico

### 1. Data Loading
- Caricamento dati in formato NumPy (.npy)
- Verifica integrità e forma dei dati
- Separazione tra dati di training e test

### 2. Preprocessing
- Normalizzazione standard dei dati
- Gestione class imbalance (dati normali vs anomali)
- Feature engineering se necessario

### 3. Model Architecture
- **Autoencoder Base**: Reconruttare l'input con compressione
  - Encoder: input → hidden layers → latent space
  - Decoder: latent space → hidden layers → output

- **Adversarial Autoencoder**: Aggiungere componente adversarial
  - Discriminatore: distinguere latent reale da generato
  - Loss combinata: reconstruction + adversarial

### 4. Training Strategy
- Pre-training dell'autoencoder
- Training adversarial con gradienti alternati
- Early stopping per evitare overfitting

### 5. Anomaly Detection
- Calcolo errore di ricostruzione
- Definizione soglia per classificazione
- Valutazione metriche: Precision, Recall, F1, AUC

## Metriche di Valutazione
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

## Riferimenti
- Kim S. et al., "Towards a Rigorous Evaluation of Time-series Anomaly Detection", 2022