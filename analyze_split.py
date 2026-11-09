import numpy as np

RAW = r"C:\Users\ungol\OneDrive\Desktop\Ing-Inf-PoliTo\secondo anno\secondo semestre\Machine Learning in Application\project\AM01\AM01-Detection-of-Anomalous-Behaviour-in-Industrial-Robot/data/raw"

cols_all = np.load(f"{RAW}/KukaColumnNames.npy", allow_pickle=True)
print("Colonne totali:", len(cols_all), "| ultima:", cols_all[-1])

def load_split(name, has_label):
    arr = np.load(f"{RAW}/{name}", allow_pickle=False)
    print(f"\n=== {name} ===")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    n_rows, n_cols = arr.shape
    if has_label:
        X = arr[:, :-1]
        y = arr[:, -1]
        print(f"  Label colonna unici: {np.unique(y)}")
    else:
        X = arr
        y = None
    mid = n_rows // 2
    X1, X2 = X[:mid], X[mid:]
    y1, y2 = (y[:mid], y[mid:]) if y is not None else (None, None)
    return X, y, X1, X2, y1, y2, n_rows, n_cols

def stats(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    rng = X.max(axis=0) - X.min(axis=0)
    diff = np.diff(X, axis=0)
    drift = np.abs(diff.mean(axis=0))
    return mu, std, rng, drift

def summarize(name, X1, X2, y1, y2, feat_cols):
    mu1, std1, rng1, drift1 = stats(X1)
    mu2, std2, rng2, drift2 = stats(X2)
    d_mu = np.abs(mu2 - mu1)
    d_std = np.abs(std2 - std1)
    d_rng = np.abs(rng2 - rng1)
    d_drift = np.abs(drift2 - drift1)
    top = np.argsort(d_mu)[::-1][:5]
    print(f"\n--- {name}: label ---")
    if y1 is not None:
        print(f"  prima metà: {int(y1.sum())} / {len(y1)} ({100*y1.mean():.2f}%)")
        print(f"  seconda metà: {int(y2.sum())} / {len(y2)} ({100*y2.mean():.2f}%)")
    else:
        print("  nessuna colonna label presente")
    print(f"\n--- {name}: top 5 feature per shift assoluto della media |mu2-mu1| ---")
    for j in top:
        print(f"  {feat_cols[j]:45s} | mu1={mu1[j]: .4f} mu2={mu2[j]: .4f} |diff|={d_mu[j]: .4f} | std1={std1[j]: .4f} std2={std2[j]: .4f} | drift1={drift1[j]: .4f} drift2={drift2[j]: .4f}")
    print(f"\n--- {name}: sintesi varianza/range/drift (media su feature) ---")
    print(f"  mean(std1)={std1.mean():.4f} mean(std2)={std2.mean():.4f} | diff={abs(std2.mean()-std1.mean()):.6f}")
    print(f"  mean(range1)={rng1.mean():.4f} mean(range2)={rng2.mean():.4f} | diff={abs(rng2.mean()-rng1.mean()):.6f}")
    print(f"  mean(drift1)={drift1.mean():.6f} mean(drift2)={drift2.mean():.6f} | diff={abs(drift2.mean()-drift1.mean()):.6f}")
    print(f"  feature |diff_std|>0.05: {int((d_std>0.05).sum())} | |diff_range|>0.5: {int((d_rng>0.5).sum())} | |diff_drift|>0.01: {int((d_drift>0.01).sum())}")
    return d_mu, d_std, d_rng, d_drift

# KukaSlow: 87 col, ultima è label 'anomaly'
X, y, X1, X2, y1, y2, nr, nc = load_split("KukaSlow.npy", has_label=True)
summarize("KukaSlow", X1, X2, y1, y2, cols_all[:-1])

# KukaNormal: 86 col, nessuna label -> nomi feature senza 'anomaly'
X, y, X1, X2, y1, y2, nr, nc = load_split("KukaNormal.npy", has_label=False)
summarize("KukaNormal", X1, X2, y1, y2, cols_all[:-1])

print("\nFine analisi.")
