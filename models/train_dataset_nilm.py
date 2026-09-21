# =============================================================================
#  train_dataset_nilm.py — NILM Classifier Training — NILM Dataset
#  NILM Dataset: fan (0.301 A) · blender (0.564 A) · hair dryer (0.546 A)
#  fs = 8.192 Hz | W = 256 | S = 128
#
#  Environment: Python 3.10 | scikit-learn 1.3 | NumPy 1.24 | pandas 2.0
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             hamming_loss, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH   = 'Dataset_NILM.csv'
HEADER_OUT = 'nilm_model.h'
W, S       = 256, 128          # window and step (samples)
RANDOM_STATE = 42              # seed for reproducibility

# ── 1. Feature extraction ─────────────────────────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extracts the vector of 9 statistical features from a raw window.

    feat[0] = mean(w)     — raw window mean (ADC bias estimate)
    feat[1] = std(a)      — standard deviation of the AC signal
    feat[2] = rms(a)      — RMS value of the AC signal (≡ std when mean(a)=0)
    feat[3] = min(a)      — minimum of the AC signal
    feat[4] = max(a)      — maximum of the AC signal
    feat[5] = pp(a)       — peak-to-peak amplitude
    feat[6] = energy(a)   — window energy (Σ a²)
    feat[7] = p05(a)      — 5th percentile
    feat[8] = p95(a)      — 95th percentile

    Note: feat[0] is computed from the raw signal w[n];
    feat[1..8] are computed from the AC signal a[n] = w[n] - mean(w).
    """
    w    = window.astype(np.float32)
    mean = w.mean()
    a    = w - mean           # DC component removal
    std  = a.std()
    rms  = np.sqrt((a ** 2).mean())
    return np.array([
        mean,
        std,
        rms,
        a.min(),
        a.max(),
        a.max() - a.min(),
        float(np.sum(a ** 2)),
        np.percentile(a, 5),
        np.percentile(a, 95),
    ], dtype=np.float32)

# ── 2. Data loading and windowing ─────────────────────────────────────────────
print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")

labels_cols = ['fan', 'blender', 'hair dryer']
combos = df[labels_cols].drop_duplicates().sort_values(labels_cols)
print(f"\nStates found: {len(combos)}")

X_list, Y_list = [], []
for _, row in combos.iterrows():
    mask = (
        (df['fan']    == row['fan'])   &
        (df['blender']== row['blender'])&
        (df['hair dryer']       == row['hair dryer'])
    )
    adc = df.loc[mask, 'adc'].values
    lbl = row[labels_cols].values.astype(np.int8)
    n_windows = (len(adc) - W) // S + 1
    for k in range(n_windows):
        X_list.append(extract_features(adc[k*S : k*S + W]))
        Y_list.append(lbl)

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.int8)
print(f"\nTotal windows: {len(X):,}  |  Features: {X.shape[1]}  |  Outputs: {Y.shape[1]}")

# ── 3. Temporal 80/20 split by state ──────────────────────────────────────────
print("\nTemporal 80/20 split by state:")
Xtr_l, Xte_l, Ytr_l, Yte_l = [], [], [], []
for lbl in np.unique(Y, axis=0):
    idx  = np.where((Y == lbl).all(axis=1))[0]
    sp   = int(len(idx) * 0.8)
    Xtr_l.append(X[idx[:sp]]);  Xte_l.append(X[idx[sp:]])
    Ytr_l.append(Y[idx[:sp]]);  Yte_l.append(Y[idx[sp:]])
    v, l, s = lbl
    nome = '+'.join([n for n, b in zip(['V','L','S'], lbl) if b]) or 'Idle'
    print(f"  {nome:8s}: {len(idx):5,} windows  →  train {sp:4,}  test {len(idx)-sp:4,}")

X_train = np.vstack(Xtr_l);  Y_train = np.vstack(Ytr_l)
X_test  = np.vstack(Xte_l);  Y_test  = np.vstack(Yte_l)

# ── 4. Standardization ────────────────────────────────────────────────────────
scaler  = StandardScaler().fit(X_train)
Xtr_n   = scaler.transform(X_train)
Xte_n   = scaler.transform(X_test)

# ── 5. Classifier training ────────────────────────────────────────────────────
print("\nTraining classifiers ...")

mlp = MultiOutputClassifier(
    MLPClassifier(
        hidden_layer_sizes=(32, 16, 8),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=32,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    ), n_jobs=-1
).fit(Xtr_n, Y_train)

rf = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
).fit(Xtr_n, Y_train)

dt = MultiOutputClassifier(
    DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=2,
        criterion='gini',
        random_state=RANDOM_STATE,
    )
).fit(Xtr_n, Y_train)

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def avaliar(nome, model, Xte, Yte):
    pred = np.column_stack([e.predict(Xte) for e in model.estimators_])
    acc  = accuracy_score(Yte, pred)
    f1   = f1_score(Yte, pred, average='macro', zero_division=0)
    hl   = hamming_loss(Yte, pred)
    print(f"  {nome:25s}  Accuracy={acc*100:.3f}%  F1={f1:.4f}  Hamming={hl:.6f}")
    return pred

print("\nResults on the test set (20%):")
pred_mlp = avaliar("MLP  9→32→16→8→3", mlp, Xte_n, Y_test)
pred_rf  = avaliar("Random Forest 100 trees", rf,  Xte_n, Y_test)
pred_dt  = avaliar("Decision Tree depth 10", dt, Xte_n, Y_test)

# ── 7. C header generation (selected MLP) ─────────────────────────────────────
print(f"\nGenerating {HEADER_OUT} ...")

def fmt_array(arr: np.ndarray, name: str) -> str:
    """Serializes a NumPy array as static const float[] in C, transposing 2-D matrices."""
    mat  = arr.T if arr.ndim == 2 else arr   # (n_out × n_in) so _fc can index W[j*n_in+i]
    vals = ', '.join(f'{v:.8f}f' for v in mat.flatten())
    sh   = 'x'.join(str(s) for s in mat.shape)
    return f"// {sh}\nstatic const float {name}[] = {{{vals}}};"

lines = [
    "// ============================================================",
    "//  nilm_model.h  —  NILM Dataset  —  MLP 9→32→16→8→3",
    f"//  Python {'.'.join(str(x) for x in __import__('sys').version_info[:3])}",
    f"//  scikit-learn {__import__('sklearn').__version__}",
    "//  Burden: 330 Ω | fs=8192 Hz | W=256 | S=128",
    "//  feat[0]=mean(w) (ADC bias) | feat[1..8] from AC signal",
    "//  Weights: transposed (n_out × n_in) → W[j*n_in+i]",
    "// ============================================================",
    "#pragma once",
    "#include <math.h>",
    "#include <string.h>",
    "",
    "#define NILM_WINDOW_SIZE   256",
    "#define NILM_STEP_SIZE     128",
    "#define NILM_N_FEATURES      9",
    "#define NILM_N_OUTPUTS       3",
    "",
]

# Scaler
mu_str  = ', '.join(f'{v:.8f}f' for v in scaler.mean_)
std_str = ', '.join(f'{v:.8f}f' for v in scaler.scale_)
lines += [
    f"static const float SCALER_MEAN[9] = {{{mu_str}}};",
    f"static const float SCALER_STD [9] = {{{std_str}}};",
    "",
]

# Weights per estimator (Binary Relevance: 3 independent estimators)
label_names = ['fan', 'blender', 'hair dryer']
for li, (est, lname) in enumerate(zip(mlp.estimators_, label_names)):
    lines.append(f"// ── estimator {li}: {lname} ─────────────────────────")
    for ki, (coef, bias) in enumerate(zip(est.coefs_, est.intercepts_)):
        lines.append(fmt_array(coef, f"W{li}L{ki}"))
        lines.append(fmt_array(bias, f"b{li}L{ki}"))
    lines.append("")

# Helper functions and inference
sz = [9] + [c.shape[1] for c in mlp.estimators_[0].coefs_]
lines.append(r"""static inline float _relu_s3(float x) { return x > 0.0f ? x : 0.0f; }
static inline float _sig_s3 (float x) { return 1.0f / (1.0f + expf(-x)); }

static inline void _fc_s3(const float *in, int n_in,
                           const float *W, const float *b,
                           float *out, int n_out, int relu) {
    for (int j = 0; j < n_out; j++) {
        float s = b[j];
        for (int i = 0; i < n_in; i++) s += in[i] * W[j * n_in + i];
        out[j] = relu ? _relu_s3(s) : _sig_s3(s);
    }
}

static inline void nilm_feat_s3(const float *w, float *f) {
    float m = 0;
    for (int i = 0; i < 256; i++) m += w[i];
    m /= 256.0f;
    float a[256];
    for (int i = 0; i < 256; i++) a[i] = w[i] - m;
    float var = 0, mn = a[0], mx = a[0], en = 0;
    for (int i = 0; i < 256; i++) {
        var += a[i] * a[i];  en += a[i] * a[i];
        if (a[i] < mn) mn = a[i];
        if (a[i] > mx) mx = a[i];
    }
    float sd = sqrtf(var / 256.0f);
    float s2[256];
    memcpy(s2, a, sizeof(float) * 256);
    for (int i = 1; i < 256; i++) {
        float k = s2[i]; int j = i - 1;
        while (j >= 0 && s2[j] > k) { s2[j + 1] = s2[j]; j--; }
        s2[j + 1] = k;
    }
    f[0] = m;         /* mean(w) — ADC bias */
    f[1] = sd;        f[2] = sd;
    f[3] = mn;        f[4] = mx;
    f[5] = mx - mn;   f[6] = en;
    f[7] = s2[12];    f[8] = s2[243];
}

static inline void nilm_run_s3(const float *w,
                                float *probs, uint8_t *labels) {
    float feat[9];
    nilm_feat_s3(w, feat);
    for (int i = 0; i < 9; i++)
        feat[i] = (feat[i] - SCALER_MEAN[i]) / SCALER_STD[i];""")

for li, lname in enumerate(label_names):
    s = sz
    lines.append(f"""    /* {lname} */
    {{ float h0[{s[1]}], h1[{s[2]}], h2[{s[3]}], out;
       _fc_s3(feat, {s[0]}, W{li}L0, b{li}L0, h0, {s[1]}, 1);
       _fc_s3(h0,   {s[1]}, W{li}L1, b{li}L1, h1, {s[2]}, 1);
       _fc_s3(h1,   {s[2]}, W{li}L2, b{li}L2, h2, {s[3]}, 1);
       _fc_s3(h2,   {s[3]}, W{li}L3, b{li}L3, &out,  1, 0);
       probs[{li}] = out;  labels[{li}] = (out >= 0.5f) ? 1 : 0; }}""")

lines.append("}")

header = '\n'.join(lines)
with open(HEADER_OUT, 'w') as fh:
    fh.write(header)

print(f"  {HEADER_OUT} saved ({len(header.encode())/1024:.1f} KB)")
print("\nCompleted.")
