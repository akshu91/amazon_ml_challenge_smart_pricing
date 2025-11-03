# src/stack_models.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import exp
from sklearn.preprocessing import StandardScaler

# -------------------
# Helpers
# -------------------
def smape(y_true, y_pred):
    denom = (abs(y_true) + abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return ((abs(y_true - y_pred) / denom).mean()) * 100.0

def log_to_price(x):
    return np.expm1(x)

# -------------------
# Paths
# -------------------
# --------------------------
# Ensure paths are relative to project root, not where the script runs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------
# Load OOF preds
# -------------------
# baseline OOF must be saved by your baseline script as models/oof_baseline_log.npy
# multimodal OOF saved earlier as models/oof_multimodal_log.npy
oof_files = {
    "baseline": os.path.join(MODEL_DIR, "oof_baseline_log.npy"),
    "multimodal": os.path.join(MODEL_DIR, "oof_multimodal_log.npy")
}

missing = [k for k,v in oof_files.items() if not os.path.exists(v)]
if missing:
    print("Missing OOF files for:", missing)
    print("Make sure train_baseline.py saved models/oof_baseline_log.npy and train_multimodal.py saved models/oof_multimodal_log.npy")
    raise SystemExit(1)

oof_baseline = np.load(oof_files["baseline"])   # log-space preds shape = (n_train,)
oof_mm       = np.load(oof_files["multimodal"])
# Basic check: shapes equal
if oof_baseline.shape != oof_mm.shape:
    raise ValueError("OOF shapes mismatch:", oof_baseline.shape, oof_mm.shape)

# load y_log (if saved), otherwise reconstruct from dataset
y_log_path = os.path.join(MODEL_DIR, "y_log.npy")
if os.path.exists(y_log_path):
    y_log = np.load(y_log_path)
else:
    import pandas as pd
    train = pd.read_csv("./dataset/train.csv")
    y_log = np.log1p(train['price'].values)

# Stack features
X_stack = np.vstack([oof_baseline, oof_mm]).T
y = y_log  # log target

# scale
scaler = StandardScaler()
X_stack_scaled = scaler.fit_transform(X_stack)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_stack_scaled = imputer.fit_transform(X_stack_scaled)
joblib.dump(scaler, os.path.join(MODEL_DIR, "stack_scaler.joblib"))

# CV for stacked model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_meta = np.zeros_like(y)
for train_idx, val_idx in kf.split(X_stack_scaled):
    clf = Ridge(alpha=1.0)
    print("Any NaNs in X_stack_scaled?", np.isnan(X_stack_scaled).any())
    print("Shape of X_stack_scaled:", X_stack_scaled.shape)
    print("Any NaNs in y?", np.isnan(y).any())
    print("NaN count per column:", np.isnan(X_stack_scaled).sum(axis=0))

    clf.fit(X_stack_scaled[train_idx], y[train_idx])
    oof_meta[val_idx] = clf.predict(X_stack_scaled[val_idx])

# Evaluate - SMAPE on original price space
y_true = np.expm1(y)
y_pred = np.expm1(oof_meta)
print("Stacked OOF SMAPE: {:.4f}%".format(smape(y_true, y_pred)))

# Train final meta model on full data and save
final_meta = Ridge(alpha=1.0)
final_meta.fit(X_stack_scaled, y)
joblib.dump(final_meta, os.path.join(MODEL_DIR, "meta_ridge.joblib"))
print("Saved meta model:", os.path.join(MODEL_DIR, "meta_ridge.joblib"))
