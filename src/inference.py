# src/inference.py
import os
import joblib
import numpy as np
import pandas as pd
from evaluate import log_to_price  # must exist in your repo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR  = os.path.join(ROOT, "dataset")
OUT_DIR   = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("MODEL_DIR:", MODEL_DIR)
print("DATA_DIR:", DATA_DIR)
print("OUT_DIR:", OUT_DIR)

# Load test file
test_path = os.path.join(DATA_DIR, "test.csv")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test file not found at {test_path}")
test = pd.read_csv(test_path)

# load baseline and multimodal test log preds
baseline_path = os.path.join(MODEL_DIR, "test_baseline_log.npy")
mm_path       = os.path.join(MODEL_DIR, "test_multimodal_log.npy")

if not os.path.exists(baseline_path):
    raise FileNotFoundError(f"Baseline predictions missing: {baseline_path}")
if not os.path.exists(mm_path):
    raise FileNotFoundError(f"Multimodal predictions missing: {mm_path}")

oof_baseline_test = np.load(baseline_path)  # shape (n_test,)
oof_mm_test       = np.load(mm_path)        # shape (n_test,)

if oof_baseline_test.shape[0] != oof_mm_test.shape[0]:
    raise ValueError("Baseline and multimodal test arrays have different lengths: "
                     f"{oof_baseline_test.shape[0]} vs {oof_mm_test.shape[0]}")

# Combine into stacking features (2 columns)
X_stack_test = np.vstack([oof_baseline_test, oof_mm_test]).T  # shape (n_test, 2)

# If stacking scaler and meta-model exist, apply them
stack_scaler_path = os.path.join(MODEL_DIR, "stack_scaler.joblib")
meta_model_path   = os.path.join(MODEL_DIR, "meta_ridge.joblib")

use_meta = True
if not os.path.exists(meta_model_path):
    print("Meta-model not found at", meta_model_path)
    use_meta = False

if use_meta:
    if os.path.exists(stack_scaler_path):
        stack_scaler = joblib.load(stack_scaler_path)
        try:
            X_stack_scaled = stack_scaler.transform(X_stack_test)
        except Exception as ex:
            print("Warning: stack scaler transform failed:", ex)
            X_stack_scaled = X_stack_test
    else:
        print("Stack scaler not found; will not scale stacking features.")
        X_stack_scaled = X_stack_test

    meta_model = joblib.load(meta_model_path)
    pred_log = meta_model.predict(X_stack_scaled)
    pred_price = log_to_price(pred_log)
    print("Used meta-model for final predictions.")
else:
    # fallback: average price-space predictions from baseline & multimodal
    price_baseline = log_to_price(oof_baseline_test)
    price_mm       = log_to_price(oof_mm_test)
    pred_price = (price_baseline + price_mm) / 2.0
    print("Meta-model missing; used simple average of baseline and multimodal prices.")

# Save final CSV
out_file = os.path.join(OUT_DIR, "test_out_final.csv")
if "sample_id" in test.columns:
    pd.DataFrame({'sample_id': test['sample_id'], 'price': pred_price}).to_csv(out_file, index=False)
else:
    pd.DataFrame({'price': pred_price}).to_csv(out_file, index=False)

print("Wrote final predictions to", out_file)
