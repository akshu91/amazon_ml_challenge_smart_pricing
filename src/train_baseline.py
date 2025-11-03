# train_baseline.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Paths (use project-root relative models/ and outputs/)
ROOT = os.path.abspath(os.getcwd())
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "outputs")
DATA_DIR = os.path.join(ROOT, "dataset")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("MODEL_DIR:", MODEL_DIR)
print("OUT_DIR:", OUT_DIR)

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

train["catalog_content"] = train["catalog_content"].fillna("").astype(str)
test["catalog_content"] = test["catalog_content"].fillna("").astype(str)

# We'll train on log1p(price) to stabilise regression
y = train["price"].values
y_log = np.log1p(y)

# Fit one TF-IDF on training text (consistent across folds) so we can reuse it at inference
print("Fitting TF-IDF on training text...")
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), min_df=2)
tfidf.fit(train["catalog_content"])
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_baseline.joblib"))
X_train_tfidf_full = tfidf.transform(train["catalog_content"])
X_test_tfidf = tfidf.transform(test["catalog_content"])

# 5-fold CV training of Ridge on TF-IDF features
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_log = np.zeros(len(train))
fold_models = []

print("Starting baseline CV training...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    print(f"Fold {fold+1}/5")
    X_tr = X_train_tfidf_full[tr_idx]
    X_val = X_train_tfidf_full[val_idx]
    y_tr = y_log[tr_idx]
    y_val = y_log[val_idx]

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_tr, y_tr)
    preds_val_log = model.predict(X_val)
    oof_log[val_idx] = preds_val_log

    mae = mean_absolute_error(np.expm1(y_val), np.expm1(preds_val_log))  # MAE in price-space
    print(f"  Fold {fold+1} MAE (price-space): {mae:.4f}")

    # save per-fold model
    model_path = os.path.join(MODEL_DIR, f"ridge_baseline_fold{fold}.joblib")
    joblib.dump(model, model_path)
    fold_models.append(model_path)

# Save OOF (log-space) and oof in price-space
np.save(os.path.join(MODEL_DIR, "oof_baseline_log.npy"), oof_log)
np.save(os.path.join(MODEL_DIR, "oof_baseline_price.npy"), np.expm1(oof_log))

# Predict on test using ensemble of fold models (average in log-space)
print("Predicting on test using saved fold models...")
test_preds_log = np.zeros(len(test))
for fold_idx, model_path in enumerate(sorted(fold_models)):
    mdl = joblib.load(model_path)
    test_preds_log += mdl.predict(X_test_tfidf) / len(fold_models)

# Save test baseline predictions (log-space) — inference expects this filename
np.save(os.path.join(MODEL_DIR, "test_baseline_log.npy"), test_preds_log)
# Also save a CSV of the final predictions (price-space)
test_price = np.expm1(test_preds_log)
submission = pd.DataFrame({
    "sample_id": test.get("sample_id", test.index),
    "price": test_price
})
submission.to_csv(os.path.join(OUT_DIR, "test_baseline_predictions.csv"), index=False)
print("Saved baseline test predictions to:", os.path.join(MODEL_DIR, "test_baseline_log.npy"))
print("Also wrote CSV to:", os.path.join(OUT_DIR, "test_baseline_predictions.csv"))
