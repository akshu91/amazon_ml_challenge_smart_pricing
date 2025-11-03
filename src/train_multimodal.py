# src/train_multimodal.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# local helpers (must exist in your repo)
from features import add_basic_features, transform_tfidf
from evaluate import log_to_price

# Helpers
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "dataset")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Config
use_sentence_transformers = False
random_state = 42
n_splits = 5
svd_text_dim = 128
pca_img_dim = 64

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Basic features
def extract_ipq(text):
    import re
    if pd.isna(text):
        return 1
    s = str(text).lower()
    m = re.search(r'pack of (\d+)', s)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*pack\b', s)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*[x×]\s*\d*', s)
    if m: return int(m.group(1))
    nums = [int(x) for x in re.findall(r'\b(\d{1,4})\b', s)]
    nums = [n for n in nums if n>1 and n<1000]
    return max(nums) if nums else 1

for df in (train, test):
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['chars'] = df['catalog_content'].apply(len)
    df['words'] = df['catalog_content'].apply(lambda x: len(x.split()))
    df['has_image'] = df['image_link'].notnull().astype(int) if 'image_link' in df.columns else 0
    df['has_currency_symbol'] = df['catalog_content'].str.contains(r'₹|\brs\b|\bINR\b', regex=True).fillna(False).astype(int)

# Text features (SBERT optional)
if use_sentence_transformers:
    try:
        from sentence_transformers import SentenceTransformer
        s_model = SentenceTransformer('all-mpnet-base-v2')
        text_embed_train = s_model.encode(train['catalog_content'].tolist(), show_progress_bar=True)
        text_embed_test  = s_model.encode(test['catalog_content'].tolist(), show_progress_bar=True)
        joblib.dump(s_model, os.path.join(MODEL_DIR, 'sbert.joblib'))
    except Exception as e:
        print("SBERT not available, falling back:", e)
        use_sentence_transformers = False

if not use_sentence_transformers:
    tfidf_path = os.path.join(MODEL_DIR, "tfidf_multimodal.joblib")
    if os.path.exists(tfidf_path):
        tfidf = joblib.load(tfidf_path)
        print("Loaded TF-IDF from", tfidf_path)
    else:
        print("Fitting TF-IDF for multimodal...")
        tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=15000, min_df=3)
        tfidf.fit(pd.concat([train['catalog_content'], test['catalog_content']]))
        joblib.dump(tfidf, tfidf_path)

    X_text_train_sparse = tfidf.transform(train['catalog_content'])
    X_text_test_sparse  = tfidf.transform(test['catalog_content'])
    svd = TruncatedSVD(n_components=svd_text_dim, random_state=random_state)
    X_text_train = svd.fit_transform(X_text_train_sparse)
    X_text_test  = svd.transform(X_text_test_sparse)
    joblib.dump(svd, os.path.join(MODEL_DIR, "svd_text.joblib"))

# Image embeddings
img_train_path = os.path.join(MODEL_DIR, "img_embs_train.npy")
img_test_path  = os.path.join(MODEL_DIR, "img_embs_test.npy")
if os.path.exists(img_train_path) and os.path.exists(img_test_path):
    img_train = np.load(img_train_path)
    img_test  = np.load(img_test_path)
    pca = PCA(n_components=pca_img_dim, random_state=random_state)
    img_train_p = pca.fit_transform(img_train)
    img_test_p  = pca.transform(img_test)
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca_img.joblib"))
else:
    img_train_p = np.zeros((len(train), pca_img_dim), dtype=np.float32)
    img_test_p  = np.zeros((len(test), pca_img_dim), dtype=np.float32)

# Numeric features & scaler
num_cols = ['ipq', 'chars', 'words', 'has_image', 'has_currency_symbol']
X_num_train = train[num_cols].fillna(0).values
X_num_test  = test[num_cols].fillna(0).values

scaler_path = os.path.join(MODEL_DIR, "scaler_multimodal.joblib")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = StandardScaler()
    scaler.fit(X_num_train)
    joblib.dump(scaler, scaler_path)

X_num_train_scaled = scaler.transform(X_num_train)
X_num_test_scaled  = scaler.transform(X_num_test)

# Final features
if use_sentence_transformers:
    X_train = np.hstack([text_embed_train, img_train_p, X_num_train_scaled])
    X_test  = np.hstack([text_embed_test,  img_test_p,  X_num_test_scaled])
else:
    X_train = np.hstack([X_text_train, img_train_p, X_num_train_scaled])
    X_test  = np.hstack([X_text_test,  img_test_p,  X_num_test_scaled])

print("Final feature shapes:", X_train.shape, X_test.shape)

# Target and CV
y = train['price'].values
y_log = np.log1p(y)

try:
    bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
except Exception:
    bins = pd.cut(y, bins=10, labels=False)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

oof = np.zeros(len(y))
test_preds = np.zeros(X_test.shape[0])

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'num_leaves': 128,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'n_jobs': -1,
    'seed': random_state
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, bins)):
    print(f"Fold {fold+1}/{n_splits}")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )

    oof[val_idx] = bst.predict(X_val, num_iteration=bst.best_iteration)
    test_preds += bst.predict(X_test, num_iteration=bst.best_iteration) / n_splits

    mdl_path = os.path.join(MODEL_DIR, f"lgbm_multimodal_fold{fold}.txt")
    bst.save_model(mdl_path)
    print("Saved", mdl_path)

# Save CV artifacts (log-space)
np.save(os.path.join(MODEL_DIR, "oof_multimodal_log.npy"), oof)
np.save(os.path.join(MODEL_DIR, "test_multimodal_log.npy"), test_preds)
joblib.dump({"features":"text+img+pct"}, os.path.join(MODEL_DIR, "multimodal_meta.joblib"))

# Eval CV
oof_price = log_to_price(oof)
cv_smape = smape(y, oof_price)
print("Multimodal CV SMAPE: {:.4f}%".format(cv_smape))

# Create submission (price-space)
test_price = log_to_price(test_preds)
out = pd.DataFrame({'sample_id': test['sample_id'], 'price': test_price})
out.to_csv(os.path.join(OUT_DIR, "test_out_multimodal.csv"), index=False)
print("Wrote", os.path.join(OUT_DIR, "test_out_multimodal.csv"))
