# src/prepare_svd.py
import os
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load training data
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
texts = train["catalog_content"].astype(str).fillna("")

# Load TF-IDF (create if missing)
tfidf_path = os.path.join(MODEL_DIR, "tfidf.joblib")
if os.path.exists(tfidf_path):
    print("✅ Found existing TF-IDF model.")
    tfidf = joblib.load(tfidf_path)
else:
    print("⚠️ TF-IDF not found. Creating one...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", lowercase=True)
    tfidf.fit(texts)
    joblib.dump(tfidf, tfidf_path)
    print("✅ TF-IDF vectorizer saved.")

# Transform text and fit SVD
X_tfidf = tfidf.transform(texts)
svd = TruncatedSVD(n_components=100, random_state=42)
svd.fit(X_tfidf)
joblib.dump(svd, os.path.join(MODEL_DIR, "svd_text.joblib"))

print("✅ SVD model saved at:", os.path.join(MODEL_DIR, "svd_text.joblib"))
