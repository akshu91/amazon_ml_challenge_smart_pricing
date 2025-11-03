# src/prepare_tfidf.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    lowercase=True
)

tfidf.fit(train["catalog_content"].astype(str))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))

print("✅ TF-IDF vectorizer saved at:", os.path.join(MODEL_DIR, "tfidf.joblib"))
