# src/features.py
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def extract_ipq(text):
    """Crude extraction of item pack quantity from catalog_content.
       Returns integer pack size (>=1)."""
    if pd.isna(text): return 1
    s = str(text).lower()
    # common patterns
    m = re.search(r'pack of (\d+)', s)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*pack\b', s)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*[x×]\s*\d*', s)
    if m: return int(m.group(1))
    # fallback: use numbers >1
    nums = [int(x) for x in re.findall(r'\b(\d{1,4})\b', s)]
    nums = [n for n in nums if n>1 and n<1000]
    if nums:
        return max(nums)
    return 1

def add_basic_features(df):
    df = df.copy()
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['chars'] = df['catalog_content'].apply(len)
    df['words'] = df['catalog_content'].apply(lambda x: len(x.split()))
    df['has_image'] = df['image_link'].notnull().astype(int)
    df['has_currency_symbol'] = df['catalog_content'].str.contains(r'₹|\brs\b|\bINR\b', regex=True).fillna(False).astype(int)
    return df

def fit_tfidf(texts, max_features=25000):
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=max_features, min_df=3, max_df=0.9)
    X = tfidf.fit_transform(texts)
    return tfidf, X

def transform_tfidf(tfidf, texts):
    return tfidf.transform(texts)

def fit_scaler(X_numeric):
    scaler = StandardScaler()
    scaler.fit(X_numeric)
    return scaler
