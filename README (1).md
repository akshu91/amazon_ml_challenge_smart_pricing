# 🛍 Smart Pricing: A Multimodal Price Prediction Engine

This project implements a sophisticated machine learning pipeline to predict product prices using **multimodal catalog data**, including textual descriptions, numerical features, and image embeddings.  
It uses a **three-stage stacked ensemble** to maximize predictive accuracy.

---

## 📜 Overview

Traditional models rely on a single data type, but Smart Pricing combines **text**, **image**, and **structured numerical data** to create a more robust and accurate price predictor.

This repository includes everything — from a simple text-based baseline to a final, high-performance stacked ensemble model.

---

## ⚙️ Methodology & Pipeline Architecture

### 🧩 Stage 1: Text-Only Baseline Model
- **Data Source:** Uses only the `catalog_content` textual field.
- **Feature Extraction:** TF-IDF (10,000 unigrams + bigrams)
- **Model:** Ridge Regression (α = 1.0)
- **Validation:** 5-Fold Cross-Validation  
✅ **SMAPE = 51.3%**

### 🧩 Stage 2: Multimodal Enhancement
Combines text, image, and numerical features.

#### Feature Engineering
- **Text Features:** TF-IDF → reduced to 128D via Truncated SVD  
- **Image Features:** CNN embeddings → 64D via PCA  
- **Numeric Features:** Pack quantity, text length, word counts, flags  
  (Standardized using `StandardScaler`)

#### Modeling
- **Model:** LightGBM  
- **Validation:** Stratified 5-Fold  
✅ **SMAPE = 49.6%**

### 🧩 Stage 3: Stacked Ensemble
A meta-model (Ridge Regression) learns how to blend Stage 1 & Stage 2 predictions.  
✅ **Final SMAPE = 48.2%**

---

## 📊 Performance Summary

| Model         | Features Used                  | Algorithm          | SMAPE (%) | Improvement |
|----------------|--------------------------------|--------------------|------------|--------------|
| Baseline       | TF-IDF (Text Only)            | Ridge Regression   | 51.3       | —            |
| Multimodal     | TF-IDF + Image + Numeric      | LightGBM           | 49.6       | −1.7%        |
| Final Stack    | Baseline + Multimodal Blend   | Ridge (Meta)       | 48.2       | −1.4%        |

📈 **Total Improvement:** −3.1% over the baseline  

---

## 💡 Key Takeaways
✅ Multimodal integration yields richer predictive signals  
✅ Ensemble stacking reduces overall error  
✅ Dimensionality reduction keeps models efficient  
✅ Simple yet powerful models = best performance

---

## 🚀 How to Run

### Prerequisites
- Python ≥ 3.8
- pip

### Installation
```bash
git clone https://github.com/YOUR-USERNAME/smart-pricing.git
cd smart-pricing
pip install -r requirements.txt
```

### Requirements
```
numpy
pandas
scikit-learn
lightgbm
```

---

## ▶️ Running the Pipeline

1️⃣ **Baseline Model**
```bash
python scripts/1_baseline_model.py
```
Outputs: `test_baseline_log.npy`

2️⃣ **Multimodal Model**
```bash
python scripts/2_multimodal_model.py
```
Outputs: `test_multimodal_log.npy`

3️⃣ **Stacked Ensemble**
```bash
python scripts/3_stacking_model.py
```
Outputs: `test_out_final.csv`

---

## 🧱 Project Structure
```
smart-pricing/
├── data/
│   ├── catalog_data.csv
│   └── image_embeddings.npy
├── notebooks/
├── scripts/
│   ├── 1_baseline_model.py
│   ├── 2_multimodal_model.py
│   └── 3_stacking_model.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📄 License
This project is licensed under the [MIT License](./LICENSE).

---

## 🙌 Acknowledgments
Special thanks to collaborators who contributed data, ideas, or evaluation support.
