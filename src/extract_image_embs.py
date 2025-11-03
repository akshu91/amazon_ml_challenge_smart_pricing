# src/extract_image_embs.py
import os, sys, time, io, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

# Try to use organizers' downloader if present
try:
    from src.utils import download_images  # adjust import if needed
    have_downloader = True
except Exception:
    have_downloader = False

import torch
from torchvision import transforms, models

DATA_DIR = "../dataset"
OUT_DIR = "../models"
os.makedirs(OUT_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove fc
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def fetch_image_tensor(url):
    import requests
    if not isinstance(url, str) or url.strip()=="":
        return None
    try:
        resp = requests.get(url, timeout=6)
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        return transform(img)
    except Exception:
        return None

def compute_embeddings(df, fname):
    embs = []
    BATCH = 32
    tensors = []
    indices = []
    for i, url in enumerate(tqdm(df['image_link'].fillna("").values, total=len(df))):
        t = fetch_image_tensor(url)
        if t is None:
            embs.append(np.zeros(2048, dtype=np.float32))
        else:
            tensors.append(t.unsqueeze(0))
            indices.append(i)
        # process batch
        if len(tensors) >= BATCH:
            batch = torch.cat(tensors, dim=0).to(device)
            with torch.no_grad():
                feat = model(batch).squeeze(-1).squeeze(-1).cpu().numpy()
            # assign
            for j, idx in enumerate(indices):
                embs.append(feat[j])
            tensors, indices = [], []
    # flush remaining tensors
    if tensors:
        batch = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            feat = model(batch).squeeze(-1).squeeze(-1).cpu().numpy()
        for j, idx in enumerate(indices):
            embs.append(feat[j])
    embs = np.vstack(embs)
    np.save(os.path.join(OUT_DIR, fname), embs)
    print("Saved", fname, embs.shape)

compute_embeddings(train_df, "img_embs_train.npy")
compute_embeddings(test_df,  "img_embs_test.npy")
