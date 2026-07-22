"""
RAGShield — MiniLM Sentence Embedding Extraction
Encodes train and validation sets using all-MiniLM-L6-v2 (384 dimensions).
Uses batched encoding with GPU acceleration where available.

Input:  data/processed/train.parquet
        data/processed/val.parquet
Output: data/features/minilm/X_train.npy
        data/features/minilm/X_val.npy
        data/features/minilm/y_train.npy
        data/features/minilm/y_val.npy

Usage:
    python 2_features/extract_minilm.py
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PROC_DIR = DATA_DIR / "processed"
FEAT_DIR = DATA_DIR / "features" / "minilm"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 512
MAX_SEQ_LEN = 128


def encode(model, texts, desc="encoding"):
    log.info(f"  {desc}: {len(texts):,} examples...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def main():
    log.info(f"Loading MiniLM model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LEN
    log.info(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    log.info("Loading data...")
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    val   = pd.read_parquet(PROC_DIR / "val.parquet")
    log.info(f"  Train: {len(train):,} | Val: {len(val):,}")

    X_train = encode(model, train["text"].tolist(), "Train")
    X_val   = encode(model, val["text"].tolist(),   "Val")
    y_train = train["label"].values.astype(np.int32)
    y_val   = val["label"].values.astype(np.int32)

    log.info(f"  X_train shape: {X_train.shape}")
    log.info(f"  X_val shape:   {X_val.shape}")

    np.save(FEAT_DIR / "X_train.npy", X_train)
    np.save(FEAT_DIR / "X_val.npy",   X_val)
    np.save(FEAT_DIR / "y_train.npy", y_train)
    np.save(FEAT_DIR / "y_val.npy",   y_val)
    log.info(f"Saved to {FEAT_DIR}")

    log.info("Next: python 2_features/extract_mpnet.py")


if __name__ == "__main__":
    main()
