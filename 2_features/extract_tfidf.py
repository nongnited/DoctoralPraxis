"""
RAGShield — TF-IDF Feature Extraction
Fits a TF-IDF vectorizer on the training set and transforms both
train and validation sets. Saves sparse matrices and the fitted vectorizer.

Configuration:
  max_features=50,000   (50K most frequent n-grams)
  ngram_range=(1,2)     (unigrams and bigrams)
  sublinear_tf=True     (log-scaled term frequencies)
  min_df=2              (ignore terms appearing in <2 documents)

Input:  data/processed/train.parquet
        data/processed/val.parquet
Output: data/features/tfidf/X_train.npz
        data/features/tfidf/X_val.npz
        data/features/tfidf/y_train.npy
        data/features/tfidf/y_val.npy
        models/tfidf_vectorizer.joblib

Usage:
    python 2_features/extract_tfidf.py
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR    = Path(os.getenv("DATA_DIR",    "./data"))
MODELS_DIR  = Path(os.getenv("MODELS_DIR",  "./models"))
PROC_DIR    = DATA_DIR / "processed"
FEAT_DIR    = DATA_DIR / "features" / "tfidf"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MAX_FEATURES = 50_000
NGRAM_RANGE  = (1, 2)


def main():
    # Load data
    log.info("Loading train/val data...")
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    val   = pd.read_parquet(PROC_DIR / "val.parquet")
    log.info(f"  Train: {len(train):,} | Val: {len(val):,}")

    # Fit vectorizer on training set only
    log.info(f"Fitting TF-IDF vectorizer (max_features={MAX_FEATURES:,})...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z_][a-zA-Z0-9_]+\b",
    )

    X_train = vectorizer.fit_transform(train["text"])
    X_val   = vectorizer.transform(val["text"])
    y_train = train["label"].values.astype(np.int32)
    y_val   = val["label"].values.astype(np.int32)

    log.info(f"  X_train shape: {X_train.shape}")
    log.info(f"  X_val shape:   {X_val.shape}")
    log.info(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Save features
    sp.save_npz(FEAT_DIR / "X_train.npz", X_train)
    sp.save_npz(FEAT_DIR / "X_val.npz",   X_val)
    np.save(FEAT_DIR / "y_train.npy", y_train)
    np.save(FEAT_DIR / "y_val.npy",   y_val)
    log.info(f"Features saved to {FEAT_DIR}")

    # Save vectorizer
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vec_path)
    log.info(f"Vectorizer saved to {vec_path}")

    log.info("Next: python 2_features/extract_minilm.py")


if __name__ == "__main__":
    main()
