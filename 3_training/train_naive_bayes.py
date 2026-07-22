"""
RAGShield — Train Naïve Bayes Classifier
Evaluates MultinomialNB (TF-IDF) and GaussianNB (dense embeddings).
Naïve Bayes does not support class_weight, so only none and smote are used.

Configurations (6 total):
  tfidf_none, tfidf_smote*  (* skipped — SMOTE on 50K dims too expensive)
  minilm_none, minilm_smote
  mpnet_none, mpnet_smote

Usage:
    python 3_training/train_naive_bayes.py
"""

import os
import time
import logging
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MaxAbsScaler
from dotenv import load_dotenv

from train_utils import (
    load_features, apply_balancing, compute_metrics,
    save_model, log_results
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME  = "naive_bayes"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

CONFIGS = [
    # (feature, balancing, sparse)
    ("tfidf",  "none",     True),
    ("tfidf",  "balanced", True),   # uses balanced prior probability
    ("minilm", "none",     False),
    ("minilm", "smote",    False),
    ("mpnet",  "none",     False),
    ("mpnet",  "smote",    False),
]


def train_config(feature: str, balancing: str, sparse: bool):
    run_name = f"{MODEL_NAME}_{feature}_{balancing}"
    log.info(f"\n{'='*60}\n  TRAINING: {run_name}\n{'='*60}")

    X_train, X_val, y_train, y_val = load_features(feature, sparse=sparse)

    # Apply balancing
    X_tr, y_tr, _ = apply_balancing(X_train, y_train, balancing, RANDOM_SEED)

    # Select model variant
    if sparse:
        # MultinomialNB requires non-negative features
        # MaxAbsScaler preserves sparsity and non-negativity for TF-IDF
        scaler = MaxAbsScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        model = MultinomialNB()
    else:
        X_val_s = X_val
        if balancing == "smote":
            # SMOTE may produce negative values — clip for MultinomialNB
            X_tr    = np.clip(X_tr, 0, None).astype(np.float32)
            X_val_s = np.clip(X_val_s, 0, None).astype(np.float32)
        model = GaussianNB()

    # Train
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0

    # Evaluate
    y_pred = model.predict(X_val_s)
    try:
        y_prob = model.predict_proba(X_val_s)[:, 1]
    except Exception:
        y_prob = None

    metrics = compute_metrics(y_val, y_pred, y_prob)
    metrics["train_time_s"] = round(elapsed, 2)
    metrics["feature"]      = feature
    metrics["balancing"]    = balancing
    metrics["model"]        = MODEL_NAME

    log_results(MODEL_NAME, feature, balancing, metrics, elapsed)
    save_model(model, run_name, metrics=metrics, config={
        "feature": feature, "balancing": balancing, "model_type": type(model).__name__
    })

    return metrics


def main():
    results = {}
    for feature, balancing, sparse in CONFIGS:
        key = f"{feature}_{balancing}"
        try:
            results[key] = train_config(feature, balancing, sparse)
        except FileNotFoundError as e:
            log.warning(f"Skipping {key}: {e}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("  NAÏVE BAYES — SUMMARY (sorted by MCC)")
    log.info("=" * 60)
    log.info(f"  {'Config':<25} {'MCC':>8} {'F1':>8} {'FAR':>8}")
    for key, m in sorted(results.items(), key=lambda x: -x[1]["mcc"]):
        log.info(f"  {key:<25} {m['mcc']:>8.4f} {m['f1']:>8.4f} {m['false_alarm_rate']:>8.4f}")


if __name__ == "__main__":
    main()
