"""
RAGShield — Train LightGBM Classifier
500 trees, 127 leaves (leaf-wise growth), auto GPU detection.

Configurations (8 total):
  tfidf_none, tfidf_balanced
  minilm_none, minilm_balanced, minilm_smote
  mpnet_none, mpnet_balanced, mpnet_smote

Usage:
    python 3_training/train_lightgbm.py
"""

import os
import time
import logging

import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
from dotenv import load_dotenv

from train_utils import (
    load_features, apply_balancing, compute_metrics,
    save_model, log_results
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME  = "lightgbm"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

CONFIGS = [
    ("tfidf",  "none",     True),
    ("tfidf",  "balanced", True),
    ("minilm", "none",     False),
    ("minilm", "balanced", False),
    ("minilm", "smote",    False),
    ("mpnet",  "none",     False),
    ("mpnet",  "balanced", False),
    ("mpnet",  "smote",    False),
]


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except ImportError:
        pass
    return "cpu"


def train_config(feature, balancing, sparse):
    run_name = f"{MODEL_NAME}_{feature}_{balancing}"
    log.info(f"\n{'='*60}\n  TRAINING: {run_name}\n{'='*60}")

    X_train, X_val, y_train, y_val = load_features(feature, sparse=sparse)
    X_tr, y_tr, class_weight = apply_balancing(X_train, y_train, balancing, RANDOM_SEED)

    sample_weight = None
    if balancing == "balanced":
        sample_weight = compute_sample_weight("balanced", y_tr)

    device = detect_device()
    log.info(f"  Device: {device}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=127,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        device_type=device,
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    elapsed = time.perf_counter() - t0

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, y_pred, y_prob)
    metrics.update({"train_time_s": round(elapsed, 2),
                    "feature": feature, "balancing": balancing,
                    "model": MODEL_NAME, "device": device})

    log_results(MODEL_NAME, feature, balancing, metrics, elapsed)
    save_model(model, run_name, metrics=metrics, config={
        "feature": feature, "balancing": balancing,
        "n_estimators": 500, "num_leaves": 127, "device": device
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

    log.info("\n" + "=" * 60)
    log.info("  LIGHTGBM — SUMMARY (sorted by MCC)")
    log.info("=" * 60)
    log.info(f"  {'Config':<25} {'MCC':>8} {'F1':>8} {'FAR':>8}")
    for key, m in sorted(results.items(), key=lambda x: -x[1]["mcc"]):
        log.info(f"  {key:<25} {m['mcc']:>8.4f} {m['f1']:>8.4f} {m['false_alarm_rate']:>8.4f}")


if __name__ == "__main__":
    main()
