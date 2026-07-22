"""
RAGShield — Train Ridge Classifier
L2-regularized linear classifier. Very fast training, especially on sparse TF-IDF.
OMP/MKL threads pinned to cpu_count() for maximum parallelism.

Configurations (8 total):
  tfidf_none, tfidf_balanced
  minilm_none, minilm_balanced, minilm_smote
  mpnet_none, mpnet_balanced, mpnet_smote

Usage:
    python 3_training/train_ridge.py
"""

import os
import time
import logging

from sklearn.linear_model import RidgeClassifier
from dotenv import load_dotenv

from train_utils import (
    load_features, apply_balancing, compute_metrics,
    save_model, log_results
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Pin threads for maximum CPU utilization
cpu_count = os.cpu_count()
os.environ["OMP_NUM_THREADS"]     = str(cpu_count)
os.environ["MKL_NUM_THREADS"]     = str(cpu_count)
os.environ["OPENBLAS_NUM_THREADS"]= str(cpu_count)

MODEL_NAME  = "ridge"
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


def train_config(feature, balancing, sparse):
    run_name = f"{MODEL_NAME}_{feature}_{balancing}"
    log.info(f"\n{'='*60}\n  TRAINING: {run_name}\n{'='*60}")

    X_train, X_val, y_train, y_val = load_features(feature, sparse=sparse)
    X_tr, y_tr, class_weight = apply_balancing(X_train, y_train, balancing, RANDOM_SEED)

    model = RidgeClassifier(
        alpha=1.0,
        class_weight=class_weight,
        copy_X=False,  # saves memory on large matrices
    )

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0

    y_pred = model.predict(X_val)
    # Ridge has no predict_proba — use decision function for AUC
    try:
        y_score = model.decision_function(X_val)
    except Exception:
        y_score = None

    metrics = compute_metrics(y_val, y_pred, y_score)
    metrics.update({"train_time_s": round(elapsed, 2),
                    "feature": feature, "balancing": balancing, "model": MODEL_NAME})

    log_results(MODEL_NAME, feature, balancing, metrics, elapsed)
    save_model(model, run_name, metrics=metrics, config={
        "feature": feature, "balancing": balancing, "alpha": 1.0
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
    log.info("  RIDGE CLASSIFIER — SUMMARY (sorted by MCC)")
    log.info("=" * 60)
    log.info(f"  {'Config':<25} {'MCC':>8} {'F1':>8} {'FAR':>8}")
    for key, m in sorted(results.items(), key=lambda x: -x[1]["mcc"]):
        log.info(f"  {key:<25} {m['mcc']:>8.4f} {m['f1']:>8.4f} {m['false_alarm_rate']:>8.4f}")


if __name__ == "__main__":
    main()
