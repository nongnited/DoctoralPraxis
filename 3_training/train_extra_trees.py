"""
RAGShield — Train Extra Trees Classifier
300 trees, sqrt max_features, fully randomized split selection.

Configurations (8 total):
  tfidf_none, tfidf_balanced
  minilm_none, minilm_balanced, minilm_smote
  mpnet_none, mpnet_balanced, mpnet_smote

Usage:
    python 3_training/train_extra_trees.py
"""

import os
import time
import logging

from sklearn.ensemble import ExtraTreesClassifier
from dotenv import load_dotenv

from train_utils import (
    load_features, apply_balancing, compute_metrics,
    save_model, log_results
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME   = "extra_trees"
RANDOM_SEED  = int(os.getenv("RANDOM_SEED", 42))
N_ESTIMATORS = 300

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

    model = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, y_pred, y_prob)
    metrics.update({"train_time_s": round(elapsed, 2),
                    "feature": feature, "balancing": balancing, "model": MODEL_NAME})

    log_results(MODEL_NAME, feature, balancing, metrics, elapsed)
    save_model(model, run_name, metrics=metrics, config={
        "feature": feature, "balancing": balancing,
        "n_estimators": N_ESTIMATORS, "max_features": "sqrt"
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
    log.info("  EXTRA TREES — SUMMARY (sorted by MCC)")
    log.info("=" * 60)
    log.info(f"  {'Config':<25} {'MCC':>8} {'F1':>8} {'FAR':>8}")
    for key, m in sorted(results.items(), key=lambda x: -x[1]["mcc"]):
        log.info(f"  {key:<25} {m['mcc']:>8.4f} {m['f1']:>8.4f} {m['false_alarm_rate']:>8.4f}")


if __name__ == "__main__":
    main()
