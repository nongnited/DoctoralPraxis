"""
RAGShield — Training Results Analysis
Reads all metrics.json files from models/ directory and produces:
  - Best config per model table (sorted by MCC)
  - Feature comparison table (tfidf vs minilm vs mpnet)
  - Balancing strategy comparison (none vs balanced vs smote)
  - Full metrics CSV

Usage:
    python 7_analysis/analyze_training_results.py
    python 7_analysis/analyze_training_results.py --models-dir /path/to/models
"""

import os
import json
import logging
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
RESULTS_DIR = Path("7_analysis/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Display order for models
MODEL_ORDER = [
    "securebert", "bert_base", "lightgbm", "extra_trees",
    "svm", "xgboost", "random_forest", "logistic_regression",
    "ridge", "naive_bayes",
]


def load_all_metrics(models_dir: Path) -> pd.DataFrame:
    records = []
    for metrics_file in sorted(models_dir.rglob("metrics.json")):
        try:
            with open(metrics_file) as f:
                m = json.load(f)
            # Extract config from directory name
            run_name = metrics_file.parent.name
            parts = run_name.split("_")
            model    = m.get("model", parts[0] if parts else "unknown")
            feature  = m.get("feature",  "unknown")
            balancing= m.get("balancing","unknown")

            records.append({
                "run_name":      run_name,
                "model":         model,
                "feature":       feature,
                "balancing":     balancing,
                "mcc":           m.get("mcc", 0),
                "f1":            m.get("f1", 0),
                "precision":     m.get("precision", 0),
                "recall":        m.get("recall", 0),
                "benign_recall": m.get("benign_recall", 0),
                "far":           m.get("false_alarm_rate", 0),
                "auc":           m.get("auc", 0),
                "train_time_s":  m.get("train_time_s", m.get("total_train_time", 0)),
                "tp": m.get("tp", 0), "fp": m.get("fp", 0),
                "tn": m.get("tn", 0), "fn": m.get("fn", 0),
            })
        except Exception as e:
            log.warning(f"Could not load {metrics_file}: {e}")

    return pd.DataFrame(records)


def best_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Return the best configuration per model by MCC."""
    idx = df.groupby("model")["mcc"].idxmax()
    best = df.loc[idx].copy()
    best["config"] = best["feature"] + "_" + best["balancing"]
    return best.sort_values("mcc", ascending=False)


def feature_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Average MCC per model × feature combination."""
    return (
        df.groupby(["model", "feature"])["mcc"]
        .mean()
        .unstack("feature")
        .round(4)
        .sort_values("tfidf", ascending=False)
    )


def balancing_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Average MCC per model × balancing strategy."""
    return (
        df.groupby(["model", "balancing"])["mcc"]
        .mean()
        .unstack("balancing")
        .round(4)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        log.error(f"Models directory not found: {models_dir}")
        return

    log.info(f"Loading metrics from: {models_dir}")
    df = load_all_metrics(models_dir)

    if df.empty:
        log.error("No metrics.json files found. Run training first.")
        return

    log.info(f"Loaded {len(df)} training configurations")

    # ── Best per model ────────────────────────────────────────────────────
    best = best_per_model(df)
    log.info("\n" + "=" * 80)
    log.info("  BEST CONFIGURATION PER MODEL (sorted by MCC)")
    log.info("=" * 80)
    log.info(f"  {'Model':<25} {'Config':<20} {'MCC':>7} {'F1':>7} {'Prec':>7} "
             f"{'Rec':>7} {'BenRec':>8} {'FAR':>7}")
    for _, row in best.iterrows():
        log.info(f"  {row['model']:<25} {row['config']:<20} "
                 f"{row['mcc']:>7.4f} {row['f1']:>7.4f} "
                 f"{row['precision']:>7.4f} {row['recall']:>7.4f} "
                 f"{row['benign_recall']:>8.4f} {row['far']:>7.4f}")

    # ── Feature comparison ────────────────────────────────────────────────
    feat_cmp = feature_comparison(df)
    log.info("\n" + "=" * 60)
    log.info("  FEATURE COMPARISON (avg MCC per model × feature)")
    log.info("=" * 60)
    log.info(feat_cmp.to_string())

    # ── Balancing comparison ──────────────────────────────────────────────
    bal_cmp = balancing_comparison(df)
    log.info("\n" + "=" * 60)
    log.info("  BALANCING COMPARISON (avg MCC per model × balancing)")
    log.info("=" * 60)
    log.info(bal_cmp.to_string())

    # Save
    df.to_csv(RESULTS_DIR / "all_training_metrics.csv", index=False)
    best.to_csv(RESULTS_DIR / "best_per_model.csv", index=False)
    feat_cmp.to_csv(RESULTS_DIR / "feature_comparison.csv")
    bal_cmp.to_csv(RESULTS_DIR / "balancing_comparison.csv")
    log.info(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
