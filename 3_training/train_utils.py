"""
RAGShield — Shared Training Utilities
Common functions used by all classifier training scripts.
"""

import os
import json
import time
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.metrics import (
    f1_score, matthews_corrcoef, precision_score,
    recall_score, accuracy_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

DATA_DIR   = Path(os.getenv("DATA_DIR",   "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
FEAT_DIR   = DATA_DIR / "features"


def load_features(feature: str, sparse: bool = False) -> Tuple:
    """
    Load pre-extracted features and labels.

    Args:
        feature: one of 'tfidf', 'minilm', 'mpnet'
        sparse:  True for TF-IDF (scipy sparse), False for dense numpy

    Returns:
        X_train, X_val, y_train, y_val
    """
    feat_path = FEAT_DIR / feature
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features not found: {feat_path}\n"
            f"Run: python 2_features/extract_{feature}.py"
        )

    if sparse:
        X_train = sp.load_npz(feat_path / "X_train.npz")
        X_val   = sp.load_npz(feat_path / "X_val.npz")
    else:
        X_train = np.load(feat_path / "X_train.npy")
        X_val   = np.load(feat_path / "X_val.npy")

    y_train = np.load(feat_path / "y_train.npy")
    y_val   = np.load(feat_path / "y_val.npy")

    log.info(f"  Loaded {feature}: X_train={X_train.shape}, X_val={X_val.shape}")
    log.info(f"  Train injection rate: {y_train.mean():.3f} | Val: {y_val.mean():.3f}")

    return X_train, X_val, y_train, y_val


def apply_balancing(X_train, y_train, strategy: str, random_seed: int = 42):
    """
    Apply class balancing strategy.

    Args:
        strategy: 'none', 'balanced', or 'smote'
        For 'balanced': pass class_weight='balanced' to classifier instead
        For 'smote': resample training data with SMOTE
    """
    if strategy == "none":
        return X_train, y_train, None

    if strategy == "balanced":
        # Return class_weight dict for sklearn classifiers that support it
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        return X_train, y_train, class_weight

    if strategy == "smote":
        log.info("  Applying SMOTE...")
        smote = SMOTE(random_state=random_seed, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        log.info(f"  After SMOTE: {len(y_res):,} examples "
                 f"({y_res.mean():.1%} injection)")
        return X_res, y_res, None

    raise ValueError(f"Unknown balancing strategy: {strategy}")


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Compute full evaluation metric set."""
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    metrics = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "benign_recall":   float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }

    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0

    return metrics


def save_model(model, model_name: str, extra_files: Dict[str, Any] = None,
               metrics: Dict = None, config: Dict = None):
    """Save trained model, metrics, and config to models/{model_name}/."""
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    log.info(f"  Model saved: {model_dir}/model.joblib")

    if extra_files:
        for fname, obj in extra_files.items():
            joblib.dump(obj, model_dir / fname)
            log.info(f"  Extra file saved: {model_dir}/{fname}")

    if metrics:
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    if config:
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    return model_dir


def log_results(model_name: str, feature: str, balancing: str, metrics: Dict, elapsed: float):
    """Print a formatted result summary."""
    log.info("=" * 70)
    log.info(f"  {model_name.upper()} | {feature} | {balancing}")
    log.info("=" * 70)
    log.info(f"  MCC:       {metrics['mcc']:.4f}")
    log.info(f"  F1:        {metrics['f1']:.4f}")
    log.info(f"  Precision: {metrics['precision']:.4f}")
    log.info(f"  Recall:    {metrics['recall']:.4f}")
    log.info(f"  FAR:       {metrics['false_alarm_rate']:.4f}")
    log.info(f"  AUC:       {metrics['auc']:.4f}")
    log.info(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    log.info(f"  Train time: {elapsed:.1f}s")
    log.info("=" * 70)
