"""
RAGShield — Build Final Training Corpus
Combines injection and benign datasets, applies stratified 80/20 split.
Produces train and validation parquet files ready for feature extraction.

Input:  data/processed/injection_combined.parquet
        data/processed/benign_combined.parquet
Output: data/processed/train.parquet
        data/processed/val.parquet
        data/processed/corpus_stats.json

Usage:
    python 1_data/build_corpus.py
    python 1_data/build_corpus.py --benign-ratio 1.0   # equal injection/benign
    python 1_data/build_corpus.py --benign-ratio 2.0   # 2x more benign
"""

import os
import json
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PROC_DIR = DATA_DIR / "processed"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TRAIN_SPLIT  = float(os.getenv("TRAIN_TEST_SPLIT", 0.8))


def main():
    parser = argparse.ArgumentParser(description="Build RAGShield training corpus")
    parser.add_argument("--benign-ratio", type=float, default=2.5,
                        help="Ratio of benign to injection examples (default: 2.5)")
    args = parser.parse_args()

    # Load injection dataset
    inj_path = PROC_DIR / "injection_combined.parquet"
    if not inj_path.exists():
        raise FileNotFoundError(f"{inj_path} not found. Run preprocess_injection.py first.")
    injection = pd.read_parquet(inj_path)
    log.info(f"Injection examples: {len(injection):,}")

    # Load benign dataset
    ben_path = PROC_DIR / "benign_combined.parquet"
    if not ben_path.exists():
        raise FileNotFoundError(f"{ben_path} not found. Run preprocess_benign.py first.")
    benign = pd.read_parquet(ben_path)
    log.info(f"Benign examples available: {len(benign):,}")

    # Sample benign to match ratio
    n_benign = min(int(len(injection) * args.benign_ratio), len(benign))
    benign_sample = benign.sample(n=n_benign, random_state=RANDOM_SEED)
    log.info(f"Benign examples sampled: {n_benign:,} (ratio {args.benign_ratio}x)")

    # Combine
    combined = pd.concat(
        [injection[["text", "label"]], benign_sample[["text", "label"]]],
        ignore_index=True
    ).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    log.info(f"Combined total: {len(combined):,}")
    log.info(f"  Injection (label=1): {combined['label'].sum():,} ({combined['label'].mean():.1%})")
    log.info(f"  Benign    (label=0): {(combined['label']==0).sum():,}")

    # Stratified split
    train, val = train_test_split(
        combined,
        test_size=1 - TRAIN_SPLIT,
        stratify=combined["label"],
        random_state=RANDOM_SEED
    )

    log.info(f"Train: {len(train):,} examples")
    log.info(f"  Injection: {train['label'].sum():,} ({train['label'].mean():.1%})")
    log.info(f"Val:   {len(val):,} examples")
    log.info(f"  Injection: {val['label'].sum():,} ({val['label'].mean():.1%})")

    # Save
    train.to_parquet(PROC_DIR / "train.parquet", index=False)
    val.to_parquet(PROC_DIR / "val.parquet", index=False)
    log.info(f"Saved: {PROC_DIR}/train.parquet")
    log.info(f"Saved: {PROC_DIR}/val.parquet")

    stats = {
        "total": len(combined),
        "train": len(train),
        "val": len(val),
        "train_injection_rate": float(train["label"].mean()),
        "val_injection_rate": float(val["label"].mean()),
        "benign_ratio": args.benign_ratio,
        "random_seed": RANDOM_SEED,
    }
    with open(PROC_DIR / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Next: python 2_features/extract_tfidf.py")


if __name__ == "__main__":
    main()
