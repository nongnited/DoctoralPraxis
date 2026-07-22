"""
RAGShield — Injection Dataset Preprocessing
Cleans, normalizes, and deduplicates the three injection datasets.
Produces a single combined injection dataset with a binary label column.

Input:  data/raw/bipia/, data/raw/tensortrust/, data/raw/llmail/
Output: data/processed/injection_combined.parquet
        data/processed/injection_stats.json

Usage:
    python 1_data/preprocess_injection.py
"""

import os
import json
import logging
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_LEN = 10    # minimum characters for a valid example
MAX_TEXT_LEN = 10000 # truncate very long texts

# ── Dataset-specific loaders ──────────────────────────────────────────────

def load_bipia() -> pd.DataFrame:
    """Load and normalize BIPIA dataset."""
    log.info("Loading BIPIA...")
    frames = []
    bipia_dir = RAW_DIR / "bipia"

    for f in sorted(bipia_dir.glob("*.jsonl")):
        df = pd.read_json(f, lines=True)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No BIPIA files found in {bipia_dir}")

    df = pd.concat(frames, ignore_index=True)
    log.info(f"  Raw BIPIA: {len(df)} rows, columns: {df.columns.tolist()}")

    # BIPIA column mapping — adjust if your version differs
    text_col = next(
        (c for c in ["text", "input", "prompt", "content", "body"] if c in df.columns),
        df.columns[0]
    )
    label_col = next(
        (c for c in ["label", "is_injection", "injected", "attack"] if c in df.columns),
        None
    )

    result = pd.DataFrame()
    result["text"]   = df[text_col].astype(str)
    result["label"]  = df[label_col].astype(int) if label_col else 1
    result["source"] = "bipia"

    log.info(f"  BIPIA loaded: {len(result)} examples, "
             f"injection rate: {result['label'].mean():.1%}")
    return result


def load_tensortrust() -> pd.DataFrame:
    """Load and normalize TensorTrust dataset."""
    log.info("Loading TensorTrust...")
    frames = []
    tt_dir = RAW_DIR / "tensortrust"

    for f in sorted(tt_dir.glob("*.jsonl")):
        df = pd.read_json(f, lines=True)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No TensorTrust files found in {tt_dir}")

    df = pd.concat(frames, ignore_index=True)
    log.info(f"  Raw TensorTrust: {len(df)} rows")

    text_col = next(
        (c for c in ["attack", "prompt", "text", "input", "hijacking_prompt"] if c in df.columns),
        df.columns[0]
    )

    result = pd.DataFrame()
    result["text"]   = df[text_col].astype(str)
    result["label"]  = 1  # all TensorTrust examples are injections
    result["source"] = "tensortrust"

    log.info(f"  TensorTrust loaded: {len(result)} injection examples")
    return result


def load_llmail() -> pd.DataFrame:
    """Load and normalize LLMail-Inject dataset."""
    log.info("Loading LLMail-Inject...")
    frames = []
    llm_dir = RAW_DIR / "llmail"

    for f in sorted(llm_dir.glob("**/*.jsonl")):
        df = pd.read_json(f, lines=True)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No LLMail files found in {llm_dir}")

    df = pd.concat(frames, ignore_index=True)
    log.info(f"  Raw LLMail: {len(df)} rows")

    text_col = next(
        (c for c in ["body", "email", "text", "content", "prompt"] if c in df.columns),
        df.columns[0]
    )
    label_col = next(
        (c for c in ["label", "injected", "is_injection"] if c in df.columns),
        None
    )

    result = pd.DataFrame()
    result["text"]   = df[text_col].astype(str)
    result["label"]  = df[label_col].astype(int) if label_col else 1
    result["source"] = "llmail"

    log.info(f"  LLMail loaded: {len(result)} examples")
    return result


# ── Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = str(text).strip()
    text = " ".join(text.split())          # normalize whitespace
    text = text[:MAX_TEXT_LEN]             # truncate very long texts
    return text


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates based on SHA-256 hash of text."""
    log.info(f"  Deduplicating {len(df)} examples...")
    df["_hash"] = df["text"].apply(lambda t: hashlib.sha256(t.encode()).hexdigest())
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    log.info(f"  After deduplication: {len(df)} examples")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    frames = []

    for loader, name in [(load_bipia, "BIPIA"),
                         (load_tensortrust, "TensorTrust"),
                         (load_llmail, "LLMail")]:
        try:
            df = loader()
            frames.append(df)
        except FileNotFoundError as e:
            log.warning(f"Skipping {name}: {e}")
        except Exception as e:
            log.error(f"Error loading {name}: {e}")
            raise

    if not frames:
        raise RuntimeError("No datasets loaded. Run download_datasets.py first.")

    log.info("=" * 60)
    log.info("COMBINING DATASETS")
    log.info("=" * 60)

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Combined raw: {len(combined)} examples")

    # Clean
    combined["text"] = combined["text"].apply(clean_text)

    # Remove too-short examples
    mask = combined["text"].str.len() >= MIN_TEXT_LEN
    log.info(f"Removing {(~mask).sum()} examples shorter than {MIN_TEXT_LEN} chars")
    combined = combined[mask].reset_index(drop=True)

    # Deduplicate
    combined = deduplicate(combined)

    # Ensure binary labels
    combined["label"] = combined["label"].astype(int).clip(0, 1)

    log.info("=" * 60)
    log.info("FINAL INJECTION DATASET STATS")
    log.info("=" * 60)
    log.info(f"Total examples:    {len(combined):,}")
    log.info(f"Injections (1):    {combined['label'].sum():,} ({combined['label'].mean():.1%})")
    log.info(f"Benign (0):        {(combined['label'] == 0).sum():,}")
    log.info(f"Source breakdown:")
    for src, count in combined["source"].value_counts().items():
        log.info(f"  {src}: {count:,}")

    # Save
    out_path = PROC_DIR / "injection_combined.parquet"
    combined.to_parquet(out_path, index=False)
    log.info(f"\nSaved to: {out_path}")

    stats = {
        "total": len(combined),
        "injections": int(combined["label"].sum()),
        "benign": int((combined["label"] == 0).sum()),
        "sources": combined["source"].value_counts().to_dict(),
    }
    with open(PROC_DIR / "injection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Next: python 1_data/preprocess_benign.py")


if __name__ == "__main__":
    main()
