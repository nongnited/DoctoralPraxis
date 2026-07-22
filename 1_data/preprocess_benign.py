"""
RAGShield — Benign Dataset Preprocessing
Loads, cleans, and normalizes all benign text sources.
Produces a combined benign dataset to balance the injection corpus.

Input:  data/raw/benign/
Output: data/processed/benign_combined.parquet
        data/processed/benign_stats.json

Usage:
    python 1_data/preprocess_benign.py
"""

import os
import json
import logging
import hashlib
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR  = DATA_DIR / "raw" / "benign"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_LEN = 50
MAX_TEXT_LEN = 2000

# ── Loaders ───────────────────────────────────────────────────────────────

def load_enron() -> pd.DataFrame:
    log.info("Loading Enron emails...")
    frames = []
    for f in sorted((RAW_DIR / "enron").glob("*.jsonl")):
        df = pd.read_json(f, lines=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    text_col = next((c for c in ["text", "body", "message", "content"] if c in df.columns), df.columns[0])
    result = pd.DataFrame({"text": df[text_col].astype(str), "source": "enron"})
    log.info(f"  Enron: {len(result):,} examples")
    return result


def load_arxiv() -> pd.DataFrame:
    log.info("Loading arXiv...")
    frames = []
    for f in sorted((RAW_DIR / "arxiv").glob("*.jsonl")):
        df = pd.read_json(f, lines=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    text_col = next((c for c in ["abstract", "text", "body", "content"] if c in df.columns), df.columns[0])
    result = pd.DataFrame({"text": df[text_col].astype(str), "source": "arxiv"})
    log.info(f"  arXiv: {len(result):,} examples")
    return result


def load_wikipedia() -> pd.DataFrame:
    log.info("Loading Wikipedia...")
    frames = []
    for f in sorted((RAW_DIR / "wikipedia").glob("*.jsonl")):
        df = pd.read_json(f, lines=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    text_col = next((c for c in ["text", "content", "article"] if c in df.columns), df.columns[0])
    result = pd.DataFrame({"text": df[text_col].astype(str), "source": "wikipedia"})
    log.info(f"  Wikipedia: {len(result):,} examples")
    return result


def load_stackexchange() -> pd.DataFrame:
    log.info("Loading StackExchange...")
    frames = []
    for f in sorted((RAW_DIR / "stackexchange").glob("*.jsonl")):
        df = pd.read_json(f, lines=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Combine question title + body + answer
    if "title_body" in df.columns:
        text = df["title_body"]
    elif "question" in df.columns and "answer" in df.columns:
        text = df["question"].astype(str) + " " + df["answer"].astype(str)
    else:
        text = df.iloc[:, 0]
    result = pd.DataFrame({"text": text.astype(str), "source": "stackexchange"})
    log.info(f"  StackExchange: {len(result):,} examples")
    return result


def load_financial() -> pd.DataFrame:
    log.info("Loading financial news...")
    frames = []
    for f in sorted((RAW_DIR / "financial").glob("*.jsonl")):
        df = pd.read_json(f, lines=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    text_col = next((c for c in ["text", "headline", "body", "content"] if c in df.columns), df.columns[0])
    result = pd.DataFrame({"text": df[text_col].astype(str), "source": "financial"})
    log.info(f"  Financial: {len(result):,} examples")
    return result


# ── Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = " ".join(text.split())
    return text[:MAX_TEXT_LEN]


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"  Deduplicating {len(df):,} examples...")
    df["_hash"] = df["text"].apply(lambda t: hashlib.sha256(t.encode()).hexdigest())
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    log.info(f"  After dedup: {len(df):,} examples")
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    loaders = [
        (load_enron,        "Enron"),
        (load_arxiv,        "arXiv"),
        (load_wikipedia,    "Wikipedia"),
        (load_stackexchange,"StackExchange"),
        (load_financial,    "Financial"),
    ]

    frames = []
    for loader, name in loaders:
        src_dir = RAW_DIR / name.lower()
        if not src_dir.exists() or not any(src_dir.glob("*.jsonl")):
            log.warning(f"Skipping {name}: no files in {src_dir}")
            continue
        try:
            df = loader()
            frames.append(df)
        except Exception as e:
            log.error(f"Error loading {name}: {e}")

    if not frames:
        raise RuntimeError("No benign datasets loaded. Run download_datasets.py first.")

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Combined raw benign: {len(combined):,}")

    combined["text"] = combined["text"].apply(clean_text)
    mask = combined["text"].str.len() >= MIN_TEXT_LEN
    log.info(f"Removing {(~mask).sum():,} examples shorter than {MIN_TEXT_LEN} chars")
    combined = combined[mask].reset_index(drop=True)
    combined = deduplicate(combined)
    combined["label"] = 0  # all benign examples are label 0

    log.info("=" * 60)
    log.info("FINAL BENIGN DATASET STATS")
    log.info("=" * 60)
    log.info(f"Total examples: {len(combined):,}")
    for src, count in combined["source"].value_counts().items():
        log.info(f"  {src}: {count:,}")

    out_path = PROC_DIR / "benign_combined.parquet"
    combined.to_parquet(out_path, index=False)
    log.info(f"\nSaved to: {out_path}")

    stats = {
        "total": len(combined),
        "sources": combined["source"].value_counts().to_dict(),
    }
    with open(PROC_DIR / "benign_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Next: python 1_data/build_corpus.py")


if __name__ == "__main__":
    main()
