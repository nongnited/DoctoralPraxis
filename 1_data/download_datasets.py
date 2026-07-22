"""
RAGShield — Dataset Download Script
Downloads all training datasets from HuggingFace and other sources.

Datasets downloaded:
  Injection:
    - microsoft/BIPIA           (~200K examples)
    - LLMail-Inject             (~173K examples)
    - HuggingFaceH4/TensorTrust (~160K examples)

  Benign:
    - Enron email corpus
    - ArXiv abstracts
    - Wikipedia (English)
    - StackExchange
    - Financial news

Usage:
    python 1_data/download_datasets.py
    python 1_data/download_datasets.py --injection-only
    python 1_data/download_datasets.py --benign-only
"""

import os
import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import requests
import zipfile
import json

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Injection Datasets ────────────────────────────────────────────────────

def download_bipia():
    """Download BIPIA (Benchmark for Indirect Prompt Injection Attacks)."""
    log.info("Downloading BIPIA...")
    out = DATA_DIR / "raw" / "bipia"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("microsoft/BIPIA", trust_remote_code=True)
        for split, data in ds.items():
            data.to_json(out / f"{split}.jsonl")
            log.info(f"  BIPIA {split}: {len(data)} examples → {out}/{split}.jsonl")
    except Exception as e:
        log.error(f"BIPIA download failed: {e}")
        log.info("  Try: huggingface-cli login")
        raise


def download_tensortrust():
    """Download TensorTrust dataset."""
    log.info("Downloading TensorTrust...")
    out = DATA_DIR / "raw" / "tensortrust"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("HuggingFaceH4/TensorTrust", trust_remote_code=True)
        for split, data in ds.items():
            data.to_json(out / f"{split}.jsonl")
            log.info(f"  TensorTrust {split}: {len(data)} examples → {out}/{split}.jsonl")
    except Exception as e:
        log.error(f"TensorTrust download failed: {e}")
        raise


def download_llmail():
    """Download LLMail-Inject dataset."""
    log.info("Downloading LLMail-Inject...")
    out = DATA_DIR / "raw" / "llmail"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("microsoft/LLMail-Inject", trust_remote_code=True)
        for split, data in ds.items():
            data.to_json(out / f"{split}.jsonl")
            log.info(f"  LLMail {split}: {len(data)} examples → {out}/{split}.jsonl")
    except Exception as e:
        log.warning(f"LLMail HuggingFace download failed: {e}")
        log.info("  Attempting direct download...")
        _download_llmail_direct(out)


def _download_llmail_direct(out: Path):
    """Fallback: download LLMail from GitHub release."""
    url = "https://github.com/microsoft/LLMail-Inject/releases/download/v1.0/llmail_inject.zip"
    zip_path = out / "llmail_inject.zip"
    log.info(f"  Downloading from {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out)
    log.info(f"  LLMail extracted to {out}")


# ── Benign Datasets ───────────────────────────────────────────────────────

def download_enron():
    """Download Enron email corpus."""
    log.info("Downloading Enron emails...")
    out = DATA_DIR / "raw" / "benign" / "enron"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("snoop2head/enron_emails", trust_remote_code=True)
        for split, data in ds.items():
            data.to_json(out / f"{split}.jsonl")
            log.info(f"  Enron {split}: {len(data)} examples")
    except Exception as e:
        log.error(f"Enron download failed: {e}")
        raise


def download_arxiv():
    """Download arXiv abstracts."""
    log.info("Downloading arXiv abstracts...")
    out = DATA_DIR / "raw" / "benign" / "arxiv"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-1T-Sample",
            data_files={"train": "arxiv*.jsonl"},
            trust_remote_code=True
        )
        ds["train"].to_json(out / "arxiv.jsonl")
        log.info(f"  arXiv: {len(ds['train'])} examples")
    except Exception as e:
        log.warning(f"arXiv RedPajama failed: {e}, trying arxiv-summarization...")
        ds = load_dataset("ccdv/arxiv-summarization", trust_remote_code=True)
        for split, data in ds.items():
            data.to_json(out / f"{split}.jsonl")
            log.info(f"  arXiv {split}: {len(data)} examples")


def download_wikipedia():
    """Download Wikipedia articles (English, sample)."""
    log.info("Downloading Wikipedia...")
    out = DATA_DIR / "raw" / "benign" / "wikipedia"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            trust_remote_code=True,
            streaming=True
        )
        records = []
        for i, record in enumerate(ds["train"]):
            records.append({"text": record["text"][:2000], "title": record["title"]})
            if i >= 200000:
                break

        import json
        with open(out / "wikipedia_200k.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        log.info(f"  Wikipedia: {len(records)} articles")
    except Exception as e:
        log.error(f"Wikipedia download failed: {e}")
        raise


def download_stackexchange():
    """Download StackExchange Q&A."""
    log.info("Downloading StackExchange...")
    out = DATA_DIR / "raw" / "benign" / "stackexchange"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(
            "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
            trust_remote_code=True
        )
        for split, data in ds.items():
            data.to_json(out / f"{split}.jsonl")
            log.info(f"  StackExchange {split}: {len(data)} examples")
    except Exception as e:
        log.error(f"StackExchange download failed: {e}")
        raise


def download_financial():
    """Download financial news text."""
    log.info("Downloading financial news...")
    out = DATA_DIR / "raw" / "benign" / "financial"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("zeroshot/twitter-financial-news-topic", trust_remote_code=True)
        for split, data in ds.items():
            data.to_json(out / f"{split}.jsonl")
            log.info(f"  Financial {split}: {len(data)} examples")
    except Exception as e:
        log.error(f"Financial news download failed: {e}")
        raise


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download RAGShield training datasets")
    parser.add_argument("--injection-only", action="store_true", help="Download only injection datasets")
    parser.add_argument("--benign-only",    action="store_true", help="Download only benign datasets")
    parser.add_argument("--skip-wikipedia", action="store_true", help="Skip Wikipedia (large download)")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        log.info("Logged into HuggingFace")

    if not args.benign_only:
        log.info("=" * 60)
        log.info("INJECTION DATASETS")
        log.info("=" * 60)
        download_bipia()
        download_tensortrust()
        download_llmail()

    if not args.injection_only:
        log.info("=" * 60)
        log.info("BENIGN DATASETS")
        log.info("=" * 60)
        download_enron()
        download_arxiv()
        download_stackexchange()
        download_financial()
        if not args.skip_wikipedia:
            download_wikipedia()
        else:
            log.info("Skipping Wikipedia (--skip-wikipedia)")

    log.info("=" * 60)
    log.info("Download complete. Run next: python 1_data/preprocess_injection.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
