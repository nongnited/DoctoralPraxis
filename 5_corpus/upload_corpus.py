"""
RAGShield — Upload Meridian Evaluation Corpus to ChromaDB
Reads all .txt files from corpus/benign/ and corpus/injected/,
embeds them via Ollama on arbiter, and uploads to ChromaDB on vault.

Corpus structure (500 documents):
  corpus/benign/emails/     (80 docs)
  corpus/benign/webpages/   (80 docs)
  corpus/benign/news/       (80 docs)
  corpus/benign/code/       (80 docs)
  corpus/benign/generic/    (80 docs)
  corpus/injected/emails/   (20 docs)
  corpus/injected/webpages/ (20 docs)
  corpus/injected/news/     (20 docs)
  corpus/injected/code/     (20 docs)
  corpus/injected/generic/  (20 docs)

Usage:
    python 5_corpus/upload_corpus.py
    python 5_corpus/upload_corpus.py --clear    # clear collection first
    python 5_corpus/upload_corpus.py --dry-run  # count files only
"""

import os
import argparse
import logging
from pathlib import Path

import httpx
import chromadb
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

VAULT_HOST       = os.getenv("VAULT_HOST", "10.0.0.107")
VAULT_PORT       = int(os.getenv("VAULT_PORT", 8080))
CHROMADB_HOST    = os.getenv("CHROMADB_HOST", VAULT_HOST)
CHROMADB_PORT    = int(os.getenv("CHROMADB_PORT", 8000))
CHROMA_COLLECTION= os.getenv("CHROMA_COLLECTION", "ragshield_docs")
ARBITER_HOST     = os.getenv("ARBITER_HOST", "10.0.0.109")
ARBITER_PORT     = int(os.getenv("ARBITER_PORT", 11434))
EMBED_MODEL      = os.getenv("EMBED_MODEL", "nomic-embed-text")

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "./5_corpus/corpus"))
BATCH_SIZE = 10


def embed(text: str) -> list[float]:
    url = f"http://{ARBITER_HOST}:{ARBITER_PORT}/api/embeddings"
    r = httpx.post(url, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def collect_files():
    """Return list of (doc_id, path, label) tuples."""
    files = []
    for label_name, label in [("benign", 0), ("injected", 1)]:
        label_dir = CORPUS_DIR / label_name
        for category_dir in sorted(label_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for txt_file in sorted(category_dir.glob("*.txt")):
                doc_id = f"{label_name}_{category_dir.name}_{txt_file.stem}"
                files.append((doc_id, txt_file, label))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear",   action="store_true", help="Clear collection before upload")
    parser.add_argument("--dry-run", action="store_true", help="Count files only, no upload")
    args = parser.parse_args()

    files = collect_files()
    benign   = sum(1 for _, _, l in files if l == 0)
    injected = sum(1 for _, _, l in files if l == 1)

    log.info(f"Corpus: {len(files)} files | benign={benign} | injected={injected}")
    if not files:
        log.error(f"No files found in {CORPUS_DIR}")
        log.error("Run: python 5_corpus/generate_meridian_corpus.py")
        return

    if args.dry_run:
        log.info("Dry run complete.")
        return

    # Connect to ChromaDB directly
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    if args.clear:
        log.info(f"Clearing collection: {CHROMA_COLLECTION}")
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    log.info(f"Collection '{CHROMA_COLLECTION}' has {collection.count()} existing docs")

    # Upload in batches
    batch_ids, batch_docs, batch_embeddings, batch_metas = [], [], [], []
    errors = 0

    for doc_id, path, label in tqdm(files, desc="Uploading corpus"):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            log.warning(f"Empty file: {path}")
            continue
        try:
            embedding = embed(text)
            batch_ids.append(doc_id)
            batch_docs.append(text)
            batch_embeddings.append(embedding)
            batch_metas.append({"label": label, "category": path.parent.name})

            if len(batch_ids) >= BATCH_SIZE:
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metas,
                )
                batch_ids, batch_docs, batch_embeddings, batch_metas = [], [], [], []

        except Exception as e:
            log.error(f"Failed to embed/upload {doc_id}: {e}")
            errors += 1

    # Upload remaining
    if batch_ids:
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
        )

    final_count = collection.count()
    log.info(f"Upload complete: {final_count} docs in collection | {errors} errors")
    log.info(f"Next: run evaluation pipeline")


if __name__ == "__main__":
    main()
