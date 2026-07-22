"""
RAGShield — Vault RAG Gateway
FastAPI v2 service that orchestrates the full RAGShield pipeline:
  1. Receive user query
  2. Retrieve relevant document chunks from ChromaDB
  3. Send each chunk to shield classifier
  4. If any chunk is flagged as injection → block and return warning
  5. If all chunks are clean → send to Ollama for generation
  6. Log all latencies to PostgreSQL

Environment variables (see .env.example):
  CHROMADB_HOST, CHROMADB_PORT, CHROMA_COLLECTION
  SHIELD_HOST, SHIELD_PORT (active classifier port)
  ARBITER_HOST, ARBITER_PORT
  LEDGER_HOST, LEDGER_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
  CHUNK_SIZE, TOP_K, LLM_MODEL, EMBED_MODEL
"""

import os
import time
import logging
import hashlib
from datetime import datetime
from typing import Optional

import httpx
import chromadb
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
CHROMADB_HOST    = os.getenv("CHROMADB_HOST", "chromadb")
CHROMADB_PORT    = int(os.getenv("CHROMADB_PORT", 8000))
CHROMA_COLLECTION= os.getenv("CHROMA_COLLECTION", "ragshield_docs")

SHIELD_HOST      = os.getenv("SHIELD_HOST", "10.0.0.108")
SHIELD_PORT      = int(os.getenv("SHIELD_PORT", 8101))  # active model port

ARBITER_HOST     = os.getenv("ARBITER_HOST", "10.0.0.109")
ARBITER_PORT     = int(os.getenv("ARBITER_PORT", 11434))
LLM_MODEL        = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "nomic-embed-text")

LEDGER_HOST      = os.getenv("LEDGER_HOST", "10.0.0.110")
LEDGER_PORT      = int(os.getenv("LEDGER_PORT", 5432))
POSTGRES_DB      = os.getenv("POSTGRES_DB", "ragshield")
POSTGRES_USER    = os.getenv("POSTGRES_USER", "ragshield_app")
POSTGRES_PASS    = os.getenv("POSTGRES_PASSWORD", "ragshield2025")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", 100))
TOP_K            = int(os.getenv("TOP_K", 5))

app = FastAPI(title="RAGShield Vault Gateway")

# ── Connections ───────────────────────────────────────────────────────────
_chroma = None
_collection = None
_db_conn = None


def get_chroma():
    global _chroma, _collection
    if _chroma is None:
        _chroma = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        _collection = _chroma.get_or_create_collection(CHROMA_COLLECTION)
        log.info(f"ChromaDB connected: {CHROMADB_HOST}:{CHROMADB_PORT} "
                 f"| collection={CHROMA_COLLECTION} "
                 f"| docs={_collection.count()}")
    return _collection


def get_db():
    global _db_conn
    try:
        if _db_conn is None or _db_conn.closed:
            _db_conn = psycopg2.connect(
                host=LEDGER_HOST, port=LEDGER_PORT,
                dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASS,
            )
            _db_conn.autocommit = True
    except Exception as e:
        log.error(f"DB connection failed: {e}")
        _db_conn = None
    return _db_conn


# ── Models ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    shield_active: bool = True
    top_k: Optional[int] = None
    run_id: Optional[str] = None
    doc_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    blocked: bool
    flagged_chunks: int
    total_chunks: int
    retrieval_ms: float
    classification_ms: float
    generation_ms: float
    total_ms: float
    run_id: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into word-level chunks."""
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


def embed_text(text: str) -> list[float]:
    """Get embedding from Ollama on arbiter."""
    url = f"http://{ARBITER_HOST}:{ARBITER_PORT}/api/embeddings"
    r = httpx.post(url, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def classify_chunk(text: str) -> dict:
    """Send chunk to active shield classifier."""
    url = f"http://{SHIELD_HOST}:{SHIELD_PORT}/classify"
    r = httpx.post(url, json={"text": text}, timeout=30)
    r.raise_for_status()
    return r.json()


def generate(query: str, context: str) -> str:
    """Generate answer using Ollama llama3.1:8b."""
    prompt = (
        f"You are a helpful assistant. Use only the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    url = f"http://{ARBITER_HOST}:{ARBITER_PORT}/api/generate"
    r = httpx.post(
        url,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["response"]


def log_latency(component: str, ms: float, run_id: str = None,
                doc_id: str = None, label: int = None):
    """Write latency record to PostgreSQL latency_log."""
    db = get_db()
    if db is None:
        return
    try:
        with db.cursor() as cur:
            cur.execute(
                """INSERT INTO latency_log
                   (component, latency_ms, run_id, doc_id, label, recorded_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (component, ms, run_id, doc_id, label, datetime.utcnow())
            )
    except Exception as e:
        log.warning(f"Latency log write failed: {e}")


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    collection = get_chroma()
    return {
        "status": "ok",
        "docs_in_collection": collection.count() if collection else 0,
        "shield_host": f"{SHIELD_HOST}:{SHIELD_PORT}",
        "llm_model": LLM_MODEL,
        "chunk_size": CHUNK_SIZE,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k = req.top_k or TOP_K
    run_id = req.run_id
    doc_id = req.doc_id
    t_start = time.perf_counter()

    # ── 1. Retrieve ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    collection = get_chroma()
    query_embedding = embed_text(req.query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000
    log_latency("retrieval", retrieval_ms, run_id, doc_id)

    raw_docs = results["documents"][0] if results["documents"] else []
    chunks = []
    for doc in raw_docs:
        chunks.extend(chunk_text(doc))

    # ── 2. Classify ──────────────────────────────────────────────────────
    classification_ms = 0.0
    flagged = 0
    blocked = False

    if req.shield_active and chunks:
        t0 = time.perf_counter()
        for chunk in chunks:
            result = classify_chunk(chunk)
            if result["label"] == 1:
                flagged += 1
        classification_ms = (time.perf_counter() - t0) * 1000
        log_latency("classification", classification_ms, run_id, doc_id,
                    label=1 if flagged > 0 else 0)
        if flagged > 0:
            blocked = True

    # ── 3. Generate ──────────────────────────────────────────────────────
    generation_ms = 0.0
    answer = ""

    if not blocked:
        context = "\n\n".join(chunks[:top_k])
        t0 = time.perf_counter()
        answer = generate(req.query, context)
        generation_ms = (time.perf_counter() - t0) * 1000
        log_latency("generation", generation_ms, run_id, doc_id)
    else:
        answer = (
            "⚠️ RAGShield blocked this query. One or more retrieved documents "
            "were flagged as potential prompt injection attempts."
        )

    total_ms = (time.perf_counter() - t_start) * 1000

    return QueryResponse(
        answer=answer,
        blocked=blocked,
        flagged_chunks=flagged,
        total_chunks=len(chunks),
        retrieval_ms=round(retrieval_ms, 2),
        classification_ms=round(classification_ms, 2),
        generation_ms=round(generation_ms, 2),
        total_ms=round(total_ms, 2),
        run_id=run_id,
    )


@app.post("/upload")
def upload_document(doc_id: str, text: str, label: int = 0):
    """Upload a document to ChromaDB."""
    collection = get_chroma()
    embedding = embed_text(text)
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"label": label}]
    )
    return {"status": "uploaded", "doc_id": doc_id}


@app.delete("/corpus")
def clear_corpus():
    """Clear all documents from ChromaDB collection."""
    global _collection
    _chroma.delete_collection(CHROMA_COLLECTION)
    _collection = _chroma.get_or_create_collection(CHROMA_COLLECTION)
    return {"status": "cleared"}
