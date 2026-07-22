"""
RAGShield — Core Evaluation Pipeline
Queries the vault gateway with each document in the evaluation corpus
and records the classifier decision, latency, and ground truth label.

Called by run_all_models.py for each scenario × model combination.

Usage (direct):
    python 6_evaluation/pipeline.py \\
        --model naive_bayes --shield-port 8101 \\
        --injection-pct 0.10 --run-id s1_naive_bayes_20260610
"""

import os
import json
import time
import logging
import argparse
import random
from pathlib import Path
from datetime import datetime

import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

VAULT_HOST    = os.getenv("VAULT_HOST", "10.0.0.107")
VAULT_PORT    = int(os.getenv("VAULT_PORT", 8080))
LEDGER_HOST   = os.getenv("LEDGER_HOST", "10.0.0.110")
LEDGER_PORT   = int(os.getenv("LEDGER_PORT", 5432))
POSTGRES_DB   = os.getenv("POSTGRES_DB", "ragshield")
POSTGRES_USER = os.getenv("POSTGRES_USER", "ragshield_app")
POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD", "ragshield2025")
CORPUS_DIR    = Path(os.getenv("CORPUS_DIR", "./5_corpus/corpus"))
RESULTS_DIR   = Path("6_evaluation/results")
RANDOM_SEED   = int(os.getenv("RANDOM_SEED", 42))

# Model → port mapping (matches docker-compose.yml)
MODEL_PORTS = {
    "naive_bayes":         8101,
    "logistic_regression": 8102,
    "ridge":               8103,
    "random_forest":       8104,
    "extra_trees":         8105,
    "xgboost":             8106,
    "lightgbm":            8107,
    "svm":                 8108,
    "bert_base":           8109,
    "securebert":          8110,
}


def get_db():
    return psycopg2.connect(
        host=LEDGER_HOST, port=LEDGER_PORT,
        dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASS,
    )


def collect_docs(injection_pct: float, random_seed: int = RANDOM_SEED):
    """
    Collect evaluation documents at specified injection rate.
    Returns list of (doc_id, text, true_label) tuples.
    """
    random.seed(random_seed)

    benign_files = sorted(CORPUS_DIR.glob("benign/**/*.txt"))
    inject_files = sorted(CORPUS_DIR.glob("injected/**/*.txt"))

    if not benign_files:
        raise FileNotFoundError(f"No benign docs in {CORPUS_DIR}/benign/")
    if not inject_files:
        raise FileNotFoundError(f"No injected docs in {CORPUS_DIR}/injected/")

    # Calculate how many of each to include
    total = len(benign_files) + len(inject_files)
    n_inject = int(total * injection_pct)
    n_benign = total - n_inject

    selected_inject = random.sample(inject_files, min(n_inject, len(inject_files)))
    selected_benign = random.sample(benign_files, min(n_benign, len(benign_files)))

    docs = []
    for f in selected_benign:
        docs.append((f"benign_{f.stem}", f.read_text(encoding="utf-8"), 0))
    for f in selected_inject:
        docs.append((f"inject_{f.stem}", f.read_text(encoding="utf-8"), 1))

    random.shuffle(docs)
    log.info(f"Evaluation set: {len(docs)} docs | "
             f"benign={sum(1 for _,_,l in docs if l==0)} | "
             f"injected={sum(1 for _,_,l in docs if l==1)}")
    return docs


def run_pipeline(model: str, injection_pct: float, run_id: str,
                 shield_active: bool = True) -> dict:
    """
    Run full evaluation pipeline for one model × scenario.
    Returns results dict with TP/FP/TN/FN and latency summary.
    """
    port = MODEL_PORTS.get(model)
    if port is None:
        raise ValueError(f"Unknown model: {model}. Valid: {list(MODEL_PORTS.keys())}")

    docs = collect_docs(injection_pct)
    results = []
    vault_url = f"http://{VAULT_HOST}:{VAULT_PORT}/query"

    log.info(f"Running evaluation: model={model} port={port} "
             f"injection_pct={injection_pct:.0%} run_id={run_id}")

    # Switch shield to the right port via env update
    # (in practice, the evaluation loop reconfigures vault between model runs)

    t_run_start = time.perf_counter()

    for i, (doc_id, text, true_label) in enumerate(docs):
        try:
            payload = {
                "query": text[:500],  # use first 500 chars as query
                "shield_active": shield_active,
                "run_id": run_id,
                "doc_id": doc_id,
            }
            resp = httpx.post(vault_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            pred_label = 1 if data["blocked"] else 0
            results.append({
                "doc_id":           doc_id,
                "true_label":       true_label,
                "pred_label":       pred_label,
                "retrieval_ms":     data["retrieval_ms"],
                "classification_ms":data["classification_ms"],
                "generation_ms":    data["generation_ms"],
                "total_ms":         data["total_ms"],
                "flagged_chunks":   data["flagged_chunks"],
                "total_chunks":     data["total_chunks"],
            })

            if (i + 1) % 50 == 0:
                log.info(f"  Progress: {i+1}/{len(docs)}")

        except Exception as e:
            log.error(f"  Error on {doc_id}: {e}")

    run_time = time.perf_counter() - t_run_start

    # Compute metrics
    tp = sum(1 for r in results if r["true_label"]==1 and r["pred_label"]==1)
    fp = sum(1 for r in results if r["true_label"]==0 and r["pred_label"]==1)
    tn = sum(1 for r in results if r["true_label"]==0 and r["pred_label"]==0)
    fn = sum(1 for r in results if r["true_label"]==1 and r["pred_label"]==0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # MCC
    denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0.0

    # Latency summary
    cls_latencies = [r["classification_ms"] for r in results if r["classification_ms"] > 0]
    ret_latencies = [r["retrieval_ms"] for r in results]
    gen_latencies = [r["generation_ms"] for r in results if r["generation_ms"] > 0]

    import statistics
    def pct(lst, p):
        if not lst: return 0.0
        s = sorted(lst)
        return s[int(len(s) * p / 100)]

    summary = {
        "run_id":           run_id,
        "model":            model,
        "injection_pct":    injection_pct,
        "n_docs":           len(docs),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "f1":        round(f1, 4),
        "mcc":       round(mcc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "far":       round(far, 4),
        "retrieval_p50":      round(pct(ret_latencies, 50), 1),
        "classification_p50": round(pct(cls_latencies, 50), 1),
        "classification_p95": round(pct(cls_latencies, 95), 1),
        "classification_p99": round(pct(cls_latencies, 99), 1),
        "generation_p50":     round(pct(gen_latencies, 50), 1),
        "run_time_s":     round(run_time, 1),
        "completed_at":   datetime.utcnow().isoformat(),
    }

    log.info(f"  MCC={summary['mcc']} F1={summary['f1']} FAR={summary['far']} "
             f"TP={tp} FP={fp} TN={tn} FN={fn}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_id}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "records": results}, f, indent=2)
    log.info(f"  Saved: {out}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         required=True,  choices=list(MODEL_PORTS.keys()))
    parser.add_argument("--injection-pct", type=float,     required=True, help="e.g. 0.10 for 10%")
    parser.add_argument("--run-id",        type=str,       default=None)
    parser.add_argument("--no-shield",     action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"{args.model}_{int(args.injection_pct*100)}pct_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_pipeline(args.model, args.injection_pct, run_id, shield_active=not args.no_shield)


if __name__ == "__main__":
    main()
