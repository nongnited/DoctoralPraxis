"""
RAGShield — Latency Analysis
Reads from PostgreSQL latency_log and produces:
  - Per-component latency summary (retrieval, classification, generation)
  - Per-model classification latency (p50/p95/p99)
  - RAGShield overhead table

Usage:
    python 7_analysis/analyze_latency.py
    python 7_analysis/analyze_latency.py --output results/latency.json
"""

import os
import json
import logging
import argparse
from pathlib import Path

import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

LEDGER_HOST   = os.getenv("LEDGER_HOST", "10.0.0.110")
LEDGER_PORT   = int(os.getenv("LEDGER_PORT", 5432))
POSTGRES_DB   = os.getenv("POSTGRES_DB", "ragshield")
POSTGRES_USER = os.getenv("POSTGRES_USER", "ragshield_app")
POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD", "ragshield2025")


def get_conn():
    return psycopg2.connect(
        host=LEDGER_HOST, port=LEDGER_PORT,
        dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASS,
    )


def percentile(series, p):
    return float(np.percentile(series.dropna(), p))


def analyze():
    log.info("Connecting to PostgreSQL...")
    conn = get_conn()

    # Load all latency records
    df = pd.read_sql(
        "SELECT component, latency_ms, run_id FROM latency_log ORDER BY id",
        conn
    )
    log.info(f"Loaded {len(df):,} latency records")

    # ── Component summary ─────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  COMPONENT LATENCY SUMMARY")
    log.info("=" * 70)
    log.info(f"  {'Component':<20} {'n':>8} {'p50 ms':>10} {'p95 ms':>10} {'avg ms':>10}")

    component_stats = {}
    for comp in ["retrieval", "classification", "generation"]:
        sub = df[df["component"] == comp]["latency_ms"]
        if len(sub) == 0:
            continue
        stats = {
            "n":    len(sub),
            "p50":  round(percentile(sub, 50), 1),
            "p95":  round(percentile(sub, 95), 1),
            "p99":  round(percentile(sub, 99), 1),
            "avg":  round(float(sub.mean()), 1),
        }
        component_stats[comp] = stats
        log.info(f"  {comp:<20} {stats['n']:>8,} {stats['p50']:>10.1f} "
                 f"{stats['p95']:>10.1f} {stats['avg']:>10.1f}")

    # ── Per-model classification latency ─────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  PER-MODEL CLASSIFICATION LATENCY (p50/p95/p99)")
    log.info("=" * 70)
    log.info(f"  {'Model':<25} {'n':>6} {'p50':>8} {'p95':>8} {'p99':>8}")

    cls_df = df[df["component"] == "classification"].copy()
    model_stats = {}

    for run_id in sorted(cls_df["run_id"].dropna().unique()):
        # Extract model name from run_id (format: model_scenario_timestamp)
        model = run_id.split("_")[0] if run_id else "unknown"
        sub = cls_df[cls_df["run_id"] == run_id]["latency_ms"]
        if len(sub) == 0:
            continue
        stats = {
            "n":   len(sub),
            "p50": round(percentile(sub, 50), 1),
            "p95": round(percentile(sub, 95), 1),
            "p99": round(percentile(sub, 99), 1),
            "avg": round(float(sub.mean()), 1),
        }
        model_stats[model] = stats
        log.info(f"  {model:<25} {stats['n']:>6,} {stats['p50']:>8.1f} "
                 f"{stats['p95']:>8.1f} {stats['p99']:>8.1f}")

    # ── Overhead table ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  RAGSHIELD OVERHEAD TABLE")
    log.info("  % Overhead = Classification ÷ (Retrieval + Generation)")
    log.info("=" * 70)
    log.info(f"  {'Model':<25} {'Ret+Gen':>10} {'Cls':>8} {'Total':>8} {'Overhead':>10}")

    ret_p50 = component_stats.get("retrieval", {}).get("p50", 0)
    gen_p50 = component_stats.get("generation", {}).get("p50", 0)
    baseline = ret_p50 + gen_p50

    overhead_table = []
    for model, stats in sorted(model_stats.items(), key=lambda x: x[1]["p50"]):
        cls_ms = stats["p50"]
        total  = baseline + cls_ms
        pct    = cls_ms / baseline * 100 if baseline > 0 else 0
        overhead_table.append({
            "model": model, "baseline_ms": baseline,
            "cls_p50_ms": cls_ms, "total_ms": total, "overhead_pct": pct
        })
        log.info(f"  {model:<25} {baseline:>10.0f} {cls_ms:>8.1f} "
                 f"{total:>8.0f} {pct:>9.1f}%")

    conn.close()

    return {
        "component_stats": component_stats,
        "model_stats":     model_stats,
        "overhead_table":  overhead_table,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = analyze()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved to {out}")


if __name__ == "__main__":
    main()
