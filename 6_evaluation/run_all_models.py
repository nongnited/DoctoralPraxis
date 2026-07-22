"""
RAGShield — Run All 10 Models for a Scenario
Iterates over all classifiers and runs the evaluation pipeline for each.
Reconfigures vault's shield port between model runs.

Usage:
    python 6_evaluation/run_all_models.py --injection-pct 0.10
    python 6_evaluation/run_all_models.py --injection-pct 0.20 --model ridge
"""

import os
import time
import json
import logging
import argparse
import httpx
from datetime import datetime
from pathlib import Path

from pipeline import run_pipeline, MODEL_PORTS
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

VAULT_HOST = os.getenv("VAULT_HOST", "10.0.0.107")
VAULT_PORT = int(os.getenv("VAULT_PORT", 8080))
SHIELD_HOST = os.getenv("SHIELD_HOST", "10.0.0.108")


def set_shield_model(model: str) -> bool:
    """Tell vault which classifier port to use via an admin endpoint."""
    port = MODEL_PORTS[model]
    # In the actual deployment, vault reads SHIELD_PORT from env.
    # For evaluation, we reload the container between models OR
    # vault accepts a /config endpoint to switch dynamically.
    try:
        url = f"http://{VAULT_HOST}:{VAULT_PORT}/config/shield"
        resp = httpx.post(url, json={"host": SHIELD_HOST, "port": port}, timeout=10)
        if resp.status_code == 200:
            log.info(f"  Shield switched to {model} (port {port})")
            return True
    except Exception as e:
        log.warning(f"  Could not switch shield dynamically: {e}")
        log.warning(f"  Restart vault with SHIELD_PORT={port} before continuing")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--injection-pct", type=float, required=True,
                        help="Injection rate, e.g. 0.10 for 10%%")
    parser.add_argument("--model", type=str, default=None,
                        help="Run single model only")
    parser.add_argument("--no-shield", action="store_true",
                        help="Run without RAGShield (baseline)")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_PORTS.keys())
    pct_str = f"{int(args.injection_pct * 100)}pct"
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    all_results = []
    for model in models:
        log.info(f"\n{'='*60}")
        log.info(f"  MODEL: {model.upper()}  |  INJECTION: {pct_str}")
        log.info(f"{'='*60}")

        set_shield_model(model)
        time.sleep(2)  # brief pause for container to acknowledge

        run_id = f"{model}_{pct_str}_{ts}"
        try:
            summary = run_pipeline(
                model=model,
                injection_pct=args.injection_pct,
                run_id=run_id,
                shield_active=not args.no_shield,
            )
            all_results.append(summary)
        except Exception as e:
            log.error(f"  FAILED: {model}: {e}")

    # Write combined summary
    out_dir = Path("6_evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / f"all_models_{pct_str}_{ts}.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    log.info(f"\n{'='*60}")
    log.info(f"  SCENARIO SUMMARY — {pct_str} injection")
    log.info(f"{'='*60}")
    log.info(f"  {'Model':<22} {'MCC':>7} {'F1':>7} {'FAR':>7} {'cls_p50':>9}")
    for r in sorted(all_results, key=lambda x: -x["mcc"]):
        log.info(f"  {r['model']:<22} {r['mcc']:>7.4f} {r['f1']:>7.4f} "
                 f"{r['far']:>7.4f} {r['classification_p50']:>9.1f}ms")

    log.info(f"\n  Results saved to: {combined}")


if __name__ == "__main__":
    main()
