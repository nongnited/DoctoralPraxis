"""
RAGShield — Full 10-Model Pipeline Evaluation
Switches vault model, runs 100 queries per model, saves results
"""
import subprocess, time, datetime, json, urllib.request, os

VAULT_USER = "tnongni"
VAULT_IP   = "10.0.0.107"
RESULTS    = f"/Users/thadnongni/Academics/GW/Praxis/pipeline3/chunk150/s4_50pct/results/ragshield_chunk150_s4_50pct_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"

MODELS = [
    ("extra_trees",         8101),
    ("ridge",               8102),
    ("logistic_regression", 8103),
    ("random_forest",       8104),
    ("lightgbm",            8105),
    ("xgboost",             8106),
    ("naive_bayes",         8107),
    ("svm",                 8108),
    ("bert_base",           8109),
    ("securebert",          8110),
]

def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

def switch_model(model_name, port):
    """Switch vault to specified shield model via SSH"""
    cmd = (
        f"cd /opt/ragshield/vault && "
        f"sudo sed -i 's/SHIELD_MODEL=.*/SHIELD_MODEL={model_name}/' docker-compose.yml && "
        f"sudo sed -i 's/SHIELD_PORT=.*/SHIELD_PORT={port}/' docker-compose.yml && "
        f"sudo docker compose up -d --no-deps fastapi > /dev/null 2>&1"
    )
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-o", "BatchMode=yes",
         f"{VAULT_USER}@{VAULT_IP}", cmd],
        capture_output=True, text=True, timeout=60
    )
    return result.returncode == 0

def verify_model(expected_model):
    """Check vault health endpoint confirms model switch"""
    try:
        r = urllib.request.urlopen(
            f"http://{VAULT_IP}:8080/health", timeout=10)
        d = json.loads(r.read())
        return d.get("shield_model", "unknown")
    except:
        return "unknown"

def run_evaluation():
    """Run pipeline_100q.py and capture output"""
    result = subprocess.run(
        ["python3", "/Users/thadnongni/Academics/GW/Praxis/pipeline3/chunk150/s4_50pct/pipeline.py"],
        capture_output=True, text=True, timeout=7200
    )
    return result.stdout + result.stderr

def main():
    with open(RESULTS, "w") as f:
        log("=" * 70, f)
        log("  RAGShield — 100-Query Full Pipeline Evaluation", f)
        log(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", f)
        log(f"  Results: {RESULTS}", f)
        log("=" * 70, f)

        summary = []

        for model_name, port in MODELS:
            log(f"\n{'#'*70}", f)
            log(f"  MODEL: {model_name}  PORT: {port}", f)
            log(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", f)
            log(f"{'#'*70}", f)

            # Switch model
            log(f"  Switching vault to {model_name}...", f)
            ok = switch_model(model_name, port)
            if not ok:
                log(f"  ERROR: SSH command failed for {model_name}", f)
                log("  Check that SSH key auth is set up and sudo is passwordless", f)
                continue

            # Wait and verify
            time.sleep(25)
            active = verify_model(model_name)
            log(f"  Vault confirmed: {active}", f)

            if active != model_name:
                log(f"  WARNING: expected {model_name} got {active} — waiting 30s", f)
                time.sleep(30)
                active = verify_model(model_name)
                log(f"  Vault re-check: {active}", f)

            # Run evaluation
            log(f"  Running 100 queries for {model_name}...", f)
            output = run_evaluation()
            log(output, f)

            # Extract summary line
            for line in output.split("\n"):
                if "TP=" in line and "F1=" in line:
                    summary.append(f"  {model_name:<22} {line.strip()}")

            log(f"  Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", f)

        # Final summary
        log(f"\n{'='*70}", f)
        log("  FINAL SUMMARY", f)
        log(f"{'='*70}", f)
        for line in summary:
            log(line, f)
        log(f"\n  Results saved to: {RESULTS}", f)
        log(f"  Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", f)
        log("=" * 70, f)

    print(f"\nResults saved to: {RESULTS}")

if __name__ == "__main__":
    main()
