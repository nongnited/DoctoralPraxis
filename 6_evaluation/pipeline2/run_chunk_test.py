"""
Chunk size sensitivity test
Runs naive_bayes and bert_base at chunk sizes 50, 100, 150
Uses s4_50pct queries (50 targeted, 50 benign)
"""
import subprocess, time, os, re, sys
from datetime import datetime

BASE     = "/Users/thadnongni/Academics/GW/Praxis/pipeline2"
PIPELINE = f"{BASE}/pipeline_chunk_test.py"

MODELS = [
    ("naive_bayes", 8107),
    ("bert_base",   8109),
]

CHUNK_SIZES = [50, 100, 150]

def set_chunk_size(size):
    print(f"\n  Setting chunk_size={size} on vault...")
    cmd = (
        f'ssh tnongni@10.0.0.107 "sudo sed -i '
        f"'s/chunk_size: int = [0-9]*/chunk_size: int = {size}/'"
        f' /opt/ragshield/vault/backend/main.py && '
        f'cd /opt/ragshield/vault && sudo docker compose restart fastapi"'
    )
    subprocess.run(cmd, shell=True, capture_output=True)
    time.sleep(25)
    print(f"  Vault restarted with chunk_size={size}")

def switch_model(model, port):
    print(f"  Switching vault to {model}...")
    cmd = (
        f'ssh tnongni@10.0.0.107 "cd /opt/ragshield/vault && '
        f'sudo sed -i \'s/SHIELD_MODEL=.*/SHIELD_MODEL={model}/\' docker-compose.yml && '
        f'sudo sed -i \'s/SHIELD_PORT=.*/SHIELD_PORT={port}/\' docker-compose.yml && '
        f'sudo docker compose up -d --no-deps fastapi"'
    )
    subprocess.run(cmd, shell=True, capture_output=True)
    time.sleep(20)

for chunk_size in CHUNK_SIZES:
    print(f"\n{'='*60}")
    print(f"  CHUNK SIZE: {chunk_size} words")
    print(f"{'='*60}")

    set_chunk_size(chunk_size)

    for model_name, port in MODELS:
        switch_model(model_name, port)

        outfile = f"{BASE}/chunk{chunk_size}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        print(f"\n  Running {model_name} chunk={chunk_size}...")
        print(f"  Output: {outfile}")

        with open(outfile, 'w') as f:
            proc = subprocess.Popen(
                ["python3", "-u", PIPELINE],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            proc.wait()

        print(f"  Completed {model_name} chunk={chunk_size}")

# Restore chunk_size=100 when done
set_chunk_size(100)
print(f"\nDone. chunk_size restored to 100.")
print(f"Results in: {BASE}")
