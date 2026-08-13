"""
Full chunk sensitivity evaluation
10 models × 3 chunk sizes × 4 scenarios = 120 runs
Results saved to pipeline3/{chunk_size}/{scenario}/results/
"""
import subprocess, time, os, sys
from datetime import datetime

BASE = "/Users/thadnongni/Academics/GW/Praxis/pipeline3"

CHUNK_SIZES = [50, 100, 150]

SCENARIOS = [
    "s1_10pct",
    "s2_20pct",
    "s3_30pct",
    "s4_50pct",
]

def set_chunk_size(size):
    print(f"\n  Setting chunk_size={size} on vault...")
    cmd = (
        f'ssh tnongni@10.0.0.107 "sed -i '
        f"\'s/chunk_size: int = [0-9]*/chunk_size: int = {size}/\'"
        f' /opt/ragshield/vault/backend/main.py && '
        f'sudo docker compose -f /opt/ragshield/vault/docker-compose.yml restart fastapi"'
    )
    subprocess.run(cmd, shell=True, capture_output=True)
    time.sleep(25)
    print(f"  Vault restarted with chunk_size={size}")

total_runs = len(CHUNK_SIZES) * len(SCENARIOS)
completed  = 0
start_time = datetime.now()

for chunk_size in CHUNK_SIZES:
    print(f"\n{'='*65}")
    print(f"  CHUNK SIZE: {chunk_size} words")
    print(f"{'='*65}")

    set_chunk_size(chunk_size)

    for scenario in SCENARIOS:
        completed += 1
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        print(f"\n[{completed}/{total_runs}] chunk={chunk_size} "
              f"scenario={scenario} elapsed={elapsed:.1f}h")

        runner  = f"{BASE}/chunk{chunk_size}/{scenario}/run_all_models.py"
        results = f"{BASE}/chunk{chunk_size}/{scenario}/results"
        os.makedirs(results, exist_ok=True)

        outfile = f"{results}/run_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

        with open(outfile, 'w') as f:
            proc = subprocess.Popen(
                ["python3", "-u", runner],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            proc.wait()

        print(f"  Saved: {outfile}")

# Restore chunk_size=100
set_chunk_size(100)

total_hours = (datetime.now() - start_time).total_seconds() / 3600
print(f"\n{'='*65}")
print(f"  ALL RUNS COMPLETE")
print(f"  Total time: {total_hours:.1f} hours")
print(f"  chunk_size restored to 100")
print(f"  Results in: {BASE}")
print(f"{'='*65}")
