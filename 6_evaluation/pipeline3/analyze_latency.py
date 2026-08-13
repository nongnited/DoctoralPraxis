"""
RAGShield Latency Analysis
Extracts classification + generation latency from all pipeline result files
"""
import re, os, glob

# Point this at wherever your v3 result txt files live
RESULT_DIRS = [
    os.path.expanduser("~/ragshield_eval/s1_10pct/results"),
    os.path.expanduser("~/ragshield_eval/s2_20pct/results"),
    os.path.expanduser("~/ragshield_eval/s3_30pct/results"),
    os.path.expanduser("~/ragshield_eval/s4_50pct/results"),
]

MODEL_ORDER = ["bert_base","securebert","naive_bayes","logistic_regression",
               "xgboost","ridge","svm","lightgbm","random_forest","extra_trees"]

# Pattern matches: "naive_bayes             113   10   96   81  0.919  0.582  0.713     37ms   3.6s"
pattern = re.compile(
    r'(extra_trees|ridge|logistic_regression|random_forest|lightgbm|xgboost|naive_bayes|svm|bert_base|securebert)\s+'
    r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)ms\s+([\d.]+)s'
)

data = {}  # model -> list of (cls_ms, gen_s)

for d in RESULT_DIRS:
    for fpath in glob.glob(f"{d}/*.txt"):
        with open(fpath) as f:
            content = f.read()
        for m in pattern.finditer(content):
            model = m.group(1)
            cls_ms = float(m.group(9))
            gen_s  = float(m.group(10))
            data.setdefault(model, []).append((cls_ms, gen_s))

print("=" * 80)
print("  RAGShield — Latency Analysis (Classification + Generation)")
print("=" * 80)
print(f"  {'Model':<22} {'N runs':>7} {'AvgCls':>10} {'AvgGen':>10} {'AvgTotal':>10}")
print("  " + "─"*70)

results = []
for model in MODEL_ORDER:
    if model not in data:
        continue
    runs = data[model]
    avg_cls = sum(r[0] for r in runs) / len(runs)
    avg_gen = sum(r[1] for r in runs) / len(runs) * 1000  # convert to ms
    avg_total = avg_cls + avg_gen
    results.append((model, len(runs), avg_cls, avg_gen, avg_total))

results.sort(key=lambda x: x[2])  # sort by classification latency

for model, n, cls, gen, total in results:
    print(f"  {model:<22} {n:>7} {cls:>9.1f}ms {gen:>9.0f}ms {total:>9.0f}ms")

print()
print("  Notes:")
print("  AvgCls   = average classification latency per query (ms)")
print("  AvgGen   = average LLM generation latency per query (converted to ms)")
print("  AvgTotal = AvgCls + AvgGen (total added latency per query)")
