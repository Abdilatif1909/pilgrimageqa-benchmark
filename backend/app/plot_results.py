import os
import csv
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(ROOT, "results")

CSV_PATH = os.path.join(RESULT_DIR, "experiment_results.csv")
METRIC_PLOT = os.path.join(RESULT_DIR, "qa_metrics_plot.png")
LATENCY_PLOT = os.path.join(RESULT_DIR, "latency_plot.png")

rows = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

systems = [r["system_mode"] for r in rows]
em_scores = [float(r["EM"]) for r in rows]
f1_scores = [float(r["F1"]) for r in rows]
bleu_scores = [float(r["BLEU"]) for r in rows]
latencies = [float(r["AVG_LATENCY"]) for r in rows]

# -----------------------
# QA Metrics Plot
# -----------------------
x = range(len(systems))

plt.figure(figsize=(10, 6))
plt.bar(x, em_scores, width=0.2, label="Exact Match")
plt.bar([i + 0.2 for i in x], f1_scores, width=0.2, label="F1 Score")
plt.bar([i + 0.4 for i in x], [b / 100 for b in bleu_scores], width=0.2, label="BLEU/100")

plt.xticks([i + 0.2 for i in x], systems, rotation=15)
plt.ylabel("Score")
plt.title("Comparative QA Performance Metrics")
plt.legend()
plt.tight_layout()
plt.savefig(METRIC_PLOT)
plt.close()

# -----------------------
# Latency Plot
# -----------------------
plt.figure(figsize=(8, 5))
plt.bar(systems, latencies)
plt.ylabel("Seconds")
plt.title("Average Response Latency")
plt.tight_layout()
plt.savefig(LATENCY_PLOT)
plt.close()

print(f"[PLOT] Saved metric chart: {METRIC_PLOT}")
print(f"[PLOT] Saved latency chart: {LATENCY_PLOT}")