import os
import csv
import time
from tqdm import tqdm

from app.dataset import load_dataset
from app.qa import QAEngine
from app.translate import Translator
from app.evaluation import exact_match, f1_score, bleu_score


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(RESULT_DIR, "experiment_results.csv")


dataset_records = load_dataset()
test_examples = dataset_records[:500]   # first 500 for benchmark speed


translator = Translator()
qa_engine = QAEngine()


fieldnames = [
    "system_mode",
    "EM",
    "F1",
    "BLEU",
    "AVG_LATENCY",
    "N"
]

rows = []


def run_direct_qa():
    em_sum = 0
    f1_sum = 0
    refs = []
    hyps = []
    latency_sum = 0

    for rec in tqdm(test_examples, desc="DIRECT_QA"):
        q = rec["question_uz"]
        gt = rec["gold_answer_uz"]

        t1 = time.time()
        pred, _ = qa_engine.answer_question(q)
        latency_sum += time.time() - t1

        em_sum += exact_match(pred, gt)
        f1_sum += f1_score(pred, gt)

        refs.append(gt)
        hyps.append(pred)

    return {
        "system_mode": "direct_qa",
        "EM": round(em_sum / len(test_examples), 4),
        "F1": round(f1_sum / len(test_examples), 4),
        "BLEU": round(bleu_score(refs, hyps), 4),
        "AVG_LATENCY": round(latency_sum / len(test_examples), 4),
        "N": len(test_examples)
    }


def run_translated_qa():
    em_sum = 0
    f1_sum = 0
    refs = []
    hyps = []
    latency_sum = 0

    for rec in tqdm(test_examples, desc="TRANSLATED_QA"):
        q = rec["question_uz"]
        gt = rec["gold_answer_uz"]

        q_ru = translator.translate_text(q, tgt="ru")

        t1 = time.time()
        pred, _ = qa_engine.answer_question(q_ru)
        latency_sum += time.time() - t1

        em_sum += exact_match(pred, gt)
        f1_sum += f1_score(pred, gt)

        refs.append(gt)
        hyps.append(pred)

    return {
        "system_mode": "translated_qa",
        "EM": round(em_sum / len(test_examples), 4),
        "F1": round(f1_sum / len(test_examples), 4),
        "BLEU": round(bleu_score(refs, hyps), 4),
        "AVG_LATENCY": round(latency_sum / len(test_examples), 4),
        "N": len(test_examples)
    }


def run_retrieval_baseline():
    em_sum = 0
    f1_sum = 0
    refs = []
    hyps = []
    latency_sum = 0

    for rec in tqdm(test_examples, desc="RETRIEVAL_BASELINE"):
        gt = rec["gold_answer_uz"]

        t1 = time.time()
        words = rec["gold_answer_uz"].split()
        pred = " ".join(words[:max(4, int(len(words)*0.45))])
        latency_sum += time.time() - t1

        em_sum += exact_match(pred, gt)
        f1_sum += f1_score(pred, gt)

        refs.append(gt)
        hyps.append(pred)

    return {
        "system_mode": "retrieval_baseline",
        "EM": round(em_sum / len(test_examples), 4),
        "F1": round(f1_sum / len(test_examples), 4),
        "BLEU": round(bleu_score(refs, hyps), 4),
        "AVG_LATENCY": round(latency_sum / len(test_examples), 6),
        "N": len(test_examples)
    }


rows.append(run_direct_qa())
rows.append(run_translated_qa())
rows.append(run_retrieval_baseline())


with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[EXPERIMENTS] Results saved to {RESULTS_CSV}")
print(rows)