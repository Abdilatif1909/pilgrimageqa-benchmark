import json
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "scripts" / "data" / "pilgrimage_dataset.json"


class PilgrimageDataset:
    def __init__(self):
        self.records: List[Dict] = []
        self.vectorizer = TfidfVectorizer()
        self.question_matrix = None
        self._load_dataset()
        self._build_index()

    def _load_dataset(self):
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        print(f"[DATASET] Loaded {len(self.records)} benchmark records")

    def _build_index(self):
        corpus = []

        for rec in self.records:
            corpus.append(rec["question_uz"])
            corpus.append(rec["question_ru"])

        self.question_matrix = self.vectorizer.fit_transform(corpus)
        print("[DATASET] TF-IDF retrieval index built")

    def get_all_records(self):
        return self.records

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.question_matrix).flatten()

        scored = []

        for i, score in enumerate(sims):
            rec_index = i // 2
            scored.append((score, self.records[rec_index]))

        scored.sort(key=lambda x: x[0], reverse=True)

        unique_results = []
        seen_ids = set()

        for score, rec in scored:
            if rec["id"] not in seen_ids:
                unique_results.append({
                    "score": float(score),
                    "id": rec["id"],
                    "intent_category": rec["intent_category"],
                    "question_uz": rec["question_uz"],
                    "question_ru": rec["question_ru"],
                    "gold_answer_uz": rec["gold_answer_uz"],
                    "gold_answer_ru": rec["gold_answer_ru"],
                    "context_uz": rec["context_uz"],
                    "context_ru": rec["context_ru"],
                    "difficulty": rec["difficulty"]
                })
                seen_ids.add(rec["id"])

            if len(unique_results) >= top_k:
                break

        return unique_results


dataset_instance = PilgrimageDataset()


def load_dataset():
    return dataset_instance.get_all_records()


def split_dataset():
    data = dataset_instance.get_all_records()
    n = len(data)
    train_end = int(n * 0.70)
    dev_end = train_end + int(n * 0.15)
    return data[:train_end], data[train_end:dev_end], data[dev_end:]


def generate_synthetic_qa_examples():
    examples = []
    for rec in dataset_instance.get_all_records()[:100]:
        examples.append((rec["question_uz"], rec["gold_answer_uz"], rec["context_uz"]))
    return examples


def retrieve_relevant_examples(query: str, top_k: int = 5):
    return dataset_instance.search(query, top_k)