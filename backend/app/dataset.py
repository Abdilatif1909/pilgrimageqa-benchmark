import json
import re
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "scripts" / "data" / "pilgrimage_dataset.json"
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_TFIDF_SCORE = 0.02
SEMANTIC_CANDIDATE_POOL = 20
SEMANTIC_MODEL = None

INTENT_CATEGORY_MAP = {
    "accommodation": {"accommodation_guidance", "cost_information", "nearest_entity_search"},
    "lost_person": {"lost_person_help", "emergency_security_support"},
    "transport": {"transport_advice", "route_navigation", "crowd_movement_support", "nearest_entity_search"},
    "food": {"food_service_guidance", "nearest_entity_search"},
    "medical_help": {"emergency_medical_support", "nearest_entity_search"},
    "security_help": {"emergency_security_support", "lost_person_help"},
    "place_lookup": {"factual_place_lookup", "nearest_entity_search", "route_navigation"},
    "zamzam": {"factual_place_lookup", "nearest_entity_search", "ritual_instruction", "prayer_support"},
    "tawaf": {"ritual_instruction", "prayer_support", "crowd_movement_support"}
}

INTENT_PATTERNS = {
    "accommodation": [r"\btunash\b", r"mehmonxona", r"hotel", r"гостин", r"ночлег", r"hostel", r"room"],
    "lost_person": [
        r"adash",
        r"adashdim",
        r"adashib qoldim",
        r"yo'qoldim",
        r"yo'qol",
        r"yoqol",
        r"topolmayapman",
        r"oilamni topolmayapman",
        r"bolam yo'qoldi",
        r"odamimni yo'qotdim",
        r"lost my family",
        r"cannot find my group",
        r"lost",
        r"missing",
        r"потерялся",
        r"не могу найти семью",
        r"потерял",
        r"пропал",
        r"заблуд"
    ],
    "transport": [r"transport", r"avtobus", r"bus", r"train", r"taxi", r"metro", r"маршрут", r"автобус", r"поезд", r"taksi", r"yo'l", r"qanday bor"],
    "food": [r"ovqat", r"taom", r"food", r"eat", r"restaurant", r"restoran", r"cafe", r"кафе", r"еда", r"ресторан"],
    "medical_help": [r"doctor", r"hospital", r"clinic", r"dorixona", r"apteka", r"medical", r"kasal", r"врач", r"больниц", r"аптек"],
    "security_help": [r"security", r"xavfsiz", r"police", r"guard", r"охран", r"полици", r"безопас"],
    "place_lookup": [r"qayerda", r"where", r"где", r"near", r"yaqin", r"locat", r"manzil", r"joylash"],
    "zamzam": [r"zamzam", r"zam-zam"],
    "tawaf": [r"tawaf", r"tavof", r"ibodat", r"ritual", r"namoz", r"dua", r"safa", r"marwa", r"marvo", r"kaaba", r"ka'ba", r"rawdah", r"sa'i", r"сай", r"таваф", r"молитв"]
}

ANCHOR_PATTERNS = {
    "Makka": [r"makka", r"makkah", r"mecca"],
    "Madina": [r"madina", r"madinah", r"medina"],
    "Mina": [r"\bmina\b"],
    "Arafat": [r"arafat", r"arafah"],
    "Muzdalifa": [r"muzdalifa", r"muzdalifah", r"muzdalifa"],
    "Ajyad": [r"ajyad"],
    "Zamzam": [r"zamzam", r"zam-zam"],
    "Haram": [r"haram", r"masjid al-haram"],
    "Nabawi": [r"nabawi", r"an-nabawi", r"masjid an-nabawi"]
}

GREETING_PATTERNS = [
    r"^\s*(salom|assalomu alaykum|assalomu aleykum|hello|hi|привет|здравствуй|здравствуйте)\s*[!.?]*\s*$",
    r"^\s*(rahmat|thanks|thank you|спасибо)\s*[!.?]*\s*$"
]

UNSUPPORTED_PATTERNS = [
    r"who are you",
    r"what can you do",
    r"how are you",
    r"sen kimsan",
    r"nima qila olasan",
    r"qalaysan",
    r"кто ты",
    r"что ты умеешь",
    r"как дела"
]


def _matches_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def analyze_query(query: str) -> Dict:
    normalized_query = query.lower().strip()
    matched_intents = [
        intent for intent, patterns in INTENT_PATTERNS.items()
        if _matches_any_pattern(normalized_query, patterns)
    ]
    matched_anchors = [
        anchor for anchor, patterns in ANCHOR_PATTERNS.items()
        if _matches_any_pattern(normalized_query, patterns)
    ]

    is_greeting = _matches_any_pattern(normalized_query, GREETING_PATTERNS)
    is_unsupported = _matches_any_pattern(normalized_query, UNSUPPORTED_PATTERNS)

    return {
        "query": query,
        "matched_intents": matched_intents,
        "matched_anchors": matched_anchors,
        "intent_categories": sorted({
            category
            for intent in matched_intents
            for category in INTENT_CATEGORY_MAP.get(intent, set())
        }),
        "is_greeting": is_greeting,
        "is_unsupported": is_unsupported and not matched_intents and not matched_anchors
    }


class PilgrimageDataset:
    def __init__(self):
        self.records: List[Dict] = []
        self.vectorizer = TfidfVectorizer()
        self.question_matrix = None
        self.search_texts: List[str] = []
        self.semantic_texts: List[str] = []
        self.filter_texts: List[str] = []
        self._load_dataset()
        self._build_index()
        self._load_semantic_model()

    def _load_dataset(self):
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        print(f"[DATASET] Loaded {len(self.records)} benchmark records")

    def _build_index(self):
        self.search_texts = []
        self.semantic_texts = []
        self.filter_texts = []
        for rec in self.records:
            self.search_texts.append(
                " ".join(
                    part for part in [
                        rec.get("question_uz", ""),
                        rec.get("question_ru", "")
                    ]
                    if part
                )
            )
            self.semantic_texts.append(
                " ".join(
                    part for part in [
                        rec.get("question_uz", ""),
                        rec.get("question_ru", ""),
                        rec.get("intent_category", ""),
                        rec.get("domain_entity", ""),
                        rec.get("context_uz", ""),
                        rec.get("context_ru", "")
                    ]
                    if part
                )
            )
            self.filter_texts.append(
                " ".join(
                    part for part in [
                        rec.get("domain_entity", ""),
                        rec.get("question_uz", ""),
                        rec.get("question_ru", ""),
                        rec.get("context_uz", ""),
                        rec.get("context_ru", "")
                    ]
                    if part
                ).lower()
            )

        self.question_matrix = self.vectorizer.fit_transform(self.search_texts)
        print("[DATASET] TF-IDF retrieval index built")

    def _load_semantic_model(self):
        global SEMANTIC_MODEL

        if not HAS_SENTENCE_TRANSFORMERS:
            return None

        if SEMANTIC_MODEL is None:
            print("[SEMANTIC] Loading multilingual embedding model...")
            SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)

        return SEMANTIC_MODEL

    def get_all_records(self):
        return self.records

    def _filter_record_indices(self, route: Dict) -> List[int]:
        all_indices = list(range(len(self.records)))

        if route.get("is_greeting") or route.get("is_unsupported"):
            return []

        intent_indices = []
        if route.get("intent_categories"):
            allowed_categories = set(route["intent_categories"])
            intent_indices = [
                idx for idx, rec in enumerate(self.records)
                if rec.get("intent_category") in allowed_categories
            ]

        anchor_indices = []
        if route.get("matched_anchors"):
            anchor_terms = {anchor.lower() for anchor in route["matched_anchors"]}
            if "makka" in anchor_terms:
                anchor_terms.update({"makkah", "mecca", "masjid al-haram", "haram"})
            if "madina" in anchor_terms:
                anchor_terms.update({"madinah", "medina", "nabawi", "masjid an-nabawi"})
            if "haram" in anchor_terms:
                anchor_terms.update({"masjid al-haram", "haram bus station", "haram train"})
            if "nabawi" in anchor_terms:
                anchor_terms.update({"masjid an-nabawi", "rawdah"})

            anchor_indices = [
                idx for idx, text in enumerate(self.filter_texts)
                if any(anchor_term in text for anchor_term in anchor_terms)
            ]

        if intent_indices and anchor_indices:
            intersection = sorted(set(intent_indices).intersection(anchor_indices))
            if intersection:
                return intersection
            return sorted(set(intent_indices).union(anchor_indices))

        if intent_indices:
            return intent_indices

        if anchor_indices:
            return anchor_indices

        return all_indices

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            return []

        route = analyze_query(query)
        candidate_indices = self._filter_record_indices(route)

        if not candidate_indices:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.question_matrix[candidate_indices]).flatten()

        ranked_positions = sims.argsort()[::-1]
        candidate_count = max(top_k * 5, SEMANTIC_CANDIDATE_POOL)
        candidates = []

        for position in ranked_positions[:candidate_count]:
            score = float(sims[position])
            if score < MIN_TFIDF_SCORE:
                break

            idx = candidate_indices[position]

            rec = self.records[idx]
            candidates.append({
                "score": score,
                "tfidf_score": score,
                "semantic_score": score,
                "id": rec["id"],
                "intent_category": rec["intent_category"],
                "question_uz": rec["question_uz"],
                "question_ru": rec["question_ru"],
                "gold_answer_uz": rec["gold_answer_uz"],
                "gold_answer_ru": rec["gold_answer_ru"],
                "context_uz": rec["context_uz"],
                "context_ru": rec["context_ru"],
                "difficulty": rec["difficulty"],
                "_semantic_text": self.semantic_texts[idx]
            })

        if not candidates:
            return []

        semantic_model = None
        try:
            semantic_model = self._load_semantic_model()
        except Exception:
            semantic_model = None

        if semantic_model is not None:
            query_embedding = semantic_model.encode([query], normalize_embeddings=True)
            candidate_embeddings = semantic_model.encode(
                [candidate["_semantic_text"] for candidate in candidates],
                normalize_embeddings=True,
                show_progress_bar=False
            )
            semantic_scores = cosine_similarity(query_embedding, candidate_embeddings).flatten()

            for candidate, semantic_score in zip(candidates, semantic_scores):
                semantic_score = float(semantic_score)
                candidate["semantic_score"] = semantic_score
                candidate["score"] = (candidate["tfidf_score"] * 0.35) + (semantic_score * 0.65)

        candidates.sort(
            key=lambda item: (item["score"], item["semantic_score"], item["tfidf_score"]),
            reverse=True
        )

        for candidate in candidates:
            candidate.pop("_semantic_text", None)

        return candidates[:top_k]


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