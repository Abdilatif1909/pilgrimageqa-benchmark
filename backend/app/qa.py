from typing import Tuple
from app.dataset import analyze_query, retrieve_relevant_examples


MIN_QA_CONFIDENCE_SCORE = 0.30
MIN_LOST_PERSON_CONFIDENCE_SCORE = 0.22


class QAEngine:
    def __init__(self):
        self.last_confidence = 0.0
        self.last_result = None
        self.last_has_relevant_answer = False
        print("[QA] Running in benchmark retrieval QA mode")

    def answer_question(self, question: str) -> Tuple[str, str]:
        route = analyze_query(question)

        if route.get("is_greeting") or route.get("is_unsupported"):
            self.last_confidence = 0.0
            self.last_result = None
            self.last_has_relevant_answer = False
            return (
                "Assalomu alaykum. Men ziyorat bo'yicha yordam bera olaman. Iltimos, mehmonxona, yo'nalish, zamzam, transport yoki boshqa amaliy savol bering.",
                "No relevant context"
            )

        results = retrieve_relevant_examples(question, top_k=3)

        if not results:
            self.last_confidence = 0.0
            self.last_result = None
            self.last_has_relevant_answer = False
            return (
                "Bu savol bo'yicha benchmark bazada aniq javob topilmadi.",
                "No context"
            )

        best = results[0]
        best_score = float(best.get("score", 0.0))
        required_score = (
            MIN_LOST_PERSON_CONFIDENCE_SCORE
            if "lost_person" in route.get("matched_intents", [])
            else MIN_QA_CONFIDENCE_SCORE
        )

        self.last_confidence = best_score
        self.last_result = best
        self.last_has_relevant_answer = best_score >= required_score

        if not self.last_has_relevant_answer:
            return (
                "Bu savol bo'yicha benchmark bazada mos va ishonchli javob topilmadi.",
                "No relevant context"
            )

        return (
            best["gold_answer_uz"],
            best["context_uz"]
        )