from app.dataset import retrieve_relevant_examples


MIN_RECOMMENDATION_SCORE = 0.30


class Recommender:
    def __init__(self):
        print("[RECOMMENDER] Benchmark semantic recommender ready")

    def recommend_by_text(self, text, top_k=3, user_location=None):
        if not text or text in {"No context", "No relevant context"}:
            return []

        results = retrieve_relevant_examples(text, top_k=top_k)
        meaningful_results = [
            result for result in results
            if float(result.get("score", 0.0)) >= MIN_RECOMMENDATION_SCORE
        ]

        recs = []

        for r in meaningful_results:
            recs.append({
                "name": r["intent_category"],
                "description": r["gold_answer_uz"]
            })

        return recs

    def recommend_by_place(self, place_name, top_k=3):
        return self.recommend_by_text(place_name, top_k=top_k)