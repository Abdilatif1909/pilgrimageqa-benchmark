import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.dataset import load_dataset
from app.qa import QAEngine
from app.recommend import Recommender
from app.translate import Translator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

qa_engine: Optional[QAEngine] = None
recommender: Optional[Recommender] = None
translator: Optional[Translator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_engine, recommender, translator

    logger.info("Initializing benchmark modules...")

    qa_engine = QAEngine()
    recommender = Recommender()
    translator = Translator()

    logger.info("All modules initialized successfully")
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title="PilgrimageQA Benchmark API",
    description="Multilingual Benchmark Question Answering API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    user_location: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    detected_language: str
    answer_uz: str
    context: str
    recommendations: List[dict]


class RecommendRequest(BaseModel):
    place_name: Optional[str] = None
    text: Optional[str] = None
    top_k: int = 3
    user_location: Optional[str] = None


class RecommendResponse(BaseModel):
    query: str
    recommendations: List[dict]


class TranslateRequest(BaseModel):
    text: str
    source_lang: Optional[str] = None
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    original: str
    source_lang: str
    target_lang: str
    translated: str


@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "message": "PilgrimageQA Benchmark API running"
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        from langdetect import detect
        try:
            detected_lang = detect(req.question)
        except Exception:
            detected_lang = "uz"

        answer_uz, context = qa_engine.answer_question(req.question)

        recommendations = []
        if getattr(qa_engine, "last_has_relevant_answer", False):
            recommendations = recommender.recommend_by_text(
                context or answer_uz,
                top_k=3
            )

        return AskResponse(
            question=req.question,
            detected_language=detected_lang,
            answer_uz=answer_uz,
            context=context,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"/ask error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    query_text = req.place_name if req.place_name else req.text

    if not query_text:
        raise HTTPException(status_code=400, detail="No query provided")

    recommendations = recommender.recommend_by_text(query_text, top_k=req.top_k)

    return RecommendResponse(
        query=query_text,
        recommendations=recommendations
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    translated = translator.translate_text(
        req.text,
        src=req.source_lang,
        tgt=req.target_lang
    )

    return TranslateResponse(
        original=req.text,
        source_lang=req.source_lang or "auto",
        target_lang=req.target_lang,
        translated=translated
    )


@app.get("/places")
async def list_places():
    data = load_dataset()
    return {
        "count": len(data),
        "sample_questions": [r["question_uz"] for r in data[:20]]
    }


@app.get("/models-info")
async def models_info():
    return {
        "qa_model": "benchmark retrieval qa",
        "retrieval": "TF-IDF semantic benchmark search",
        "recommendation": "semantic nearest benchmark recommendation",
        "translation": "MarianMT/googletrans hybrid",
        "dataset_size": len(load_dataset()),
        "supported_languages": ["uz", "ru", "en"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)