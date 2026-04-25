"""
FastAPI Backend for Multilingual Pilgrimage Tourism QA System

This module provides REST API endpoints for:
- Question answering over pilgrimage dataset
- Place recommendations based on text similarity and location
- Text translation across multiple languages

The system uses transformer-based models (HuggingFace) for QA,
TF-IDF for retrieval and recommendations, and MarianMT/googletrans for translation.
"""

import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import project modules
from app.dataset import PLACES
from app.qa import QAEngine
from app.recommend import Recommender
from app.translate import Translator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances (initialized once at startup)
qa_engine: Optional[QAEngine] = None
recommender: Optional[Recommender] = None
translator: Optional[Translator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup, cleanup on shutdown."""
    global qa_engine, recommender, translator
    
    logger.info("Initializing models...")
    try:
        qa_engine = QAEngine(places=PLACES)
        logger.info("✓ QA Engine initialized")
        
        recommender = Recommender(places=PLACES)
        logger.info("✓ Recommender initialized")
        
        translator = Translator()
        logger.info("✓ Translator initialized")
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")
    # Cleanup if needed
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Pilgrimage Tourism QA System",
    description="Multilingual NLP Question Answering and Recommendation System for Pilgrimage Sites in Uzbekistan",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic request/response models
class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    question: str = Field(..., min_length=1, max_length=500, description="User question in Uzbek, Russian or English")
    user_location: Optional[str] = Field(None, description="User location as 'lat,lon' for location-aware recommendations")


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    question: str
    detected_language: str
    answer_uz: str
    context: str
    recommendations: List[dict]


class RecommendRequest(BaseModel):
    """Request model for /recommend endpoint"""
    place_name: Optional[str] = Field(None, description="Name of place to find similar places")
    text: Optional[str] = Field(None, description="Query text to find recommendations")
    top_k: int = Field(3, ge=1, le=10, description="Number of recommendations to return")
    user_location: Optional[str] = Field(None, description="User location as 'lat,lon'")


class RecommendResponse(BaseModel):
    """Response model for /recommend endpoint"""
    query: str
    recommendations: List[dict]


class TranslateRequest(BaseModel):
    """Request model for /translate endpoint"""
    text: str = Field(..., min_length=1, max_length=1000)
    source_lang: Optional[str] = Field(None, description="Source language code (auto-detect if None)")
    target_lang: str = Field("en", description="Target language code")


class TranslateResponse(BaseModel):
    """Response model for /translate endpoint"""
    original: str
    source_lang: str
    target_lang: str
    translated: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: bool
    message: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify server and models are running."""
    models_ok = (qa_engine is not None and 
                 recommender is not None and 
                 translator is not None)
    
    return HealthResponse(
        status="ok" if models_ok else "degraded",
        models_loaded=models_ok,
        message="All systems operational" if models_ok else "Some models failed to load"
    )


@app.post("/ask", response_model=AskResponse, tags=["QA"])
async def ask(req: AskRequest):
    """
    Full QA pipeline endpoint.
    
    1. Detects language of user question
    2. Translates to English if needed
    3. Runs QA model to extract answer
    4. Generates recommendations based on answer context
    5. Translates answer back to Uzbek
    
    Args:
        question: User's question in any supported language
        user_location: Optional GPS coordinates for location-aware recommendations
    
    Returns:
        Question, detected language, answer in Uzbek, source context, and recommendations
    """
    if qa_engine is None or recommender is None or translator is None:
        logger.error("Models not initialized")
        raise HTTPException(status_code=503, detail="Models not loaded. Try again later.")
    
    try:
        # Detect language
        from langdetect import detect
        try:
            detected_lang = detect(req.question)
        except Exception:
            detected_lang = "en"
        
        logger.info(f"Detected language: {detected_lang}")
        
        # Translate question to English for QA if needed
        question_en = req.question
        if detected_lang != "en":
            question_en = translator.translate_text(
                req.question, 
                src=detected_lang, 
                tgt="en"
            )
            logger.info(f"Translated question: {question_en}")
        
        # Run QA
        answer_en, context = qa_engine.answer_question(question_en)
        logger.info(f"QA output: {answer_en[:100]}...")
        
        # Get recommendations based on context
        recommendations = recommender.recommend_by_text(
            context or answer_en,
            top_k=3,
            user_location=req.user_location
        )
        logger.info(f"Got {len(recommendations)} recommendations")
        
        # Translate answer to Uzbek
        answer_uz = translator.translate_text(answer_en, src="en", tgt="uz")
        
        return AskResponse(
            question=req.question,
            detected_language=detected_lang,
            answer_uz=answer_uz,
            context=context[:200] if context else "",
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(req: RecommendRequest):
    """
    Get recommendations for similar places.
    
    Can search by:
    1. place_name: Find places similar to a specific place
    2. text: Find places similar to a query text
    
    Location-aware scoring can be enabled via user_location.
    
    Args:
        place_name: Name of reference place (optional)
        text: Query text for semantic search (optional)
        top_k: Number of recommendations (1-10, default 3)
        user_location: GPS coordinates as 'lat,lon' for distance weighting
    
    Returns:
        List of recommended places with similarity scores and optional distances
    """
    if recommender is None:
        logger.error("Recommender not initialized")
        raise HTTPException(status_code=503, detail="Recommender not loaded.")
    
    try:
        recommendations = None
        query_label = ""
        
        # Route to appropriate recommendation method
        if req.place_name and req.place_name.strip():
            query_label = f"place: {req.place_name}"
            recommendations = recommender.recommend_by_place(
                req.place_name,
                top_k=req.top_k
            )
            
            if recommendations is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Place '{req.place_name}' not found in dataset"
                )
        
        elif req.text and req.text.strip():
            query_label = f"text: {req.text[:50]}"
            recommendations = recommender.recommend_by_text(
                req.text,
                top_k=req.top_k,
                user_location=req.user_location
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'place_name' or 'text' parameter"
            )
        
        logger.info(f"Recommendations for {query_label}: {len(recommendations)} results")
        
        return RecommendResponse(
            query=query_label,
            recommendations=recommendations or []
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /recommend endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate(req: TranslateRequest):
    """
    Translate text between languages.
    
    Uses MarianMT (HuggingFace) with googletrans fallback.
    Supports auto-detection of source language.
    
    Args:
        text: Text to translate
        source_lang: Source language code (optional, auto-detect if None)
        target_lang: Target language code (default: 'en')
    
    Returns:
        Original text, detected source language, target language, and translated text
    """
    if translator is None:
        logger.error("Translator not initialized")
        raise HTTPException(status_code=503, detail="Translator not loaded.")
    
    try:
        # Auto-detect source language if not provided
        source_lang = req.source_lang
        if not source_lang:
            from langdetect import detect
            try:
                source_lang = detect(req.text)
            except Exception:
                source_lang = "en"
        
        # Translate
        translated_text = translator.translate_text(
            req.text,
            src=source_lang,
            tgt=req.target_lang
        )
        
        logger.info(f"Translated from {source_lang} to {req.target_lang}: {req.text[:50]}... -> {translated_text[:50]}...")
        
        return TranslateResponse(
            original=req.text,
            source_lang=source_lang,
            target_lang=req.target_lang,
            translated=translated_text
        )
    
    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ============================================================================
# Additional utility endpoints
# ============================================================================

@app.get("/places", tags=["Data"])
async def list_places():
    """Get all available places in the dataset."""
    try:
        return {
            "count": len(PLACES),
            "places": [
                {
                    "name": p.get("name"),
                    "category": p.get("category"),
                    "location": p.get("location")
                }
                for p in PLACES
            ]
        }
    except Exception as e:
        logger.error(f"Error in /places endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models-info", tags=["Info"])
async def models_info():
    """Get information about loaded models."""
    return {
        "qa_model": "mrm8488/bert-multilingual-cased-finetuned-xquadv1 (with distilbert fallback)",
        "retrieval": "TF-IDF vectorizer",
        "recommendation": "TF-IDF similarity + optional haversine distance weighting",
        "translation": "MarianMT (Helsinki-NLP/opus-mt-*) with googletrans fallback",
        "language_detection": "langdetect",
        "dataset_size": len(PLACES),
        "supported_languages": ["en", "uz", "ru"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
