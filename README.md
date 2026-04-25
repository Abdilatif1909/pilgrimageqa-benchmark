Pilgrimage Assistant — Multilingual Pilgrimage Tourism AI

Overview

This project is a full-stack prototype of a multilingual pilgrimage tourism assistant for Uzbekistan. It contains a Python/FastAPI backend with a transformer-based QA engine and a simple React frontend chat UI.

Repository structure

- backend/app: FastAPI application and modules (QA engine, recommender, translator, dataset)
- frontend: React chat interface
- requirements.txt: Python dependencies for the backend

Features

- Ask questions in Uzbek, Russian or English
- Answers are returned in Uzbek
- Transformer (HuggingFace) QA pipeline over a small sample dataset of pilgrimage places
- Language detection (langdetect) and translation (MarianMT / googletrans fallback)
- Simple recommendation engine based on text similarity and optional user location
- React frontend with chat interface and recommendation cards

Backend setup (Python)

1. Create a virtual environment and activate it:

   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Start the FastAPI server:

   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Notes:
- The first run may download transformer models and sentencepiece; this can take time and requires sufficient disk and memory.
- If MarianMT models for certain language pairs are not available, the translator falls back to googletrans (internet required) or returns the original text.

Frontend setup (React)

1. From the frontend folder, install node dependencies:

   cd frontend
   npm install

2. Start the development server:

   npm start

By default the frontend expects the backend at http://localhost:8000. CORS is enabled for http://localhost:3000 in the backend for development.

API endpoints

- POST /ask
  - body: { "question": "...", "user_location": "lat,lon" }
  - returns: answer in Uzbek, detected language, source context and recommendations

- POST /recommend
  - body: { "place_name": "Bukhara", "top_k": 3 }
  - returns: similar places

- POST /translate
  - body: { "text": "...", "target_lang": "uz" }
  - returns: translated_text

Extending and notes

- The QAEngine uses a TF-IDF retrieval over the small dataset and a HuggingFace QA pipeline for extraction; you can replace the retrieval with a stronger dense retriever or expand the dataset.
- To support production deployment: persist models, add caching, rate limits, and secure API keys for paid translation services.

License

This prototype is provided as-is for demonstration and educational purposes. Add a license file if you plan to publish it publicly.
