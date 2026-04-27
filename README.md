# Pilgrimage Assistant  
### Smart Hajj & Umrah Guidance Platform

Pilgrimage Assistant is a multilingual AI-powered web platform designed to provide real-time practical support for Hajj and Umrah pilgrims.  
The system offers guidance for accommodation, transportation, route navigation, Zamzam locations, emergency support, lost-person assistance, and nearby pilgrimage services.

This project combines:

- Intent-aware query routing
- TF-IDF benchmark retrieval
- Semantic multilingual reranking
- FastAPI backend
- React premium frontend UI

---

## Features

- Multilingual support (Uzbek / English / Russian)
- Smart pilgrimage location guidance
- Accommodation and hotel assistance
- Transport and movement recommendations
- Lost person and emergency help
- Zamzam / ritual / landmark lookup
- Recommended nearby service suggestions
- Premium Islamic-themed chat interface

---

## Project Structure

```bash
pilgrimage-assistant/
│
├── backend/          # FastAPI backend and AI retrieval engine
├── frontend/         # React frontend interface
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Local Installation Guide

### 1. Clone the repository

```bash
git clone <your_repo_url>
cd pilgrimage-assistant
```

---

### 2. Create Python virtual environment

```bash
python -m venv .venv
```

Activate virtual environment:

**Windows PowerShell**
```bash
.venv\Scripts\activate
```

---

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Project

### Start Backend

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Backend runs on:

```bash
http://127.0.0.1:8000
```

---

### Start Frontend

Open another terminal:

```bash
cd frontend
npm start
```

Frontend runs on:

```bash
http://localhost:3000
```

---

## Notes

- The local `.venv` virtual environment is intentionally excluded from GitHub.
- `frontend/node_modules` is excluded and should be installed locally with `npm install`.
- Large benchmark duplicates and local caches are ignored for repository optimization.

---

## Research Context

This platform was developed as an experimental multilingual semantic retrieval framework for intelligent Hajj and Umrah guidance.

Potential academic contributions include:

- AI pilgrimage assistant systems
- multilingual semantic retrieval
- intent-routed benchmark QA
- smart religious tourism support

---

## Author

**Abdilatif Meyliyev**