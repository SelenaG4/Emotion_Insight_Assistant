"""Emotion Insight Assistant.

A new, standalone app (not layered onto the audio apps): a facial-emotion CNN --
a faithful port of "Model 3" from the completed MIT capstone, 77.3% test accuracy --
paired with a RAG/GenAI layer that turns raw per-frame predictions into grounded,
client-facing natural-language answers and reports.

Run locally:
    uvicorn app.main:app --reload

Then:
    python scripts/load_sample.py
    curl -X POST localhost:8000/ask -H "Content-Type: application/json" \\
        -d '{"question": "Was the focus group mostly positive this morning?"}'
    curl -X POST localhost:8000/predict -F "file=@some_face.jpg"
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from app.emotion_model import EmotionClassifier
from app.llm import llm_client
from app.models import (
    AskRequest,
    AskResponse,
    DailyReportResponse,
    IngestRequest,
    IngestResponse,
    PredictResponse,
)
from app.retrieval import TfidfRetriever, events_to_documents, load_knowledge_base
from app.store import event_store

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "model3.weights.h5"

app = FastAPI(
    title="Emotion Insight Assistant",
    description="Facial-emotion CNN (capstone Model 3) + RAG layer over its detections.",
    version="0.1.0",
)

classifier = EmotionClassifier(weights_path=WEIGHTS_PATH if WEIGHTS_PATH.exists() else None)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_mode": llm_client.mode,
        "events_stored": len(event_store),
        "cnn_weights_loaded": classifier.weights_loaded,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("L")  # grayscale
    array = np.array(image)
    label, confidence, per_class = classifier.predict(array)
    return PredictResponse(
        label=label, confidence=confidence, per_class=per_class, weights_loaded=classifier.weights_loaded
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    stored = event_store.add(payload.events)
    return IngestResponse(stored=stored, total_events=len(event_store))


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    start = time.perf_counter()

    kb_docs = load_knowledge_base(DATA_DIR / "class_notes.md")
    event_docs = events_to_documents(event_store.for_session(payload.session_id))
    retriever = TfidfRetriever(kb_docs + event_docs)
    chunks = retriever.query(payload.question, top_k=payload.top_k)

    answer = llm_client.generate(payload.question, chunks)
    latency_ms = (time.perf_counter() - start) * 1000

    return AskResponse(answer=answer, mode=llm_client.mode, retrieved=chunks, latency_ms=latency_ms)


@app.get("/report/daily", response_model=DailyReportResponse)
def daily_report(session_id: str | None = None) -> DailyReportResponse:
    start = time.perf_counter()
    events = event_store.for_session(session_id)

    question = (
        "Summarize today's facial-emotion detections in plain language for a "
        "non-technical client, grouped by time of day, framed as model observations "
        "rather than certain facts, and flag anything with unusually low confidence."
    )
    kb_docs = load_knowledge_base(DATA_DIR / "class_notes.md")
    event_docs = events_to_documents(events)
    retriever = TfidfRetriever(kb_docs + event_docs)
    chunks = retriever.query(question, top_k=min(10, max(1, len(event_docs))))

    summary = llm_client.generate(question, chunks)
    latency_ms = (time.perf_counter() - start) * 1000

    return DailyReportResponse(
        session_id=session_id,
        event_count=len(events),
        summary=summary,
        mode=llm_client.mode,
        latency_ms=latency_ms,
    )
