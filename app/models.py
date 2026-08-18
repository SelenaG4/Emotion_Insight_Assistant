from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EmotionEvent(BaseModel):
    """One row of model output: a single facial-emotion prediction."""

    timestamp: datetime
    session_id: str
    label: str = Field(description="one of: happy, sad, neutral, surprise")
    confidence: float = Field(ge=0.0, le=1.0)
    source_model: str = Field(default="model3_facial_emotion_cnn")


class IngestRequest(BaseModel):
    events: list[EmotionEvent]


class IngestResponse(BaseModel):
    stored: int
    total_events: int


class PredictResponse(BaseModel):
    label: str
    confidence: float
    per_class: dict[str, float]
    weights_loaded: bool = Field(
        description="False means the CNN is running on random-initialized weights "
        "(no trained model3 weights file found) -- prediction is not meaningful."
    )


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    source: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    mode: str  # "azure_openai" | "openai" | "offline_fallback"
    retrieved: list[RetrievedChunk]
    latency_ms: float


class DailyReportResponse(BaseModel):
    session_id: Optional[str]
    event_count: int
    summary: str
    mode: str
    latency_ms: float
