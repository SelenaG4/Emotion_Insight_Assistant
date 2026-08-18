"""TF-IDF + cosine similarity retrieval -- same approach as the audio-domain version
of this pattern (see the separate Audio Insight Assistant project), applied here to
emotion-detection events and class notes instead of sound-detection events.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import EmotionEvent, RetrievedChunk


@dataclass
class Document:
    source: str
    text: str


def load_knowledge_base(path: Path) -> list[Document]:
    docs: list[Document] = []
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        first_line, *rest = block.splitlines()
        docs.append(Document(source=first_line.strip("# ").strip(), text=" ".join(rest).strip()))
    return docs


def _time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late at night"


def events_to_documents(events: list[EmotionEvent]) -> list[Document]:
    docs = []
    for e in events:
        tod = _time_of_day(e.timestamp.hour)
        text = (
            f"At {e.timestamp.isoformat()} ({tod}) the {e.source_model} model "
            f"detected '{e.label}' with confidence {e.confidence:.2f} in session {e.session_id}."
        )
        docs.append(Document(source=f"detection:{e.session_id}:{e.timestamp.isoformat()}", text=text))
    return docs


class TfidfRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [d.text for d in documents] or [""]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def query(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self._documents:
            return []
        q_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(zip(self._documents, scores), key=lambda t: t[1], reverse=True)
        return [
            RetrievedChunk(source=doc.source, text=doc.text, score=float(score))
            for doc, score in ranked[:top_k]
            if score > 0
        ]
