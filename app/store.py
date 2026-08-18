"""In-memory event store. Swap for a real database (or the SQLite pattern used in
ModelBench/AudioClassifier) once this needs to persist across restarts."""
from __future__ import annotations

from app.models import EmotionEvent


class EventStore:
    def __init__(self) -> None:
        self._events: list[EmotionEvent] = []

    def add(self, events: list[EmotionEvent]) -> int:
        self._events.extend(events)
        return len(events)

    def all(self) -> list[EmotionEvent]:
        return list(self._events)

    def for_session(self, session_id: str | None) -> list[EmotionEvent]:
        if session_id is None:
            return self.all()
        return [e for e in self._events if e.session_id == session_id]

    def __len__(self) -> int:
        return len(self._events)


event_store = EventStore()
