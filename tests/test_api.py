import io
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.store import event_store

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_events.jsonl"

client = TestClient(app)


def _load_sample_events() -> list[dict]:
    return [json.loads(line) for line in DATA_FILE.read_text().splitlines() if line.strip()]


def setup_function() -> None:
    event_store._events.clear()  # test isolation


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_mode"] in {"azure_openai", "openai", "offline_fallback"}
    assert body["cnn_weights_loaded"] is False  # no trained weights shipped in this repo


def test_predict_endpoint_runs_end_to_end() -> None:
    fake_face = (np.random.rand(48, 48) * 255).astype("uint8")
    img = Image.fromarray(fake_face)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post("/predict", files={"file": ("face.png", buf, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"happy", "sad", "neutral", "surprise"}
    assert body["weights_loaded"] is False


def test_ingest_then_ask() -> None:
    events = _load_sample_events()
    ingest_resp = client.post("/ingest", json={"events": events})
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["stored"] == len(events)

    ask_resp = client.post(
        "/ask",
        json={"question": "Was there anything late at night in the usability test?"},
    )
    assert ask_resp.status_code == 200
    body = ask_resp.json()
    assert body["mode"] == "offline_fallback"
    assert len(body["retrieved"]) > 0
    top_sources = [c["text"] for c in body["retrieved"][:3]]
    assert any("22:0" in t or "usability-test-02" in t for t in top_sources)


def test_ask_about_class_reliability_hits_knowledge_base() -> None:
    resp = client.post("/ask", json={"question": "Which emotion class is least reliable?"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["retrieved"]) > 0
    assert any("sad" in c["text"].lower() for c in body["retrieved"])


def test_daily_report() -> None:
    events = _load_sample_events()
    client.post("/ingest", json={"events": events})

    resp = client.get("/report/daily", params={"session_id": "focus-group-01"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["event_count"] == sum(1 for e in events if e["session_id"] == "focus-group-01")
