"""Loads data/sample_events.jsonl into a running instance via POST /ingest.

Usage:
    python scripts/load_sample.py [base_url]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_events.jsonl"


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    events = [json.loads(line) for line in DATA_FILE.read_text().splitlines() if line.strip()]

    resp = httpx.post(f"{base_url}/ingest", json={"events": events}, timeout=10.0)
    resp.raise_for_status()
    print(resp.json())


if __name__ == "__main__":
    main()
