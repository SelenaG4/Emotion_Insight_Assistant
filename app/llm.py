"""LLM client with graceful degradation -- identical pattern to the audio-domain
version: tries Azure OpenAI, then OpenAI, then a deterministic offline fallback so
the service is fully demoable with zero API keys configured.
"""
from __future__ import annotations

import os
import textwrap

import httpx

from app.models import RetrievedChunk


class LLMClient:
    def __init__(self) -> None:
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    @property
    def mode(self) -> str:
        if self.azure_endpoint and self.azure_key and self.azure_deployment:
            return "azure_openai"
        if self.openai_key:
            return "openai"
        return "offline_fallback"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n".join(f"- {c.text}" for c in chunks) or "(no matching context found)"
        mode = self.mode

        if mode == "azure_openai":
            return self._call_azure_openai(question, context)
        if mode == "openai":
            return self._call_openai(question, context)
        return self._offline_fallback(question, chunks)

    def _call_azure_openai(self, question: str, context: str) -> str:
        url = (
            f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}"
            "/chat/completions?api-version=2024-02-15-preview"
        )
        headers = {"api-key": self.azure_key, "Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _call_openai(self, question: str, context: str) -> str:
        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _offline_fallback(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "No matching detections or reference material were found for this "
                "question. (offline fallback mode -- no LLM key configured)"
            )
        top = chunks[:3]
        bullet_lines = "\n".join(f"  - {c.text} (relevance {c.score:.2f})" for c in top)
        return textwrap.dedent(
            f"""\
            [offline fallback mode -- no LLM key configured, showing retrieved evidence]
            Question: {question}
            Most relevant evidence:
            {bullet_lines}
            """
        )


_SYSTEM_PROMPT = (
    "You are an assistant summarizing facial-emotion detection results for a "
    "non-technical client. Answer using only the provided context. If the context "
    "does not contain the answer, say so plainly rather than guessing. Never present "
    "emotion detections as certain -- always frame them as model output with a "
    "confidence level, not as fact about how someone feels."
)

llm_client = LLMClient()
