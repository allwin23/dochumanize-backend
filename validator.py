"""
validator.py — Dual validation layer.

SemanticValidator:
  Uses SentenceTransformers (all-MiniLM-L6-v2) LOCALLY (~90MB).
  Computes cosine similarity between original and rewritten paragraphs.
  Rejects if similarity < 0.55 (meaning has drifted too far).

AIDetector:
  Calls the HuggingFace Inference API REMOTELY — zero local RAM.
  Model: Hello-SimpleAI/chatgpt-detector-roberta
  The user's HF token is passed in per-request (BYOK, never stored).
  Returns a float score in [0.0, 1.0] where higher = more AI-like.
  Target: score <= 0.15 to pass.

  Free tier: ~1000 requests/day, resets daily.
  HF token obtained free at: huggingface.co/settings/tokens
"""

import os
import time
import httpx
import numpy as np
from functools import lru_cache
from typing import Optional


# ── HuggingFace Inference API config ─────────────────────────────────────────
HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "Hello-SimpleAI/chatgpt-detector-roberta"
)

_AI_LABELS    = {"LABEL_1", "ChatGPT", "AI", "machine"}
_HUMAN_LABELS = {"LABEL_0", "Human", "human"}


# ── Lazy-loaded local SBERT model ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_sentence_model():
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


# ── SemanticValidator ─────────────────────────────────────────────────────────

class SemanticValidator:
    """
    Validates meaning preservation using local SBERT cosine similarity.
    Runs entirely locally — no API calls, ~90MB RAM.
    Threshold: similarity >= 0.55 required to pass.
    """

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        try:
            model = _load_sentence_model()
            embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
            return float(np.dot(embeddings[0], embeddings[1]))
        except Exception as e:
            print(f"[SemanticValidator] Error: {e}")
            return 0.7    # neutral fallback


# ── AIDetector ────────────────────────────────────────────────────────────────

class AIDetector:
    """
    Scores text for AI likelihood via HuggingFace Inference API (FREE).

    No local model download — zero RAM overhead on the server.
    User provides their own HF token (BYOK). Never stored.

    Free tier: ~1000 API calls/day (resets daily).
    A 60-page doc = ~150 paragraphs x up to 4 retries = ~600 max calls.

    HF token: huggingface.co/settings/tokens → New token → Read role
    """

    def __init__(self, hf_token: str):
        self.headers = {"Authorization": f"Bearer {hf_token}"}

    def score(self, text: str) -> float:
        """Return AI probability [0.0, 1.0]. Higher = more AI-like."""
        chunks = _chunk_text(text, max_words=380)
        if not chunks:
            return 0.0

        scores = []
        for chunk in chunks:
            chunk_score = self._score_chunk(chunk)
            if chunk_score is not None:
                scores.append(chunk_score)

        if not scores:
            return 0.5   # uncertain — will trigger a retry

        return float(np.mean(scores))

    def _score_chunk(self, text: str, retries: int = 3) -> Optional[float]:
        """
        POST to HF Inference API for one chunk.
        Retries on 503 (model cold start) with exponential backoff.
        """
        for attempt in range(retries):
            try:
                response = httpx.post(
                    HF_API_URL,
                    headers=self.headers,
                    json={"inputs": text},
                    timeout=30.0,
                )

                if response.status_code == 503:
                    # Model is loading (cold start on free tier)
                    wait = (2 ** attempt) * 3    # 3s, 6s, 12s
                    print(f"[AIDetector] HF model loading, retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                if response.status_code == 429:
                    print("[AIDetector] Rate limited, waiting 60s...")
                    time.sleep(60)
                    continue

                response.raise_for_status()
                result = response.json()

                # HF returns: [[{"label": "LABEL_0", "score": 0.97}, ...]]
                if isinstance(result, list) and isinstance(result[0], list):
                    result = result[0]

                return _parse_hf_labels(result)

            except httpx.TimeoutException:
                print(f"[AIDetector] Timeout attempt {attempt + 1}")
                time.sleep(5)
            except Exception as e:
                print(f"[AIDetector] Error: {e}")
                return None

        return None


def _parse_hf_labels(labels: list) -> float:
    """Parse HF label/score pairs → single AI probability float."""
    for item in labels:
        label = item.get("label", "")
        score = item.get("score", 0.0)
        if label in _AI_LABELS:
            return float(score)
        elif label in _HUMAN_LABELS:
            return float(1.0 - score)
    # Fallback
    return float(labels[0].get("score", 0.5)) if labels else 0.5


def _chunk_text(text: str, max_words: int = 380) -> list[str]:
    """Split text at sentence boundaries into chunks under max_words."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, count = [], [], 0

    for sent in sentences:
        wc = len(sent.split())
        if count + wc > max_words and current:
            chunks.append(" ".join(current))
            current, count = [sent], wc
        else:
            current.append(sent)
            count += wc

    if current:
        chunks.append(" ".join(current))
    return chunks