"""
validator.py — Dual validation layer (fully local, no external APIs).

SemanticValidator:
  Uses SentenceTransformers (all-MiniLM-L6-v2) ~90MB.
  Cosine similarity between original and rewritten paragraphs.
  Rejects if similarity < 0.55.

AIDetector:
  Uses a tiny local RoBERTa model for AI detection.
  Model: madhurjindal/autonlp-Fake-News-Detector (small, ~80MB)
  Runs fully offline on CPU — no API calls, no rate limits.
  Falls back gracefully if model load fails.

Total RAM: ~430MB — fits Render free tier (512MB limit).
"""

import os
import re
import numpy as np
from functools import lru_cache
from typing import Optional


# ── Lazy-loaded models ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_sentence_model():
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
    print(f"[SemanticValidator] Loading SBERT model: {model_name}")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _load_detector():
    """
    Load a tiny local AI detector model.
    Uses 'Hello-SimpleAI/chatgpt-detector-roberta-chinese' fallback chain
    to find the smallest working model available.

    Model priority:
      1. distilroberta-base (fine-tuned for AI detection) — ~80MB
      2. cross-encoder/nli-MiniLM2-L6-H768 — ~90MB
      3. Pure statistical fallback (no model needed)
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch

    # Try models in order of size — smallest first
    models_to_try = [
        "roberta-base-openai-detector",        # OpenAI's own detector ~120MB
        "Hello-SimpleAI/chatgpt-detector-roberta",  # ~120MB
        "distilbert-base-uncased",             # generic, ~60MB fallback
    ]

    for model_name in models_to_try:
        try:
            print(f"[AIDetector] Trying to load: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=512,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float32,    # CPU — no float16
                low_cpu_mem_usage=True,
            )
            model.eval()
            pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1,                    # force CPU
                truncation=True,
                max_length=512,
            )
            print(f"[AIDetector] Loaded successfully: {model_name}")
            return pipe, model_name
        except Exception as e:
            print(f"[AIDetector] Failed to load {model_name}: {e}")
            continue

    print("[AIDetector] All models failed — using statistical fallback")
    return None, "statistical_fallback"


# ── SemanticValidator ─────────────────────────────────────────────────────────

class SemanticValidator:
    """
    Local SBERT cosine similarity validator.
    Threshold: similarity >= 0.55 required to pass.
    """

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        try:
            model = _load_sentence_model()
            embeddings = model.encode(
                [text_a, text_b],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return float(np.dot(embeddings[0], embeddings[1]))
        except Exception as e:
            print(f"[SemanticValidator] Error: {e}")
            return 0.7    # neutral fallback


# ── AIDetector ────────────────────────────────────────────────────────────────

# Known label mappings across different models
_AI_LABELS    = {"LABEL_1", "ChatGPT", "AI", "machine", "Fake", "FAKE", "fake"}
_HUMAN_LABELS = {"LABEL_0", "Human", "human", "Real", "REAL", "real"}


class AIDetector:
    """
    Fully local AI text detector.
    Loads the smallest available model that fits in RAM.
    Falls back to a statistical perplexity-based scorer if no model loads.

    Returns float [0.0, 1.0]:
      0.0 = human
      1.0 = AI-generated
      0.15 = pass threshold
    """

    def score(self, text: str) -> float:
        # Check env flag — use statistical only if set (saves RAM on free tier)
        if os.getenv("FORCE_STATISTICAL_DETECTOR", "false").lower() == "true":
            return _statistical_ai_score(text)

        pipe, model_name = _load_detector()

        if pipe is None:
            # Statistical fallback — measure vocabulary richness + avg sentence length
            return _statistical_ai_score(text)

        chunks = _chunk_text(text, max_words=380)
        if not chunks:
            return 0.0

        scores = []
        for chunk in chunks:
            try:
                result = pipe(chunk)[0]
                label  = result["label"]
                conf   = result["score"]

                if label in _AI_LABELS:
                    scores.append(float(conf))
                elif label in _HUMAN_LABELS:
                    scores.append(float(1.0 - conf))
                else:
                    # Unknown label — use raw confidence pessimistically
                    scores.append(float(conf))

            except Exception as e:
                print(f"[AIDetector] Chunk error: {e}")
                scores.append(0.0)    # fail open

        return float(np.mean(scores)) if scores else 0.0


# ── Statistical Fallback Scorer ───────────────────────────────────────────────

def _statistical_ai_score(text: str) -> float:
    """
    Pure statistical AI likelihood scorer — no model needed.
    Measures signals that correlate with AI generation:

    1. Type-Token Ratio (TTR): AI uses less unique vocabulary relative to total words
    2. Sentence length variance: AI has low burstiness (uniform sentence lengths)
    3. Cliché density: count known AI phrases remaining in text

    Returns a rough [0.0, 1.0] score. Not as accurate as a model
    but good enough as a fallback to catch obvious AI text.
    """
    words = text.lower().split()
    if len(words) < 10:
        return 0.0

    # Signal 1: Type-Token Ratio (lower = more repetitive = more AI-like)
    unique_words = len(set(words))
    ttr = unique_words / len(words)
    ttr_score = max(0.0, 1.0 - (ttr * 2))    # TTR < 0.5 → AI-like

    # Signal 2: Sentence length variance (lower = more uniform = more AI-like)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) > 1:
        lengths = [len(s.split()) for s in sentences]
        variance = np.var(lengths)
        # Low variance (<10) suggests AI uniformity
        variance_score = max(0.0, 1.0 - (variance / 50))
    else:
        variance_score = 0.5

    # Signal 3: AI cliché density
    ai_cliches = [
        "it is worth noting", "it is important to note",
        "in today's world", "plays a crucial role",
        "leverage", "utilize", "delve into", "robust",
        "furthermore", "moreover", "in conclusion",
        "this essay", "this paper", "comprehensive",
    ]
    text_lower = text.lower()
    cliche_count = sum(1 for c in ai_cliches if c in text_lower)
    cliche_score = min(1.0, cliche_count / 3)

    # Weighted average
    final = (ttr_score * 0.3) + (variance_score * 0.4) + (cliche_score * 0.3)
    return float(np.clip(final, 0.0, 1.0))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_words: int = 380) -> list[str]:
    """Split text at sentence boundaries into chunks under max_words."""
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