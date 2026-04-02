"""
tasks.py — Celery task that orchestrates the full humanization pipeline.

Pipeline per paragraph:
  1. Cliché strip (rule-based)
  2. SpaCy anchor extraction
  3. Gemini Pass 1 — structural rewrite
  4. Gemini Pass 2 — conversational degradation
  5. Discourse marker injection
  6. (Optional) Evasion layer
  7. Semantic similarity validation  → re-roll if drift > threshold
  8. HuggingFace AI detection loop  → re-roll if score > 15%
  9. Reconstruct DOCX and deliver
"""

import os
import json
import tempfile
import asyncio

import redis
from celery import Task
from celery_app import celery_app

from parser import DocxParser
from humanizer import GeminiHumanizer
from validator import SemanticValidator, AIDetector
from cliche_stripper import strip_cliches
from discourse_injector import inject_discourse_markers, seed_micro_errors
from evasion import apply_evasion

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OUTPUT_DIR = tempfile.mkdtemp(prefix="dochumanize_out_")

# Maximum rewrite attempts before accepting best-so-far
MAX_RETRIES = 4


def publish(r: redis.Redis, job_id: str, payload: dict):
    """Push a progress event to the Redis pub/sub channel."""
    r.publish(f"progress:{job_id}", json.dumps(payload))


@celery_app.task(bind=True, name="tasks.humanize_document")
def humanize_document(
    self: Task,
    input_path: str,
    gemini_key: str,
    hf_token: str,
    job_id: str,
    evasion_mode: bool = False,
):
    """
    Main humanization task.
    - Parses the DOCX into a paragraph map
    - Runs each paragraph through the full hybrid NLP pipeline
    - Reconstructs and saves the DOCX
    - Publishes granular progress events via Redis pub/sub
    """
    r = redis.from_url(REDIS_URL, decode_responses=True)

    try:
        # ── Stage 1: Parse ────────────────────────────────────────────────
        publish(r, job_id, {"event": "stage", "msg": "Parsing document structure..."})
        parser = DocxParser(input_path)
        paragraph_map = parser.extract_paragraphs()
        total = len(paragraph_map)

        publish(r, job_id, {
            "event": "parsed",
            "total_paras": total,
            "msg": f"Extracted {total} paragraphs for processing.",
        })

        # ── Stage 2: Initialize pipeline components ───────────────────────
        humanizer = GeminiHumanizer(api_key=gemini_key)
        validator = SemanticValidator()
        detector = AIDetector(hf_token=hf_token)

        results: dict[int, str] = {}     # para_id → humanized text

        # ── Stage 3: Humanization loop ────────────────────────────────────
        for idx, para in enumerate(paragraph_map):
            para_id = para["id"]
            original_text = para["text"]

            # Skip very short paragraphs (headings, single words, etc.)
            if len(original_text.strip()) < 40:
                results[para_id] = original_text
                continue

            publish(r, job_id, {
                "event": "progress",
                "para": idx + 1,
                "total_paras": total,
                "pct": round(((idx + 1) / total) * 100),
                "preview": original_text[:60] + "...",
            })

            best_candidate = original_text   # fallback: keep original
            best_ai_score = 1.0              # worst possible

            for attempt in range(MAX_RETRIES):

                # Step A: Cliché strip
                cleaned = strip_cliches(original_text)

                # Step B: Gemini Pass 1 — structural rewrite
                pass1 = humanizer.rewrite_structural(
                    text=cleaned,
                    original_char_count=len(original_text),
                    attempt=attempt,
                )
                if not pass1:
                    continue

                # Step C: Gemini Pass 2 — conversational degradation
                pass2 = humanizer.rewrite_conversational(
                    text=pass1,
                    attempt=attempt,
                )
                if not pass2:
                    continue

                # Step D: Post-processing injections
                enriched = inject_discourse_markers(pass2)
                enriched = seed_micro_errors(enriched)

                # Step E: Optional evasion layer
                if evasion_mode:
                    enriched = apply_evasion(enriched)

                # Step F: Semantic similarity guard
                similarity = validator.cosine_similarity(original_text, enriched)
                if similarity < 0.55:
                    # Meaning drifted too far — retry
                    publish(r, job_id, {
                        "event": "retry",
                        "para": idx + 1,
                        "reason": f"Semantic drift (sim={similarity:.2f}), attempt {attempt+1}",
                    })
                    continue

                # Step G: AI detection loop
                ai_score = detector.score(enriched)

                # Track best candidate across retries
                if ai_score < best_ai_score:
                    best_ai_score = ai_score
                    best_candidate = enriched

                if ai_score <= 0.15:
                    # Passed — move on
                    break
                else:
                    publish(r, job_id, {
                        "event": "retry",
                        "para": idx + 1,
                        "reason": f"AI score too high ({ai_score:.0%}), attempt {attempt+1}",
                    })

            results[para_id] = best_candidate

        # ── Stage 4: Reconstruct DOCX ─────────────────────────────────────
        publish(r, job_id, {"event": "stage", "msg": "Reconstructing document..."})
        out_path = os.path.join(OUTPUT_DIR, f"{job_id}_humanized.docx")
        parser.reconstruct(results, out_path)

        # ── Stage 5: Done ─────────────────────────────────────────────────
        publish(r, job_id, {
            "event": "done",
            "download_url": f"/download/{job_id}",
            "msg": "Humanization complete.",
        })

        return {"status": "done", "output": out_path}

    except Exception as exc:
        publish(r, job_id, {"event": "error", "detail": str(exc)})
        raise

    finally:
        # Clean up input temp file
        try:
            os.remove(input_path)
        except OSError:
            pass
        r.close()