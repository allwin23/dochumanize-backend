"""
tasks.py — Main Celery background task (parallel edition).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS FILE DOES (plain English):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When a user uploads a DOCX, this task wakes up in the background
and runs the full humanization pipeline on every paragraph.

It processes paragraphs IN PARALLEL (8 at a time by default)
instead of one-by-one, which cuts processing time from hours → minutes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE PER PARAGRAPH (in order):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 1 — Cliché Strip
           Rule-based removal of known AI phrases
           ("In today's world", "delve into", "leverage", etc.)
           Done BEFORE sending to Gemini so LLM doesn't reproduce them.

  Step 2 — Gemini Pass 1: Structural Rewrite
           Sends cleaned text to Gemini with strict instructions:
           - Change sentence structure (passive→active, cleft sentences)
           - Enforce burstiness (vary sentence lengths: long/short/medium)
           - Stay within ±5% of original character count
           - Preserve all facts, terms, and technical content

  Step 3 — Gemini Pass 2: Conversational Degradation
           Takes Pass 1 output and deliberately roughens it:
           - Adds discourse markers ("That said,", "Curiously,")
           - Injects mild hedges ("arguably", "in most cases")
           - Allows sentence-initial "And" / "But"
           Makes it sound like a human expert, not a polished AI.

  Step 4 — Discourse Marker Injection (post-processing)
           Rule-based layer that adds additional human connectives
           at sentence boundaries (30% probability per boundary).

  Step 5 — Micro-Error Seeding (post-processing)
           Probabilistically injects human stylistic patterns:
           contractions, em-dashes, sentence fragments.
           These are invisible to readers but spike perplexity scores.

  Step 6 — Evasion Layer (optional, user-toggled)
           If evasion_mode=True, injects zero-width spaces and
           homoglyph character substitutions to confuse tokenizers.

  Step 7 — Semantic Similarity Check (local SBERT model)
           Computes cosine similarity between original and rewritten text.
           If similarity < 0.55 → meaning drifted too far → RETRY.
           Uses all-MiniLM-L6-v2 running locally (~90MB RAM).

  Step 8 — AI Detection Check (local RoBERTa model)
           Scores the rewritten text for AI likelihood [0.0 → 1.0].
           If score > 0.15 (15%) → still sounds like AI → RETRY.
           Uses roberta-base-openai-detector running locally (~120MB RAM).
           Max 3 retries — accepts best candidate if threshold not met.

  Step 9 — Reconstruct DOCX
           Stitches all humanized paragraphs back into the original
           DOCX XML structure. Tables, images, fonts, layout = untouched.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARALLEL PROCESSING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Old (sequential):  Para1(34s) → Para2(34s) → Para3(34s) = 85 min for 150 paras
  New (parallel):    Para1 ↘
                     Para2 → all 8 run at once → ~2 min for 150 paras
                     Para3 ↗
  Controlled by PARALLEL_WORKERS env var (default: 8).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU WILL SEE IN THE CELERY WORKER TERMINAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ============================================================
    STAGE 1/4 — Parsing document...
  ============================================================
    STAGE 1/4 — DONE. 87 paragraphs found.

    ⏱  Estimated time: ~1m 52s (8 parallel workers)

  ============================================================
    STAGE 3/4 — Humanizing in parallel...
  ============================================================
    [██░░░░░░░░░░░░░░░░░░] 10%  Para 9/87 — Started (245 chars)
      Para 9 | Step 1: Cliché strip done
      Para 9 | Step 2: Gemini Pass 1...
      Para 9 | Step 3: Gemini Pass 2...
      Para 9 | Step 4+5: Discourse + micro-errors injected
      Para 9 | Step 7: Semantic similarity = 0.74 ✓
      Para 9 | Step 8: AI score = 11% ✓ PASSED
    [██░░░░░░░░░░░░░░░░░░] 10%  Para 9/87 — ✓ DONE

    ✅ 9/87 paragraphs complete
"""

import os
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import redis
from celery import Task
from celery_app import celery_app

from parser import DocxParser
from humanizer import GeminiHumanizer
from validator import SemanticValidator, AIDetector
from cliche_stripper import strip_cliches
from discourse_injector import inject_discourse_markers, seed_micro_errors
from evasion import apply_evasion

# ── Config ────────────────────────────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OUTPUT_DIR = "/tmp/dochumanize_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Max times to retry a single paragraph before accepting best result
MAX_RETRIES = 3

# How many paragraphs to process at the same time.
# 8 is safe for Gemini free tier without hitting rate limits.
# Increase to 12-16 if you have a paid Gemini plan.
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "8"))


# ── Logging helpers ───────────────────────────────────────────────────────────

def publish(r: redis.Redis, job_id: str, payload: dict):
    """
    Push a progress event to the Redis pub/sub channel.
    The FastAPI WebSocket handler picks this up and forwards it to the browser.
    """
    r.publish(f"progress:{job_id}", json.dumps(payload))


def log(msg: str):
    """
    Print a bold stage-level banner to the Celery worker terminal.
    Use this for major milestones (stage changes, job complete/fail).
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'='*60}\n", flush=True)


def log_para(para_num: int, total: int, msg: str):
    """
    Print a progress bar + paragraph status to the worker terminal.
    Example:
      [████░░░░░░░░░░░░░░░░] 20%  Para 4/20 — ✓ DONE
    """
    pct = round((para_num / total) * 100)
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    print(f"  [{bar}] {pct:3d}%  Para {para_num}/{total} — {msg}", flush=True)


def log_step(para_num: int, msg: str):
    """
    Print a step-level detail line for a specific paragraph.
    Example:
      Para 4 | Step 2: Gemini Pass 1...
    """
    print(f"      Para {para_num} | {msg}", flush=True)


# ── Single paragraph pipeline ─────────────────────────────────────────────────

def process_single_paragraph(
    para: dict,
    para_num: int,
    total: int,
    humanizer: GeminiHumanizer,
    validator: SemanticValidator,
    detector: AIDetector,
    evasion_mode: bool,
    r: redis.Redis,
    job_id: str,
    lock: threading.Lock,
) -> tuple[int, str]:
    """
    Runs the full 9-step pipeline on a single paragraph.
    Called from a thread pool — runs in parallel with other paragraphs.

    Args:
        para        — dict with keys: id, text, style, word_count
        para_num    — 1-based display number (for logging)
        total       — total paragraph count (for progress bar)
        humanizer   — GeminiHumanizer instance (shared across threads, thread-safe)
        validator   — SemanticValidator instance (shared, thread-safe)
        detector    — AIDetector instance (shared, thread-safe)
        evasion_mode — whether to apply zero-width space / homoglyph injection
        r           — Redis client for publishing WebSocket progress events
        job_id      — unique job identifier
        lock        — threading.Lock to prevent interleaved terminal output

    Returns:
        (para_id, humanized_text) — tuple to be collected by the main task
    """
    para_id  = para["id"]
    original = para["text"]

    # ── Skip short paragraphs (headings, captions, single lines) ─────────────
    # These are not worth rewriting — too short to have meaningful AI signals
    if len(original.strip()) < 40:
        with lock:
            log_para(para_num, total, "SKIPPED — too short to rewrite")
        return (para_id, original)

    # ── Log start + publish WebSocket event ───────────────────────────────────
    with lock:
        log_para(para_num, total, f"Started ({len(original.split())} words)")
        publish(r, job_id, {
            "event": "progress",
            "para": para_num,
            "total_paras": total,
            "pct": round((para_num / total) * 100),
            "preview": original[:60] + "...",
        })

    # Track the best rewrite across retries
    # If we never pass the AI detector, we use the lowest-scoring attempt
    best_candidate = original   # fallback = keep original if everything fails
    best_ai_score  = 1.0        # 1.0 = worst possible (fully AI-sounding)

    # ── Retry loop ────────────────────────────────────────────────────────────
    for attempt in range(MAX_RETRIES):

        if attempt > 0:
            with lock:
                print(f"      Para {para_num} | ↻ Retry {attempt}/{MAX_RETRIES - 1}", flush=True)

        try:
            # ── STEP 1: Cliché Strip ──────────────────────────────────────────
            # Remove known AI phrases BEFORE sending to Gemini.
            # This prevents the LLM from seeing and reproducing them.
            # Examples stripped: "In today's world", "delve into", "leverage",
            # "it is important to note", "plays a crucial role", etc.
            with lock:
                log_step(para_num, "Step 1: Stripping AI clichés...")
            cleaned = strip_cliches(original)

            # ── STEP 2: Gemini Pass 1 — Structural Rewrite ───────────────────
            # Sends the cleaned text to Gemini with instructions to:
            #   - Change sentence structure (passive→active, cleft sentences)
            #   - Vary sentence lengths in a specific rhythm (burstiness)
            #   - Stay within ±5% of original character count
            #   - Preserve all facts, proper nouns, technical terms
            # Higher attempt number = more aggressive restructuring prompt
            with lock:
                log_step(para_num, "Step 2: Gemini Pass 1 (structural rewrite)...")
            pass1 = humanizer.rewrite_structural(
                text=cleaned,
                original_char_count=len(original),
                attempt=attempt,
            )
            if not pass1:
                with lock:
                    log_step(para_num, "Step 2: ✗ Pass 1 returned empty — retrying")
                continue

            # ── STEP 3: Gemini Pass 2 — Conversational Degradation ───────────
            # Takes the structurally rewritten text and deliberately roughens it.
            # Instructions include:
            #   - Add one discourse marker ("That said,", "Curiously,")
            #   - Allow sentence-initial "And" or "But"
            #   - Add mild hedges ("arguably", "in most cases")
            #   - Allow one intentional fragment for rhythm
            # This second pass removes the "too polished" quality that AI detectors flag.
            with lock:
                log_step(para_num, "Step 3: Gemini Pass 2 (conversational polish)...")
            pass2 = humanizer.rewrite_conversational(text=pass1, attempt=attempt)
            if not pass2:
                with lock:
                    log_step(para_num, "Step 3: ✗ Pass 2 returned empty — retrying")
                continue

            # ── STEP 4: Discourse Marker Injection ───────────────────────────
            # Rule-based post-processing layer.
            # Walks sentence boundaries and probabilistically inserts
            # human reasoning connectives (30% chance per boundary):
            # "That said,", "Worth noting here is that", "The trouble is,", etc.
            # These connectives are rare in AI output and boost human-likeness scores.
            with lock:
                log_step(para_num, "Step 4: Injecting discourse markers...")
            enriched = inject_discourse_markers(pass2)

            # ── STEP 5: Micro-Error Seeding ───────────────────────────────────
            # Injects subtle human stylistic patterns that AI detectors
            # cannot penalise without also flagging real human writing:
            #   - Contractions (does not → doesn't)
            #   - Em-dash interruptions (comma → em-dash)
            #   - Sentence-initial conjunctions (And / But)
            # Applied probabilistically (~20% of eligible positions).
            with lock:
                log_step(para_num, "Step 5: Seeding micro-errors...")
            enriched = seed_micro_errors(enriched)

            # ── STEP 6: Evasion Layer (optional) ─────────────────────────────
            # Only runs if user enabled evasion_mode toggle.
            # Injects invisible Unicode characters:
            #   - Zero-width spaces (U+200B) at 8% of word boundaries
            #   - Homoglyph substitutions (Latin 'e' → Cyrillic 'е') at 3%
            #   - Soft hyphens (U+00AD) inside long words at 10%
            # Text looks identical to humans but breaks AI detector tokenization.
            if evasion_mode:
                with lock:
                    log_step(para_num, "Step 6: Applying evasion layer (ZWS + homoglyphs)...")
                enriched = apply_evasion(enriched)
            else:
                with lock:
                    log_step(para_num, "Step 6: Evasion layer OFF (skipped)")

            # ── STEP 7: Semantic Similarity Check ────────────────────────────
            # Uses local SBERT model (all-MiniLM-L6-v2, ~90MB RAM).
            # Encodes both original and rewritten text as vectors,
            # then computes cosine similarity [0.0 → 1.0].
            #
            # Thresholds:
            #   >= 0.55 → meaning preserved → proceed to AI detection
            #   <  0.55 → meaning drifted too far → RETRY
            #
            # Why 0.55? Below this, the rewrite has likely hallucinated
            # new facts or omitted critical information.
            with lock:
                log_step(para_num, "Step 7: Checking semantic similarity (SBERT)...")
            similarity = validator.cosine_similarity(original, enriched)

            if similarity < 0.55:
                with lock:
                    log_step(para_num, f"Step 7: ✗ FAILED — similarity {similarity:.2f} < 0.55 (meaning drifted) — retrying")
                    publish(r, job_id, {
                        "event": "retry",
                        "para": para_num,
                        "reason": f"Semantic drift (sim={similarity:.2f})",
                    })
                continue
            else:
                with lock:
                    log_step(para_num, f"Step 7: ✓ PASSED — similarity {similarity:.2f}")

            # ── STEP 8: AI Detection Check ────────────────────────────────────
            # Uses local RoBERTa model (roberta-base-openai-detector, ~120MB RAM).
            # Scores the rewritten text: 0.0 = human, 1.0 = AI-generated.
            #
            # Thresholds:
            #   <= 0.15 (15%) → passes as human → done, move to next paragraph
            #   >  0.15       → still sounds like AI → RETRY with more aggression
            #
            # Best candidate across all retries is always saved.
            # If we exhaust MAX_RETRIES without passing, we use the
            # lowest-scoring attempt rather than reverting to original.
            with lock:
                log_step(para_num, "Step 8: Running AI detection (local RoBERTa)...")
            ai_score = detector.score(enriched)

            # Always save the best result we've seen so far
            if ai_score < best_ai_score:
                best_ai_score  = ai_score
                best_candidate = enriched

            if ai_score <= 0.15:
                with lock:
                    log_step(para_num, f"Step 8: ✓ PASSED — AI score {ai_score:.0%} (under 15% target)")
                break   # exit retry loop — this paragraph is done
            else:
                with lock:
                    log_step(para_num, f"Step 8: ✗ FAILED — AI score {ai_score:.0%} (over 15%) — retrying")
                    publish(r, job_id, {
                        "event": "retry",
                        "para": para_num,
                        "reason": f"AI score {ai_score:.0%} > 15%",
                    })

        except Exception as e:
            # Don't let one paragraph crash the whole job.
            # Log the error, skip this attempt, and retry.
            with lock:
                log_step(para_num, f"✗ Exception on attempt {attempt + 1}: {e}")
            continue

    # ── End of retry loop ─────────────────────────────────────────────────────
    # Use whichever attempt scored lowest on AI detection.
    # Even if we never hit the 15% target, a 40% score is better than 100%.
    with lock:
        log_para(para_num, total, f"✓ DONE — best AI score: {best_ai_score:.0%}")

    return (para_id, best_candidate)


# ── Main Celery task ──────────────────────────────────────────────────────────

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
    Root Celery task. Called when a document is uploaded via POST /upload.

    Args:
        input_path  — path to the uploaded .docx on disk (temp file)
        gemini_key  — user's Gemini API key (BYOK, never stored)
        hf_token    — user's HuggingFace token (kept for API compatibility,
                      not used since detector now runs locally)
        job_id      — UUID for this job, used for WebSocket pub/sub channel
        evasion_mode — if True, apply zero-width space + homoglyph injection

    Flow:
        Stage 1 — Parse DOCX → extract paragraph map
        Stage 2 — Load AI models (SBERT + RoBERTa)
        Stage 3 — Process all paragraphs in parallel (ThreadPoolExecutor)
        Stage 4 — Reconstruct DOCX with humanized text
        Stage 5 — Publish done event → user downloads file
    """
    r = redis.from_url(REDIS_URL, decode_responses=True)

    try:
        # ── STAGE 1: Parse DOCX ───────────────────────────────────────────────
        # Opens the .docx file and walks the XML to find all paragraphs
        # that are eligible for rewriting (body text, not tables/images/headings).
        # Each paragraph gets a unique ID and is stored in a map.
        log("STAGE 1/4 — Parsing DOCX structure...")
        publish(r, job_id, {"event": "stage", "msg": "Parsing document..."})

        parser        = DocxParser(input_path)
        paragraph_map = parser.extract_paragraphs()
        total         = len(paragraph_map)

        log(f"STAGE 1/4 — DONE. Found {total} paragraphs eligible for rewriting.")
        publish(r, job_id, {
            "event": "parsed",
            "total_paras": total,
            "msg": f"Found {total} paragraphs.",
        })

        # ── STAGE 2: Load AI models ───────────────────────────────────────────
        # Initialise all pipeline components.
        # Models are lazy-loaded with @lru_cache so they load once per worker
        # process and stay in memory for subsequent tasks — no repeated downloads.
        log("STAGE 2/4 — Initialising pipeline components...")

        humanizer = GeminiHumanizer(api_key=gemini_key)
        # GeminiHumanizer: wraps the Gemini API client, holds the two prompt builders

        validator = SemanticValidator()
        # SemanticValidator: loads SBERT all-MiniLM-L6-v2 (~90MB) on first call

        detector = AIDetector()
        # AIDetector: loads roberta-base-openai-detector (~120MB) on first call

        # Estimate processing time for user feedback
        est_seconds = (total / PARALLEL_WORKERS) * 8
        est_mins    = int(est_seconds // 60)
        est_secs    = int(est_seconds % 60)

        log(f"STAGE 2/4 — DONE. Ready to process {total} paragraphs.")
        print(f"\n  ⏱  Estimated time: ~{est_mins}m {est_secs}s ({PARALLEL_WORKERS} parallel workers)\n", flush=True)

        # ── STAGE 3: Parallel humanization ───────────────────────────────────
        # This is the main processing stage.
        # ThreadPoolExecutor submits all paragraphs as independent tasks.
        # Up to PARALLEL_WORKERS paragraphs run simultaneously.
        # A threading.Lock prevents interleaved/garbled terminal output.
        # Results are collected as futures complete (not in order — order restored later).
        log(f"STAGE 3/4 — Humanizing {total} paragraphs ({PARALLEL_WORKERS} at a time)...")
        publish(r, job_id, {"event": "stage", "msg": f"Humanizing {total} paragraphs..."})

        results = {}          # para_id → final humanized text
        lock    = threading.Lock()   # prevents garbled output from concurrent threads
        completed_count = 0

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:

            # Submit all paragraphs to the thread pool at once
            # Each future maps back to its para_id for error recovery
            futures = {
                executor.submit(
                    process_single_paragraph,
                    para,           # paragraph dict (id, text, style, word_count)
                    idx + 1,        # 1-based para number for display
                    total,          # total count for progress bar
                    humanizer,      # shared Gemini client (thread-safe)
                    validator,      # shared SBERT model (thread-safe)
                    detector,       # shared RoBERTa model (thread-safe)
                    evasion_mode,   # evasion toggle
                    r,              # Redis client for WebSocket events
                    job_id,         # job UUID
                    lock,           # output lock
                ): para["id"]
                for idx, para in enumerate(paragraph_map)
            }

            # Collect results as each thread completes (order doesn't matter here —
            # results dict uses para_id as key, order restored in Stage 4)
            for future in as_completed(futures):
                try:
                    para_id, humanized_text = future.result()
                    results[para_id] = humanized_text
                    completed_count += 1
                    print(f"\n  ✅ {completed_count}/{total} paragraphs complete\n", flush=True)

                except Exception as e:
                    # If a paragraph's thread crashes entirely, fall back to original text
                    para_id = futures[future]
                    print(f"\n  ❌ Para {para_id} thread crashed: {e} — keeping original text\n", flush=True)
                    original = next(
                        (p["text"] for p in paragraph_map if p["id"] == para_id),
                        ""
                    )
                    results[para_id] = original
                    completed_count += 1

        log(f"STAGE 3/4 — DONE. All {total} paragraphs processed.")

        # ── STAGE 4: Reconstruct DOCX ─────────────────────────────────────────
        # Takes the results dict {para_id: humanized_text} and surgically
        # replaces the text nodes in the original DOCX XML.
        # Everything else (tables, images, fonts, page layout) is untouched.
        # The output file is saved to OUTPUT_DIR for the download endpoint.
        log("STAGE 4/4 — Rebuilding DOCX with humanized text...")
        publish(r, job_id, {"event": "stage", "msg": "Rebuilding document..."})

        out_path = os.path.join(OUTPUT_DIR, f"{job_id}_humanized.docx")
        parser.reconstruct(results, out_path)

        # ── STAGE 5: Done ─────────────────────────────────────────────────────
        log(f"✅ JOB COMPLETE — File ready at /download/{job_id}")
        publish(r, job_id, {
            "event": "done",
            "download_url": f"/download/{job_id}",
            "msg": "Humanization complete. Your document is ready.",
        })

        return {"status": "done", "output": out_path}

    except Exception as exc:
        # Top-level catch — log clearly and re-raise so Celery marks job as FAILURE
        log(f"❌ JOB FAILED — {str(exc)}")
        publish(r, job_id, {"event": "error", "detail": str(exc)})
        raise

    finally:
        # Always clean up the uploaded temp file regardless of success/failure
        # The output file is kept (for download) — only the input is deleted
        try:
            os.remove(input_path)
        except OSError:
            pass
        r.close()