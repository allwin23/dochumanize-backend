"""
tasks.py — Parallel humanization pipeline with NLP surgery.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE PER PARAGRAPH (9 steps):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1 — Cliché Strip (rule-based)
           Remove 50+ known AI phrases before Gemini sees the text.

  Step 2 — Gemini Pass 1: Structural Rewrite
           Aggressively restructure syntax, enforce burstiness rhythm,
           inject rare grammatical constructions, replace AI vocabulary.

  Step 3 — Gemini Pass 2: Conversational Degradation
           Add human reasoning connectives, hedges, mild imperfections.

  Step 4 — NLP Surgery (LOCAL — no API call)
           The most powerful step. Runs after Gemini to fix what LLMs can't:
             a) POS-aware synonym replacement (rarer vocabulary)
             b) Sentence splitting to enforce true burstiness
             c) Parenthetical interruption injection (em-dash asides)
           This directly attacks perplexity and burstiness signals.

  Step 5 — Discourse Marker Injection (post-processing)
           Probabilistic insertion of reasoning connectives at boundaries.

  Step 6 — Micro-Error Seeding (post-processing)
           Contractions, em-dashes, sentence-initial "And"/"But".

  Step 7 — Word-Merge Safety Net
           Strip all invisible characters, fix any merged words.

  Step 8 — Semantic Similarity Check (local SBERT)
           Cosine similarity >= 0.55 required. Retry if meaning drifted.

  Step 9 — AI Detection Check (local RoBERTa)
           Score <= 0.15 (15%) required. Retry with more aggression if not.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY NLP SURGERY BEATS PROMPT ENGINEERING ALONE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Gemini cannot escape its own output distribution no matter how we prompt it.
  The NLP surgeon operates BELOW the LLM layer — at raw grammatical structure.
  It directly modifies the signals (perplexity, burstiness, n-gram patterns)
  that AI detectors measure, in ways that no LLM can replicate reliably.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROCESSING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  All paragraphs processed in parallel (PARALLEL_WORKERS threads).
  Default: 8 workers. 60-page doc (~112 paras) = ~2 minutes.
"""

import os
import re
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
from evasion import apply_evasion, strip_evasion
from nlp_surgeon import NLPSurgeon

REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OUTPUT_DIR       = tempfile.mkdtemp(prefix="dochumanize_out_")
MAX_RETRIES      = 3
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "8"))


# ── Logging ───────────────────────────────────────────────────────────────────

def publish(r, job_id, payload):
    r.publish(f"progress:{job_id}", json.dumps(payload))

def log(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}\n", flush=True)

def log_para(num, total, msg):
    pct    = round((num / total) * 100)
    filled = int(pct / 5)
    bar    = "█" * filled + "░" * (20 - filled)
    print(f"  [{bar}] {pct:3d}%  Para {num}/{total} — {msg}", flush=True)

def log_step(num, msg):
    print(f"      Para {num} | {msg}", flush=True)


# ── Word-merge safety net ─────────────────────────────────────────────────────

def _fix_word_merging(text: str) -> str:
    """
    Strip all invisible evasion characters and fix common word-merge patterns.
    Runs on every paragraph before validation to catch artifacts.

    Common merges fixed:
      "inpractice" → "in practice"
      "wemust"     → "we must"
      "ofthe"      → "of the"
    """
    # Strip ZWS and soft hyphens
    text = strip_evasion(text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # Fix common word-merge patterns caused by invisible character artifacts
    fixes = [
        (r'\bin(practice|theory|general|fact|particular|conclusion|addition|contrast|summary)\b',
         lambda m: 'in ' + m.group(1)),
        (r'\bof(our|their|its|the|a|an|this|that)\b',
         lambda m: 'of ' + m.group(1)),
        (r'\bfor(the|a|an|this|that|these|those|each)\b',
         lambda m: 'for ' + m.group(1)),
        (r'\band(the|a|an|this|that|its|their|our)\b',
         lambda m: 'and ' + m.group(1)),
        (r'\bwe(must|can|should|will|need|have)\b',
         lambda m: 'we ' + m.group(1)),
        (r'\bthis(means|shows|suggests|indicates|requires)\b',
         lambda m: 'this ' + m.group(1)),
        (r'\bthat(the|a|an|this|these)\b',
         lambda m: 'that ' + m.group(1)),
    ]
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text.strip()


# ── Single paragraph pipeline ─────────────────────────────────────────────────

def process_single_paragraph(
    para, para_num, total,
    humanizer, validator, detector, surgeon,
    evasion_mode, r, job_id, lock
):
    """
    Full 9-step pipeline for one paragraph.
    Runs in a thread — safe for parallel execution.
    Returns (para_id, humanized_text).
    """
    para_id  = para["id"]
    original = para["text"]

    # Skip very short paragraphs — not worth rewriting
    if len(original.strip()) < 40:
        with lock:
            log_para(para_num, total, "SKIPPED — too short")
        return (para_id, original)

    with lock:
        log_para(para_num, total, f"Started ({len(original.split())} words)")
        publish(r, job_id, {
            "event": "progress",
            "para": para_num,
            "total_paras": total,
            "pct": round((para_num / total) * 100),
            "preview": original[:60] + "...",
        })

    best_candidate = original
    best_ai_score  = 1.0

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            with lock:
                print(f"      Para {para_num} | ↻ Retry {attempt}/{MAX_RETRIES-1}", flush=True)
        try:

            # ── STEP 1: Cliché Strip ──────────────────────────────────────
            # Remove AI phrases BEFORE Gemini sees them.
            # Prevents the LLM from seeing and reproducing clichés.
            with lock: log_step(para_num, "Step 1: Stripping AI clichés...")
            cleaned = strip_cliches(original)

            # ── STEP 2: Gemini Pass 1 — Structural Rewrite ───────────────
            # Aggressively restructure syntax. Higher attempt = more aggressive.
            # Enforces burstiness rhythm, rare structures, vocabulary replacement.
            with lock: log_step(para_num, "Step 2: Gemini Pass 1 (structural)...")
            pass1 = humanizer.rewrite_structural(
                text=cleaned,
                original_char_count=len(original),
                attempt=attempt,
            )
            if not pass1:
                with lock: log_step(para_num, "Step 2: ✗ Empty response — retrying")
                continue

            # ── STEP 3: Gemini Pass 2 — Conversational Degradation ───────
            # Add human reasoning markers, hedges, mild imperfections.
            # Removes the "too polished" quality that detectors flag.
            with lock: log_step(para_num, "Step 3: Gemini Pass 2 (conversational)...")
            pass2 = humanizer.rewrite_conversational(text=pass1, attempt=attempt)
            if not pass2:
                with lock: log_step(para_num, "Step 3: ✗ Empty response — retrying")
                continue

            # ── STEP 4: NLP Surgery ───────────────────────────────────────
            # LOCAL processing — no API call.
            # Directly attacks perplexity and burstiness signals:
            #   a) POS-aware synonym replacement (rarer vocabulary)
            #   b) Sentence splitting (true burstiness enforcement)
            #   c) Parenthetical em-dash injection
            # This is what actually gets us below 15%.
            with lock: log_step(para_num, "Step 4: NLP surgery (synonyms + burstiness + parentheticals)...")
            surgered = surgeon.operate(pass2)

            # ── STEP 5: Discourse Marker Injection ───────────────────────
            # Rule-based: insert reasoning connectives at sentence boundaries.
            # 25% probability per eligible boundary.
            with lock: log_step(para_num, "Step 5: Injecting discourse markers...")
            enriched = inject_discourse_markers(surgered)

            # ── STEP 6: Micro-Error Seeding ───────────────────────────────
            # Contractions, em-dashes, sentence-initial "And"/"But".
            # Impossible for detectors to flag without flagging real human text.
            with lock: log_step(para_num, "Step 6: Seeding micro-errors...")
            enriched = seed_micro_errors(enriched)

            # ── STEP 6b: Evasion Layer (optional) ────────────────────────
            if evasion_mode:
                with lock: log_step(para_num, "Step 6b: Evasion layer (ZWS + homoglyphs)...")
                enriched = apply_evasion(enriched)

            # ── STEP 7: Word-Merge Safety Net ─────────────────────────────
            # Strip all invisible characters, fix merged words.
            # Ensures clean readable output before validation.
            enriched = _fix_word_merging(enriched)

            # ── STEP 8: Semantic Similarity Check ─────────────────────────
            # Local SBERT model. Cosine similarity >= 0.55 required.
            # Ensures meaning was preserved — facts, terms, content intact.
            with lock: log_step(para_num, "Step 8: Semantic similarity check (SBERT)...")
            similarity = validator.cosine_similarity(original, enriched)

            if similarity < 0.55:
                with lock:
                    log_step(para_num, f"Step 8: ✗ FAILED — sim={similarity:.2f} < 0.55 — retrying")
                    publish(r, job_id, {"event": "retry", "para": para_num,
                                        "reason": f"Semantic drift ({similarity:.2f})"})
                continue
            else:
                with lock: log_step(para_num, f"Step 8: ✓ PASSED — sim={similarity:.2f}")

            # ── STEP 9: AI Detection Check ─────────────────────────────────
            # Local RoBERTa model. Score <= 0.15 (15%) required.
            # Retries with increasing aggression if score too high.
            with lock: log_step(para_num, "Step 9: AI detection check (RoBERTa)...")
            ai_score = detector.score(enriched)

            if ai_score < best_ai_score:
                best_ai_score  = ai_score
                best_candidate = enriched

            if ai_score <= 0.15:
                with lock: log_step(para_num, f"Step 9: ✓ PASSED — AI score {ai_score:.0%} ✅")
                break
            else:
                with lock:
                    log_step(para_num, f"Step 9: ✗ FAILED — AI score {ai_score:.0%} > 15% — retrying")
                    publish(r, job_id, {"event": "retry", "para": para_num,
                                        "reason": f"AI score {ai_score:.0%}"})

        except Exception as e:
            with lock: log_step(para_num, f"✗ Exception attempt {attempt+1}: {e}")
            continue

    with lock:
        log_para(para_num, total, f"✓ DONE — best AI score: {best_ai_score:.0%}")

    return (para_id, best_candidate)


# ── Main Celery task ──────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="tasks.humanize_document")
def humanize_document(self: Task, input_path, gemini_key, hf_token,
                       job_id, evasion_mode=False):
    """
    Root Celery task. Orchestrates the full parallel humanization pipeline.
    Called when user uploads a DOCX via POST /upload.
    """
    r = redis.from_url(REDIS_URL, decode_responses=True)

    try:
        # ── STAGE 1: Parse DOCX ───────────────────────────────────────────
        # Extract all rewritable paragraphs from the DOCX XML.
        # Freezes tables, images, headings, textboxes — only body text rewritten.
        log("STAGE 1/4 — Parsing DOCX...")
        publish(r, job_id, {"event": "stage", "msg": "Parsing document..."})

        parser        = DocxParser(input_path)
        paragraph_map = parser.extract_paragraphs()
        total         = len(paragraph_map)

        log(f"STAGE 1/4 — DONE. {total} paragraphs found.")
        publish(r, job_id, {"event": "parsed", "total_paras": total})

        # ── STAGE 2: Initialise pipeline ──────────────────────────────────
        # All models use @lru_cache — load once, stay in memory.
        # SBERT ~90MB + RoBERTa ~120MB + PyTorch overhead = ~430MB total.
        log("STAGE 2/4 — Loading pipeline components...")
        humanizer = GeminiHumanizer(api_key=gemini_key)
        validator  = SemanticValidator()
        detector   = AIDetector()
        surgeon    = NLPSurgeon()   # no model loading needed — pure Python NLP

        est_secs = (total / PARALLEL_WORKERS) * 8
        log(f"STAGE 2/4 — DONE. Ready.")
        print(f"\n  ⏱  Est. time: ~{int(est_secs//60)}m {int(est_secs%60)}s "
              f"({PARALLEL_WORKERS} parallel workers)\n", flush=True)

        # ── STAGE 3: Parallel humanization ───────────────────────────────
        # ThreadPoolExecutor submits all paragraphs simultaneously.
        # Lock prevents interleaved terminal output from threads.
        log(f"STAGE 3/4 — Humanizing {total} paragraphs in parallel...")
        publish(r, job_id, {"event": "stage", "msg": f"Humanizing {total} paragraphs..."})

        results   = {}
        lock      = threading.Lock()
        completed = 0

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_single_paragraph,
                    para, idx+1, total,
                    humanizer, validator, detector, surgeon,
                    evasion_mode, r, job_id, lock
                ): para["id"]
                for idx, para in enumerate(paragraph_map)
            }

            for future in as_completed(futures):
                try:
                    para_id, text = future.result()
                    results[para_id] = text
                    completed += 1
                    print(f"\n  ✅ {completed}/{total} paragraphs complete\n", flush=True)
                except Exception as e:
                    para_id = futures[future]
                    original = next((p["text"] for p in paragraph_map if p["id"] == para_id), "")
                    results[para_id] = original
                    completed += 1
                    print(f"\n  ❌ Para {para_id} failed: {e} — kept original\n", flush=True)

        log("STAGE 3/4 — DONE. All paragraphs processed.")

        # ── STAGE 4: Reconstruct DOCX ─────────────────────────────────────
        # Surgically replace text nodes in original XML.
        # Fonts, layout, images, tables = completely untouched.
        log("STAGE 4/4 — Rebuilding DOCX...")
        publish(r, job_id, {"event": "stage", "msg": "Rebuilding document..."})

        out_path = os.path.join(OUTPUT_DIR, f"{job_id}_humanized.docx")
        parser.reconstruct(results, out_path)

        log(f"✅ JOB COMPLETE — /download/{job_id}")
        publish(r, job_id, {
            "event": "done",
            "download_url": f"/download/{job_id}",
            "msg": "Done. Your humanized document is ready.",
        })
        return {"status": "done", "output": out_path}

    except Exception as exc:
        log(f"❌ JOB FAILED — {exc}")
        publish(r, job_id, {"event": "error", "detail": str(exc)})
        raise

    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass
        r.close()