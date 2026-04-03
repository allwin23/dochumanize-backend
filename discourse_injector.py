"""
discourse_injector.py — Post-processing humanization layer.

TWO FUNCTIONS:

1. inject_discourse_markers()
   Inserts human reasoning connectives at sentence boundaries.
   These are the "thinking glue" words that AI almost never generates:
   "That said,", "The trouble is,", "Worth noting here is that", etc.

   WHY THIS WORKS:
   AI detectors measure how "surprising" each word is (perplexity).
   Words like "That said," at a sentence boundary are statistically
   rare and unexpected — they spike perplexity scores significantly.

   SAFE INJECTION:
   - Never injects on the first sentence (preserve the opening)
   - Never injects two markers in a row
   - Ensures proper spacing — no word merging possible
   - 25% probability per eligible boundary

2. seed_micro_errors()
   Injects subtle human stylistic patterns that real expert writers use.
   These patterns are impossible for AI detectors to penalise without
   also flagging large amounts of legitimate human academic writing.

   Techniques:
   - Contractions (does not → doesn't) — makes text feel less robotic
   - Em-dash interruptions — replaces comma pauses with dramatic dashes
   - Sentence-initial "And" / "But" — extremely common in human prose
"""

import re
import random
from typing import List


# ── Discourse marker bank ─────────────────────────────────────────────────────
# Grouped by rhetorical function for contextually appropriate injection.
# Each group serves a different logical purpose in academic prose.

MARKERS_CONTRAST = [
    "That said,",
    "Even so,",
    "Then again,",
    "At the same time,",
]

MARKERS_ELABORATION = [
    "What this means, in practice, is that",
    "Worth noting here is that",
    "The implication is clear:",
    "More precisely,",
    "To put it plainly,",
]

MARKERS_REASONING = [
    "The trouble is,",
    "This matters because",
    "Part of the reason is that",
    "The explanation is fairly straightforward:",
]

MARKERS_CONCESSION = [
    "Admittedly,",
    "To be fair,",
    "Granted,",
]

MARKERS_EMPHASIS = [
    "Crucially,",
    "What stands out is",
    "Perhaps most telling is that",
]

MARKERS_CURIOSITY = [
    "Curiously,",
    "Interestingly,",
    "Somewhat surprisingly,",
]

ALL_MARKERS: List[str] = (
    MARKERS_CONTRAST
    + MARKERS_ELABORATION
    + MARKERS_REASONING
    + MARKERS_CONCESSION
    + MARKERS_EMPHASIS
    + MARKERS_CURIOSITY
)


def inject_discourse_markers(text: str, probability: float = 0.25) -> str:
    """
    Walk sentence boundaries and probabilistically prepend discourse markers.

    SAFE: always inserts a clean space after the marker before the next sentence.
    No word merging possible — the marker is a complete prepended phrase.

    Args:
        text        — input paragraph text
        probability — chance of injection per eligible boundary (default 25%)

    Returns:
        Text with discourse markers injected at sentence boundaries.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        # Too short to inject — return as-is
        return text

    result = [sentences[0]]
    last_injected = False

    for sentence in sentences[1:]:
        sentence = sentence.strip()
        if not sentence:
            continue

        should_inject = (
            not last_injected               # never two in a row
            and random.random() < probability
        )

        if should_inject:
            marker = random.choice(ALL_MARKERS)

            # Markers ending with ":" or "that" get the next word lowercased
            # to form a grammatically correct continuation
            if marker.endswith(":") or marker.lower().endswith("that"):
                sentence = sentence[0].lower() + sentence[1:] if len(sentence) > 1 else sentence
                # Ensure single clean space between marker and sentence
                injected = f"{marker} {sentence}"
            else:
                # Ensure sentence still starts with capital after marker
                injected = f"{marker} {sentence}"

            result.append(injected)
            last_injected = True
        else:
            result.append(sentence)
            last_injected = False

    # Join with single space — CRITICAL: never double-space, never merge words
    return " ".join(result)


def seed_micro_errors(text: str, probability: float = 0.20) -> str:
    """
    Inject subtle human stylistic patterns at sentence level.

    Applies at most ONE technique per sentence to avoid over-humanizing
    (too many micro-errors can itself be a detection signal).

    Techniques applied probabilistically:
      - Sentence-initial conjunction (And / But) — 10% of sentences
      - Contraction insertion — 15% of sentences
      - Em-dash substitution — 10% of sentences

    Args:
        text        — input paragraph (post discourse injection)
        probability — overall probability of any seeding per sentence

    Returns:
        Text with micro-errors seeded.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 4:
        return text

    result = [sentences[0]]   # never touch the first sentence

    for sentence in sentences[1:]:
        sentence = sentence.strip()
        if not sentence:
            continue

        roll = random.random()

        if roll < 0.10:
            # Technique 1: sentence-initial conjunction
            # "AI is powerful. It changes industries." →
            # "AI is powerful. And it changes industries."
            if not sentence.lower().startswith(("and ", "but ", "or ", "yet ")):
                conj = random.choice(["And ", "But ", "Yet "])
                sentence = conj + sentence[0].lower() + sentence[1:]

        elif roll < 0.25:
            # Technique 2: contraction
            # "does not" → "doesn't", "it is" → "it's"
            sentence = _apply_contraction(sentence)

        elif roll < 0.35:
            # Technique 3: em-dash interruption
            # Replace a comma separating two substantial clauses with em-dash
            sentence = _apply_em_dash(sentence)

        result.append(sentence)

    return " ".join(result)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at .!? boundaries."""
    return re.split(r'(?<=[.!?])\s+', text.strip())


_CONTRACTIONS = [
    (r'\bdoes not\b', "doesn't"),
    (r'\bdo not\b',   "don't"),
    (r'\bis not\b',   "isn't"),
    (r'\bare not\b',  "aren't"),
    (r'\bwas not\b',  "wasn't"),
    (r'\bhas not\b',  "hasn't"),
    (r'\bhave not\b', "haven't"),
    (r'\bwill not\b', "won't"),
    (r'\bwould not\b',"wouldn't"),
    (r'\bshould not\b',"shouldn't"),
    (r'\bcould not\b', "couldn't"),
    (r'\bcannot\b',   "can't"),
    (r'\bthey are\b', "they're"),
    (r'\bwe are\b',   "we're"),
    (r'\bit is\b',    "it's"),
    (r'\bthat is\b',  "that's"),
]


def _apply_contraction(sentence: str) -> str:
    """Replace the first eligible long form with its contraction."""
    for pattern, replacement in _CONTRACTIONS:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
    return sentence


def _apply_em_dash(sentence: str) -> str:
    """
    Replace a comma separating two substantial clauses with an em-dash.
    Only applies if both sides of the comma have 5+ words.
    Creates a more dramatic, human-like pause.

    Example:
      "The results were clear, suggesting that further research was needed."
      → "The results were clear — suggesting that further research was needed."
    """
    if "," not in sentence or len(sentence) < 50:
        return sentence

    positions = [m.start() for m in re.finditer(r",", sentence)]
    for pos in positions:
        before = sentence[:pos].strip()
        after  = sentence[pos+1:].strip()
        if len(before.split()) >= 5 and len(after.split()) >= 4:
            # Use proper em-dash with spaces (—) not hyphen
            return before + " — " + after
    return sentence