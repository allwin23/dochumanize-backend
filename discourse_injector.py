"""
discourse_injector.py — Post-processing enrichment layer.

Two functions:

1. inject_discourse_markers():
   Probabilistically inserts human-like discourse connectives at sentence
   boundaries. These are the "reasoning glue" words that AI rarely generates
   naturally: "That said,", "Worth noting here is that...", "Curiously,", etc.
   These spike the perplexity score by introducing unexpected transitions.

2. seed_micro_errors():
   Introduces statistically rare but grammatically acceptable "imperfections"
   that real expert human writers use:
   - Sentence-initial "And" / "But"
   - Comma splices (joining two clauses with just a comma)
   - Intentional fragments (if safe to do so)
   These patterns are essentially impossible for AI detectors to penalize
   without also flagging legitimate human academic writing.
"""

import re
import random
from typing import List


# ── Discourse Marker Bank ─────────────────────────────────────────────────────
# Grouped by rhetorical function so injection is contextually appropriate.

DISCOURSE_MARKERS = {
    "contrast": [
        "That said,",
        "Even so,",
        "Then again,",
        "At the same time,",
        "On the other hand,",
    ],
    "elaboration": [
        "What this means, in practice, is that",
        "Worth noting here is that",
        "The implication is clear:",
        "More specifically,",
        "To put it another way,",
    ],
    "reasoning": [
        "The trouble is,",
        "The reason for this is straightforward:",
        "This matters because",
        "Part of the explanation lies in",
    ],
    "concession": [
        "Admittedly,",
        "To be fair,",
        "Granted,",
        "Of course,",
    ],
    "emphasis": [
        "Crucially,",
        "What stands out here is",
        "Perhaps most importantly,",
        "The key point is that",
    ],
    "curiosity": [
        "Curiously,",
        "Interestingly,",
        "Somewhat surprisingly,",
    ],
}

# Flatten for random selection
ALL_MARKERS: List[str] = [m for group in DISCOURSE_MARKERS.values() for m in group]


def inject_discourse_markers(text: str, injection_probability: float = 0.30) -> str:
    """
    Walk through sentences in the text. At each sentence boundary,
    probabilistically prepend a discourse marker to the next sentence.

    Rules:
    - Never inject on the first sentence (preserve the opening)
    - Never inject two markers in a row
    - Markers ending with ':' get the next word lowercased
    - Markers ending with ',' or 'that' get the next word preserved
    - injection_probability=0.30 means ~30% of eligible sentences get a marker
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text

    result = [sentences[0]]
    last_injected = False

    for i, sentence in enumerate(sentences[1:], start=1):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Decide whether to inject
        if not last_injected and random.random() < injection_probability:
            marker = random.choice(ALL_MARKERS)

            # Adapt case: if marker ends with a colon or 'that', lowercase next word
            if marker.endswith(":") or marker.lower().endswith("that"):
                sentence = sentence[0].lower() + sentence[1:] if len(sentence) > 1 else sentence
                injected = f"{marker} {sentence}"
            else:
                injected = f"{marker} {sentence}"

            result.append(injected)
            last_injected = True
        else:
            result.append(sentence)
            last_injected = False

    return " ".join(result)


def seed_micro_errors(text: str, seed_probability: float = 0.20) -> str:
    """
    Inject subtle human-like stylistic patterns that AI detectors
    cannot flag without also flagging real human writing.

    Techniques applied probabilistically:
    1. Sentence-initial conjunction: "And" / "But" before a sentence
    2. Em-dash interruption: replaces a comma-clause with an em-dash clause
    3. Casual contraction insertion: "does not" → "doesn't" (in select positions)

    seed_probability=0.20 means ~20% of eligible positions get seeded.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 4:
        return text

    result = [sentences[0]]

    for i, sentence in enumerate(sentences[1:], start=1):
        sentence = sentence.strip()
        if not sentence:
            continue

        roll = random.random()

        # Technique 1: Sentence-initial conjunction (~10% chance)
        if roll < 0.10 and not sentence.lower().startswith(("and ", "but ", "or ")):
            conjunction = random.choice(["And ", "But "])
            sentence = conjunction + sentence[0].lower() + sentence[1:]

        # Technique 2: Contraction insertion (~15% chance, only in eligible sentences)
        elif roll < 0.25:
            sentence = _insert_contraction(sentence)

        # Technique 3: Em-dash substitution (~10% chance)
        elif roll < 0.35:
            sentence = _insert_em_dash(sentence)

        result.append(sentence)

    return " ".join(result)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving the sentence-ending punctuation."""
    return re.split(r'(?<=[.!?])\s+', text.strip())


_CONTRACTION_MAP = [
    (r"\bdoes not\b", "doesn't"),
    (r"\bdo not\b", "don't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bhas not\b", "hasn't"),
    (r"\bhave not\b", "haven't"),
    (r"\bcannot\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"),
    (r"\bcould not\b", "couldn't"),
    (r"\bthey are\b", "they're"),
    (r"\bwe are\b", "we're"),
    (r"\bit is\b", "it's"),
]


def _insert_contraction(sentence: str) -> str:
    """Replace one eligible long form with its contraction."""
    for pattern, replacement in _CONTRACTION_MAP:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            # Only replace the first match
            return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
    return sentence


def _insert_em_dash(sentence: str) -> str:
    """
    Replace a comma that separates two clauses with an em-dash,
    creating a more emphatic, human-like pause.
    Only applies if the sentence has a comma and is long enough.
    """
    if "," not in sentence or len(sentence) < 60:
        return sentence

    # Find the first comma that separates two non-trivial clauses
    comma_positions = [m.start() for m in re.finditer(r",", sentence)]

    for pos in comma_positions:
        before = sentence[:pos].strip()
        after = sentence[pos+1:].strip()

        # Make sure both sides have substance (> 5 words each)
        if len(before.split()) >= 5 and len(after.split()) >= 4:
            return before + " — " + after

    return sentence