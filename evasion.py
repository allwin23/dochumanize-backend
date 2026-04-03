"""
evasion.py — Evasion layer (FIXED: no more word-merging).

BUG FIXED:
  Previous version inserted Zero-Width Spaces at word BOUNDARIES (between spaces).
  When text was copy-pasted into AI detectors, spaces were lost, merging words:
  "in practice" → "inpractice", "we must" → "wemust"

  Fix: ZWS now inserted only INSIDE long words (mid-word position).
  Mid-word ZWS is ignored by browsers and copy-paste — completely invisible.

Techniques (all fixed and safe):
  1. Mid-word ZWS injection — inside long words only, never at spaces
  2. Soft hyphen injection — inside long words, invisible unless line-breaks
  3. Homoglyph substitution — very low rate (1.5%) to avoid over-detection
"""

import random
import re

# Visually identical Unicode alternatives for Latin characters
HOMOGLYPH_MAP: dict[str, list[str]] = {
    'a': ['а'],   # Cyrillic а (U+0430)
    'c': ['с'],   # Cyrillic с (U+0441)
    'e': ['е'],   # Cyrillic е (U+0435)
    'o': ['о'],   # Cyrillic о (U+043E)
    'p': ['р'],   # Cyrillic р (U+0440)
    'A': ['А'],   # Cyrillic А (U+0410)
    'E': ['Е'],   # Cyrillic Е (U+0415)
    'O': ['О'],   # Cyrillic О (U+041E)
}

ZWS         = "\u200B"   # Zero Width Space — invisible mid-word
SOFT_HYPHEN = "\u00AD"   # Soft Hyphen — invisible unless line wraps


def apply_evasion(text: str) -> str:
    """
    Apply all evasion techniques safely.
    GUARANTEE: No word boundaries are touched — no word merging possible.
    """
    text = _inject_mid_word_zws(text, rate=0.08)
    text = _inject_soft_hyphens(text, min_len=9, rate=0.12)
    text = _substitute_homoglyphs(text, rate=0.015)
    return text


def _inject_mid_word_zws(text: str, rate: float = 0.08) -> str:
    """
    Insert ZWS in the MIDDLE of long words (7+ chars) only.
    NEVER touches spaces or word boundaries.

    Safe because:
    - Mid-word ZWS is ignored by all modern browsers
    - Copy-paste strips ZWS automatically in most contexts
    - Only affects tokenizer n-gram analysis (which is what detectors use)
    """
    def inject(word: str) -> str:
        if (len(word) >= 7
                and not word.isupper()          # skip acronyms
                and not re.search(r'\d', word)  # skip numbers
                and not word.startswith('http') # skip URLs
                and random.random() < rate):
            mid = len(word) // 2
            return word[:mid] + ZWS + word[mid:]
        return word

    # Split on whitespace, process each token, rejoin with original whitespace
    tokens = re.split(r'(\s+)', text)
    return ''.join(inject(t) if not t.isspace() else t for t in tokens)


def _inject_soft_hyphens(text: str, min_len: int = 9, rate: float = 0.12) -> str:
    """
    Insert soft hyphens (U+00AD) inside long words.
    Soft hyphens are completely invisible in rendered text.
    They only appear as a hyphen when the word needs to break at a line end.
    Alters the tokenisation fingerprint without any visual effect.
    """
    def inject(word: str) -> str:
        if (len(word) >= min_len
                and not word.isupper()
                and not re.search(r'\d', word)
                and random.random() < rate):
            mid = len(word) // 2
            return word[:mid] + SOFT_HYPHEN + word[mid:]
        return word

    tokens = re.split(r'(\s+)', text)
    return ''.join(inject(t) if not t.isspace() else t for t in tokens)


def _substitute_homoglyphs(text: str, rate: float = 0.015) -> str:
    """
    Replace a tiny fraction of eligible characters with visually identical
    Unicode alternatives. Rate of 1.5% means roughly 1-2 chars per paragraph.

    Safe because:
    - Characters look completely identical to human readers
    - DOCX stores as Unicode natively so no rendering issues
    - Only affects byte-level tokenizer fingerprinting
    """
    sentence_starts = _get_sentence_starts(text)
    result = list(text)

    for i, ch in enumerate(result):
        if ch in HOMOGLYPH_MAP and i not in sentence_starts:
            if not (ch.isupper() and _in_acronym(text, i)):
                if random.random() < rate:
                    result[i] = random.choice(HOMOGLYPH_MAP[ch])

    return ''.join(result)


def strip_evasion(text: str) -> str:
    """Remove all injected evasion characters. Useful for diffs and debugging."""
    for ch in [ZWS, SOFT_HYPHEN]:
        text = text.replace(ch, '')
    reverse = {v: k for k, vs in HOMOGLYPH_MAP.items() for v in vs}
    return ''.join(reverse.get(c, c) for c in text)


def _get_sentence_starts(text: str) -> set[int]:
    starts = {0}
    for m in re.finditer(r'(?<=[.!?])\s+', text):
        starts.add(m.end())
    return starts


def _in_acronym(text: str, pos: int) -> bool:
    s, e = pos, pos
    while s > 0 and text[s-1].isupper(): s -= 1
    while e < len(text)-1 and text[e+1].isupper(): e += 1
    return (e - s + 1) >= 2