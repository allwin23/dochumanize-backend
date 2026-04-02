"""
evasion.py — Aggressive evasion layer (optional toggle).

WARNING: This module applies techniques specifically designed to confuse
AI-detection tools at the character level. It should only be enabled
via the evasion_mode toggle by users who explicitly request it.

Techniques:
1. Zero-Width Space (ZWS) injection:
   Inserts U+200B (ZERO WIDTH SPACE) at word boundaries in selected positions.
   The text renders identically to the human eye but breaks the tokenization
   patterns that AI detectors rely on for n-gram analysis.

2. Homoglyph substitution:
   Replaces a small percentage of Latin characters with visually identical
   Unicode counterparts from Cyrillic, Greek, or other scripts.
   e.g. Latin 'a' (U+0061) → Cyrillic 'а' (U+0430)
   The document looks identical but the underlying character codes differ.

3. Soft hyphen injection:
   Inserts U+00AD (SOFT HYPHEN) inside long words. Soft hyphens are invisible
   unless the word needs to break at a line end, so they're visually harmless
   but alter the tokenization fingerprint.

Note on DOCX compatibility:
   ZWS and soft hyphens survive .docx save/load via python-docx.
   Homoglyphs survive as-is since DOCX is Unicode-native.
   All three techniques are stripped automatically by most copy-paste
   operations into plain text environments.
"""

import random
import re


# ── Homoglyph Map ─────────────────────────────────────────────────────────────
# Latin → visually identical Unicode alternatives.
# Only includes characters with near-perfect visual match.

HOMOGLYPH_MAP: dict[str, list[str]] = {
    'a': ['а'],         # Cyrillic а (U+0430)
    'c': ['с'],         # Cyrillic с (U+0441)
    'e': ['е'],         # Cyrillic е (U+0435)
    'o': ['о'],         # Cyrillic о (U+043E)
    'p': ['р'],         # Cyrillic р (U+0440)
    'x': ['х'],         # Cyrillic х (U+0445)
    'y': ['у'],         # Cyrillic у (U+0443)
    'i': ['і'],         # Cyrillic і (U+0456) — Ukrainian
    'A': ['А'],         # Cyrillic А (U+0410)
    'B': ['В'],         # Cyrillic В (U+0412)
    'C': ['С'],         # Cyrillic С (U+0421)
    'E': ['Е'],         # Cyrillic Е (U+0415)
    'H': ['Н'],         # Cyrillic Н (U+041D)
    'K': ['К'],         # Cyrillic К (U+041A)
    'M': ['М'],         # Cyrillic М (U+041C)
    'O': ['О'],         # Cyrillic О (U+041E)
    'P': ['Р'],         # Cyrillic Р (U+0420)
    'T': ['Т'],         # Cyrillic Т (U+0422)
    'X': ['Х'],         # Cyrillic Х (U+0425)
}

# Unicode invisible/zero-width characters
ZWS = "\u200B"          # Zero Width Space
SOFT_HYPHEN = "\u00AD"  # Soft Hyphen
ZWJ = "\u200D"          # Zero Width Joiner (occasional use)


def apply_evasion(text: str) -> str:
    """
    Master evasion function. Applies all three techniques with
    conservative probabilities to avoid visible artefacts.
    """
    text = _inject_zero_width_spaces(text, rate=0.08)
    text = _substitute_homoglyphs(text, rate=0.03)
    text = _inject_soft_hyphens(text, min_word_length=8, rate=0.10)
    return text


def _inject_zero_width_spaces(text: str, rate: float = 0.08) -> str:
    """
    Insert ZWS between words at a specified rate.
    rate=0.08 means ~8% of word boundaries get a ZWS injected.

    Avoids injecting inside URLs, emails, or technical terms (all-caps).
    """
    words = text.split(" ")
    result_parts = []

    for i, word in enumerate(words):
        result_parts.append(word)

        # Don't inject before last word, inside URLs, or all-caps acronyms
        if i < len(words) - 1:
            next_word = words[i + 1]
            is_url = word.startswith(("http", "www", "ftp"))
            is_acronym = word.isupper() and len(word) <= 6
            is_email = "@" in word

            if not (is_url or is_acronym or is_email) and random.random() < rate:
                result_parts.append(ZWS)
            else:
                result_parts.append(" ")
        # Note: we don't add a space after the last word

    # Reconstruct: join parts (spaces are already included in result_parts)
    return "".join(result_parts)


def _substitute_homoglyphs(text: str, rate: float = 0.03) -> str:
    """
    Replace a small fraction of eligible characters with their homoglyphs.
    rate=0.03 means ~3% of eligible characters get substituted.

    Avoids substituting in:
    - Technical terms (all-caps)
    - Numbers
    - The first character of a sentence (would look odd if monitored carefully)
    """
    result = list(text)
    sentence_starts = _get_sentence_start_positions(text)

    for i, char in enumerate(result):
        if char in HOMOGLYPH_MAP:
            # Don't touch sentence-starting characters
            if i in sentence_starts:
                continue
            # Don't touch characters in all-caps sequences (acronyms)
            if char.isupper() and _is_in_acronym(text, i):
                continue

            if random.random() < rate:
                result[i] = random.choice(HOMOGLYPH_MAP[char])

    return "".join(result)


def _inject_soft_hyphens(text: str, min_word_length: int = 8, rate: float = 0.10) -> str:
    """
    Insert soft hyphens (U+00AD) inside long words.
    The soft hyphen is invisible unless the word breaks at a line end.

    Inserts roughly in the middle of eligible long words.
    """
    def add_soft_hyphen(word: str) -> str:
        if len(word) >= min_word_length and random.random() < rate:
            mid = len(word) // 2
            return word[:mid] + SOFT_HYPHEN + word[mid:]
        return word

    # Process word-by-word, preserving punctuation attachment
    words = re.split(r'(\s+)', text)
    return "".join(
        add_soft_hyphen(token) if re.match(r'\w{8,}', token) else token
        for token in words
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_sentence_start_positions(text: str) -> set[int]:
    """Return the character positions of the first letter of each sentence."""
    positions = {0}
    for m in re.finditer(r'(?<=[.!?])\s+', text):
        pos = m.end()
        if pos < len(text):
            positions.add(pos)
    return positions


def _is_in_acronym(text: str, pos: int) -> bool:
    """Check if character at pos is part of a consecutive all-caps sequence (acronym)."""
    start = pos
    end = pos
    while start > 0 and text[start - 1].isupper():
        start -= 1
    while end < len(text) - 1 and text[end + 1].isupper():
        end += 1
    acronym_length = end - start + 1
    return acronym_length >= 2


def strip_evasion(text: str) -> str:
    """
    Utility: remove all injected evasion characters from text.
    Useful for debugging or when producing a clean plain-text diff.
    """
    # Remove ZWS, soft hyphens, ZWJ
    for char in [ZWS, SOFT_HYPHEN, ZWJ]:
        text = text.replace(char, "")

    # Replace homoglyphs back to Latin equivalents
    reverse_map = {v: k for k, variants in HOMOGLYPH_MAP.items() for v in variants}
    result = []
    for char in text:
        result.append(reverse_map.get(char, char))

    return "".join(result)