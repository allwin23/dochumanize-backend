"""
cliche_stripper.py — Rule-based pre-processor that removes known AI "tells"
before the text reaches the LLM rewriter.

Strategy:
  1. Exact phrase replacement (most AI clichés are fixed strings)
  2. Regex pattern replacement (handles inflected / varied forms)
  3. Sentence-opening removal (strips AI-typical sentence starters entirely)

This runs BEFORE the Gemini call so the LLM is working with cleaner
input and is less likely to reproduce the same patterns.
"""

import re
from typing import Tuple


# ── 1. Exact phrase → replacement ────────────────────────────────────────────
# Format: (original_phrase, replacement)
# Use empty string "" to delete, or a better phrase to substitute.

EXACT_REPLACEMENTS: list[Tuple[str, str]] = [
    # Classic AI openers
    ("In today's world,", ""),
    ("In today's rapidly evolving world,", ""),
    ("In the modern era,", ""),
    ("In recent years,", ""),
    ("In the digital age,", ""),
    ("It is worth noting that", "Notably,"),
    ("It is important to note that", ""),
    ("It should be noted that", ""),
    ("It is crucial to understand that", ""),
    ("It goes without saying that", ""),
    ("Needless to say,", ""),

    # Overused verbs
    ("delve into", "examine"),
    ("delve deeper", "look more closely"),
    ("delves into", "examines"),
    ("delving into", "examining"),
    ("leverage", "use"),
    ("leveraging", "using"),
    ("leverages", "uses"),
    ("utilize", "use"),
    ("utilizes", "uses"),
    ("utilizing", "using"),
    ("facilitate", "enable"),
    ("facilitates", "enables"),
    ("facilitate the process", "help"),
    ("foster", "build"),
    ("fosters", "builds"),
    ("cultivate", "develop"),
    ("cultivates", "develops"),
    ("underscore", "highlight"),
    ("underscores", "highlights"),

    # Overused adjectives / adverbs
    ("robust", "strong"),
    ("cutting-edge", "advanced"),
    ("state-of-the-art", "advanced"),
    ("seamless", "smooth"),
    ("seamlessly", "smoothly"),
    ("groundbreaking", "significant"),
    ("revolutionary", "major"),
    ("transformative", "important"),
    ("pivotal", "key"),
    ("crucial", "important"),
    ("vital", "important"),
    ("paramount", "essential"),
    ("comprehensive", "thorough"),
    ("holistic", "overall"),

    # Filler phrases
    ("As mentioned earlier,", ""),
    ("As previously discussed,", ""),
    ("As we can see,", ""),
    ("As outlined above,", ""),
    ("Based on the above,", ""),
    ("In light of the above,", ""),
    ("With this in mind,", ""),
    ("Taking everything into account,", ""),
    ("All in all,", ""),
    ("In conclusion, it is evident that", ""),
    ("In summary,", ""),
    ("To summarize,", ""),
    ("To conclude,", ""),

    # Academic AI padding
    ("plays a crucial role in", "is central to"),
    ("plays a pivotal role in", "is key to"),
    ("plays an important role in", "matters for"),
    ("plays a significant role in", "shapes"),
    ("a wide range of", "many"),
    ("a wide variety of", "many"),
    ("a plethora of", "many"),
    ("a myriad of", "many"),
    ("numerous", "many"),
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("at this point in time", "now"),
    ("in the event that", "if"),
    ("for the purpose of", "to"),
    ("with the intention of", "to"),
    ("on a regular basis", "regularly"),
    ("in a timely manner", "promptly"),

    # AI hedging / meta-commentary
    ("This essay will explore", ""),
    ("This paper will discuss", ""),
    ("This report aims to", ""),
    ("The purpose of this section is to", ""),
    ("The following section will", ""),
]

# ── 2. Regex replacements ─────────────────────────────────────────────────────
# Format: (pattern, replacement)

REGEX_REPLACEMENTS: list[Tuple[str, str]] = [
    # "By [verb]ing ..., we can ..." → remove the "we can" scaffolding
    (r"\bwe can\b", ""),
    # "it is X that" constructions used as filler
    (r"\bit is (clear|evident|obvious|apparent) that\b", ""),
    # Phrases like "In today's X world" or "In the X age"
    (r"\bIn today's\s+\w+\s+world[,.]?\b", ""),
    # Remove trailing "in conclusion" if standalone sentence
    (r"(?i)^(in conclusion|to summarize|in summary)[,.]?\s*", ""),
    # "plays a [adj] role in" generalisation
    (r"\bplays\s+a[n]?\s+\w+\s+role\s+in\b", "is central to"),
    # "key [noun]" redundancy in certain contexts
    (r"\bkey\s+(factor|aspect|element|component|area)\b", r"important \1"),
    # Remove AI signposting phrases at sentence start
    (r"(?i)^(notably|importantly|significantly|essentially|fundamentally),\s*", ""),
]

# ── 3. Sentence-opening patterns to strip entirely ───────────────────────────
SENTENCE_OPENERS_TO_STRIP: list[str] = [
    r"(?i)^as we (?:can |may )?(?:see|observe|note)[,.]?\s*",
    r"(?i)^as (?:mentioned|discussed|noted|described|explained) (?:above|earlier|previously)[,.]?\s*",
    r"(?i)^it is (?:worth|important) (?:noting|mentioning|emphasizing) that\s*",
    r"(?i)^overall[,.]?\s*",
    r"(?i)^clearly[,.]?\s*",
    r"(?i)^obviously[,.]?\s*",
    r"(?i)^evidently[,.]?\s*",
]


def strip_cliches(text: str) -> str:
    """
    Apply all three stripping strategies in sequence.
    Returns cleaned text ready for the LLM rewriter.
    """
    result = text

    # Step 1: Exact phrase replacements (case-sensitive for precision)
    for phrase, replacement in EXACT_REPLACEMENTS:
        result = result.replace(phrase, replacement)

        # Also handle capitalized version at sentence start
        cap_phrase = phrase[0].upper() + phrase[1:] if phrase else phrase
        cap_repl = replacement[0].upper() + replacement[1:] if replacement else replacement
        result = result.replace(cap_phrase, cap_repl)

    # Step 2: Regex replacements
    for pattern, replacement in REGEX_REPLACEMENTS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Step 3: Sentence-opening strips
    # Split into sentences, strip openers from each, rejoin
    sentences = re.split(r'(?<=[.!?])\s+', result.strip())
    cleaned_sentences = []
    for sentence in sentences:
        s = sentence.strip()
        for opener_pattern in SENTENCE_OPENERS_TO_STRIP:
            s = re.sub(opener_pattern, "", s)
        s = s.strip()
        if s:
            # Ensure sentence starts with a capital letter after stripping
            s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
            cleaned_sentences.append(s)

    result = " ".join(cleaned_sentences)

    # Final cleanup: collapse multiple spaces
    result = re.sub(r"  +", " ", result).strip()

    return result