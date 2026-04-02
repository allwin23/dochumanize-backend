"""
humanizer.py — Two-pass Gemini rewriter.

Pass 1 — Structural Rewrite:
  Focuses on syntactic skeleton transformation:
  passive↔active voice, nominalization→verb-led constructions,
  cleft sentences, fronted adverbials.
  Enforces burstiness via explicit sentence-length pattern.
  Strict ±5% character count constraint.

Pass 2 — Conversational Degradation:
  Takes Pass 1 output and deliberately roughens it:
  adds hedging language, first-person-adjacent phrasing,
  informal discourse connectives, controlled imperfections.
  Makes the prose sound like an expert human, not a polished AI.
"""

import os
import re
import time
import google.generativeai as genai
from typing import Optional


# Burstiness seed patterns — injected into the prompt to hint at rhythm variety
BURSTINESS_PATTERNS = [
    "long, short, long, medium, short",
    "medium, short, long, short, long, medium",
    "long, medium, short, long, short",
    "short, long, medium, short, long",
]


def _char_window(original_len: int, tolerance: float = 0.05) -> tuple[int, int]:
    """Return (min_chars, max_chars) for the ±5% constraint."""
    delta = int(original_len * tolerance)
    return (original_len - delta, original_len + delta)


def _build_pass1_prompt(text: str, original_char_count: int, attempt: int) -> str:
    """
    Constructs the Pass 1 structural rewrite prompt.
    Attempt number increases aggression on retries.
    """
    min_c, max_c = _char_window(original_char_count)
    pattern = BURSTINESS_PATTERNS[attempt % len(BURSTINESS_PATTERNS)]

    # Aggression scale: higher attempt = more deviation from AI baseline
    aggression_note = ""
    if attempt >= 2:
        aggression_note = (
            "\n- IMPORTANT: The previous rewrite was still detected as AI. "
            "Be significantly more unconventional. Use unusual sentence openers. "
            "Use a rhetorical question if natural. Vary vocabulary aggressively."
        )

    return f"""You are a professional academic editor transforming AI-generated text into authentic human writing.

## HARD RULES (never break these):
1. Output ONLY the rewritten paragraph — no preamble, no commentary, no quotes.
2. The output MUST be between {min_c} and {max_c} characters. Count carefully.
3. Preserve ALL technical terms, proper nouns, acronyms, and factual claims exactly.
4. Do NOT add new facts or remove existing ones.

## STRUCTURAL TRANSFORMATION RULES (apply all):
- Transform passive voice to active voice wherever natural.
- Convert nominalized phrases back to verb-led constructions (e.g., "the implementation of" → "implementing").
- Use at least one cleft sentence structure ("It is X that..." or "What makes this significant is...").
- Use at least one fronted adverbial ("Crucially, ...", "In practice, ...", "For this reason, ...").
- Vary sentence length in this exact rhythm pattern: {pattern}.
- Replace any Latinate vocabulary with Anglo-Saxon equivalents where possible without losing precision.{aggression_note}

## INPUT PARAGRAPH:
{text}

## REWRITTEN PARAGRAPH:"""


def _build_pass2_prompt(text: str, attempt: int) -> str:
    """
    Constructs the Pass 2 conversational degradation prompt.
    Takes the structurally rewritten text and makes it sound more human.
    """
    return f"""You are editing academic writing to sound like a knowledgeable human expert — not a polished AI assistant.

## HARD RULES:
1. Output ONLY the final paragraph — no preamble, commentary, or quotes.
2. Do NOT change the meaning, facts, or technical terminology.
3. The character count must stay within ±8% of the input.

## HUMANIZATION RULES (apply selectively, not all at once):
- Add ONE discourse connective that shows reasoning: e.g., "That said,", "The trouble is,", "What this means, in practice,", "Curiously,", "Worth noting here is that...".
- Replace one instance of a formal phrase with a slightly more direct equivalent.
- It is acceptable (and encouraged) to start one sentence with "And" or "But" for rhythm.
- Add one mild hedge if natural: "arguably", "in most cases", "to some extent".
- One sentence may be a deliberate fragment for rhythm, if it reads naturally.
- Do NOT make the text sound casual or unprofessional — it must remain academic.

## INPUT PARAGRAPH:
{text}

## HUMANIZED PARAGRAPH:"""


class GeminiHumanizer:
    """
    Handles both rewrite passes using the Google Gemini API.
    Uses gemini-1.5-flash for speed and cost efficiency.
    Implements exponential backoff on rate-limit errors.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.85,      # High enough for creativity, not too random
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )

    def _call_with_backoff(self, prompt: str, max_attempts: int = 3) -> Optional[str]:
        """Call Gemini with exponential backoff for rate-limit / quota errors."""
        for i in range(max_attempts):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                )
                if response.text:
                    return response.text.strip()
                return None
            except Exception as e:
                err = str(e).lower()
                if "quota" in err or "rate" in err or "429" in err:
                    wait = (2 ** i) * 5    # 5s, 10s, 20s
                    time.sleep(wait)
                else:
                    raise
        return None

    def rewrite_structural(
        self,
        text: str,
        original_char_count: int,
        attempt: int = 0,
    ) -> Optional[str]:
        """
        Pass 1: Structural transformation.
        Returns rewritten text or None on failure.
        """
        prompt = _build_pass1_prompt(text, original_char_count, attempt)
        result = self._call_with_backoff(prompt)

        if result:
            # Strip any accidental markdown formatting the model might add
            result = re.sub(r"^```[a-z]*\n?|```$", "", result, flags=re.MULTILINE).strip()

        return result

    def rewrite_conversational(
        self,
        text: str,
        attempt: int = 0,
    ) -> Optional[str]:
        """
        Pass 2: Conversational degradation.
        Returns humanized text or None on failure.
        """
        prompt = _build_pass2_prompt(text, attempt)
        result = self._call_with_backoff(prompt)

        if result:
            result = re.sub(r"^```[a-z]*\n?|```$", "", result, flags=re.MULTILINE).strip()

        return result