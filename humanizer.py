"""
humanizer.py — Gemini Paraphraser (plagiarism avoidance only).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE CHANGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gemini's job is NO LONGER to humanize text.
It was failing at that — LLMs cannot escape their own output distribution.

Gemini's new job:
  1. PARAPHRASE — change enough words for plagiarism avoidance
  2. PRESERVE MEANING — never hallucinate or omit facts
  3. STAY NEUTRAL — do not try to "sound human", just rephrase

ALL humanization (perplexity, burstiness, syntax) is handled by:
  → nlp_surgeon.py (SpaCy + WordNet + NLTK offline pipeline)

This separation gives us two independent levers:
  Gemini  → plagiarism score < 15%
  Surgeon → AI detection score < 5%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY ONE PASS NOW (not two):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two Gemini passes gave two chances for AI patterns to compound.
One focused paraphrase pass + heavy NLP surgery = better result.
Also cuts processing time roughly in half.
"""

import re
import time
from typing import Optional

from google import genai
from google.genai import types


def _build_paraphrase_prompt(text: str, char_count: int, attempt: int) -> str:
    """
    Minimal, focused paraphrase prompt.
    Gemini is instructed NOT to try to humanize — just rephrase.
    The surgeon handles all humanization after this.

    Attempt number increases paraphrase distance on retries.
    """
    min_c = int(char_count * 0.88)
    max_c = int(char_count * 1.12)

    distance = [
        "Rephrase moderately — change at least 60% of non-technical words.",
        "Rephrase substantially — use completely different sentence structures.",
        "Rephrase aggressively — virtually no phrase should remain identical.",
    ][min(attempt, 2)]

    return f"""Rephrase the academic paragraph below for plagiarism avoidance.

RULES:
1. Output ONLY the rephrased paragraph. No labels, no commentary.
2. Output length: {min_c}–{max_c} characters.
3. Preserve ALL facts, data, technical terms, proper nouns exactly.
4. Do NOT add or remove information.
5. Do NOT try to "humanize" or make it sound casual — just rephrase.
6. Do NOT use these overused phrases: "it is worth noting", "it is important to note",
   "plays a crucial role", "in today's world", "delve into", "leverage", "utilize",
   "furthermore", "moreover", "additionally", "groundbreaking", "revolutionary",
   "seamless", "robust", "cutting-edge", "holistic", "paramount", "pivotal".
7. {distance}

INPUT:
{text}

REPHRASED:"""


class GeminiHumanizer:
    """
    Gemini paraphraser — handles plagiarism avoidance only.
    All humanization is done by NLPSurgeon after this step.

    Uses gemini-2.0-flash for speed (~2-3s per call).
    Single pass only — faster, less API quota used.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client     = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.config     = types.GenerateContentConfig(
            temperature=0.75,    # Lower temperature = more faithful paraphrase
            top_p=0.92,
            top_k=40,
            max_output_tokens=2048,
        )

    def _call(self, prompt: str, retries: int = 3) -> Optional[str]:
        """Call Gemini with backoff. Strip any markdown artifacts."""
        for i in range(retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.config,
                )
                if resp.text:
                    text = resp.text.strip()
                    # Strip markdown fences
                    text = re.sub(r'^```[a-z]*\n?|```$', '', text, flags=re.MULTILINE).strip()
                    # Strip label prefixes model occasionally adds
                    text = re.sub(r'^(REPHRASED|OUTPUT|PARAGRAPH):?\s*', '', text, flags=re.IGNORECASE).strip()
                    return text
                return None
            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ['quota', 'rate', '429', 'limit']):
                    time.sleep((2 ** i) * 5)
                else:
                    raise
        return None

    def paraphrase(self, text: str, original_char_count: int, attempt: int = 0) -> Optional[str]:
        """
        Single-pass paraphrase for plagiarism avoidance.
        Returns rephrased text or None on failure.
        """
        prompt = _build_paraphrase_prompt(text, original_char_count, attempt)
        return self._call(prompt)

    # Keep these method names for backward compatibility with tasks.py
    def rewrite_structural(self, text: str, original_char_count: int, attempt: int = 0) -> Optional[str]:
        """Alias for paraphrase() — keeps tasks.py compatible."""
        return self.paraphrase(text, original_char_count, attempt)

    def rewrite_conversational(self, text: str, attempt: int = 0) -> Optional[str]:
        """
        Second pass is now a no-op — surgeon handles humanization.
        Returns text unchanged so pipeline stays compatible.
        """
        return text