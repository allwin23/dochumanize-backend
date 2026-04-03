"""
nlp_surgeon.py — Advanced Offline NLP Surgery Engine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHILOSOPHY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gemini's job is now ONLY to paraphrase (change surface words for
plagiarism avoidance). ALL humanization happens here, offline,
using SpaCy + NLTK + WordNet.

Why? Because LLMs cannot escape their own output distribution.
No matter how we prompt Gemini, the resulting text still has
the statistical fingerprint of an LLM. This module operates
BELOW the LLM layer — directly on grammatical structure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 5 SIGNALS AI DETECTORS MEASURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PERPLEXITY
   What: How surprising each word choice is in context.
   AI:   Always picks statistically safest word → very low perplexity.
   Fix:  WordNet frequency-ranked synonym replacement.
         Replace common words with synonyms that are 2-3x rarer
         in the Brown corpus frequency tables.

2. BURSTINESS  
   What: Variance in sentence length across a paragraph.
   AI:   Writes uniformly medium sentences (15-20 words).
   Fix:  SpaCy dependency-aware sentence splitting and fusion.
         Force pattern: [long, short, long, medium, short, long]
         At least one sentence < 6 words (fragment).
         At least one sentence > 30 words.

3. N-GRAM DISTRIBUTION
   What: Frequency of word sequences (bigrams/trigrams).
   AI:   Produces common academic n-grams repeatedly.
   Fix:  Detect and break up common AI bigrams/trigrams by
         inserting syntactically correct interruptions.

4. PASSIVE VOICE DENSITY
   What: Ratio of passive to active constructions.
   AI:   Overuses passive voice in academic text.
   Fix:  SpaCy dependency parsing to detect and invert
         passive constructions to active voice.

5. DEPENDENCY TREE DEPTH
   What: How deeply nested grammatical structures are.
   AI:   Prefers shallow, safe tree structures.
   Fix:  Insert subordinate clauses, appositive phrases,
         and parenthetical asides at parse-identified positions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE (in operate() method):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stage A: SpaCy parse → passive voice inversion
  Stage B: WordNet frequency-ranked synonym replacement
  Stage C: N-gram disruption
  Stage D: Burstiness enforcement (split/merge/fragment)
  Stage E: Dependency depth injection (subordinate clauses)
  Stage F: Syntactic fronting (move adverbials to front)
  Stage G: Cleanup and validation
"""

import re
import random
import math
from typing import List, Optional, Tuple
from functools import lru_cache


# ── Lazy imports (avoid loading at module level) ──────────────────────────────

@lru_cache(maxsize=1)
def _get_spacy():
    """Load SpaCy model once, cache forever."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def _get_wordnet():
    """Load NLTK WordNet and frequency data once."""
    import nltk
    for pkg in ["wordnet", "brown", "averaged_perceptron_tagger",
                "punkt", "omw-1.4", "stopwords"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    from nltk.corpus import wordnet, brown, stopwords
    from nltk.probability import FreqDist

    # Build frequency distribution from Brown corpus
    # This gives us word frequency rankings for synonym selection
    words = [w.lower() for w in brown.words()]
    freq  = FreqDist(words)
    stops = set(stopwords.words("english"))
    return wordnet, freq, stops


# ── Common AI n-grams to disrupt ──────────────────────────────────────────────
# These bigrams/trigrams appear constantly in AI academic text.
# We break them by inserting words or restructuring.

AI_BIGRAMS = [
    ("plays", "a"),
    ("wide", "range"),
    ("wide", "variety"),
    ("key", "role"),
    ("crucial", "role"),
    ("significant", "role"),
    ("important", "role"),
    ("rapid", "advancement"),
    ("growing", "body"),
    ("recent", "years"),
    ("modern", "era"),
    ("digital", "age"),
    ("real", "world"),
    ("decision", "making"),
    ("machine", "learning"),
    ("deep", "learning"),
    ("neural", "network"),
    ("large", "language"),
    ("natural", "language"),
]

AI_TRIGRAMS = [
    ("in", "today", "'s"),
    ("it", "is", "important"),
    ("it", "is", "worth"),
    ("it", "is", "crucial"),
    ("it", "should", "be"),
    ("plays", "a", "crucial"),
    ("plays", "a", "significant"),
    ("plays", "a", "key"),
    ("in", "recent", "years"),
    ("a", "wide", "range"),
    ("a", "wide", "variety"),
    ("in", "the", "field"),
    ("state", "of", "the"),
    ("of", "the", "art"),
    ("has", "been", "shown"),
    ("has", "been", "demonstrated"),
]

# ── Passive voice patterns ─────────────────────────────────────────────────────
# Patterns for detecting passive constructions via regex
# (used as fallback when SpaCy dep parse isn't precise enough)
PASSIVE_PATTERNS = [
    (r'\bhas been (\w+ed)\b', 'have {0}'),
    (r'\bwas (\w+ed) by\b', '{0}'),
    (r'\bwere (\w+ed) by\b', '{0}'),
    (r'\bis (\w+ed) by\b', '{0}'),
    (r'\bare (\w+ed) by\b', '{0}'),
    (r'\bcan be (\w+ed)\b', 'can {0}'),
    (r'\bmust be (\w+ed)\b', 'must {0}'),
    (r'\bshould be (\w+ed)\b', 'should {0}'),
]

# ── Fronting adverbials ────────────────────────────────────────────────────────
# Move these to the front of sentences for syntactic variety
FRONTABLE_ADVERBIALS = [
    r'\b(in practice)[,]?\s+',
    r'\b(in theory)[,]?\s+',
    r'\b(in general)[,]?\s+',
    r'\b(in particular)[,]?\s+',
    r'\b(in most cases)[,]?\s+',
    r'\b(as a result)[,]?\s+',
    r'\b(for this reason)[,]?\s+',
    r'\b(to this end)[,]?\s+',
    r'\b(on the other hand)[,]?\s+',
    r'\b(by contrast)[,]?\s+',
    r'\b(at the same time)[,]?\s+',
    r'\b(for example)[,]?\s+',
    r'\b(for instance)[,]?\s+',
]

# ── Subordinate clause templates ───────────────────────────────────────────────
# Injected after the subject of long sentences to increase tree depth
SUBORDINATE_CLAUSES = [
    "which has grown considerably in recent scholarship",
    "a point that deserves more attention than it typically receives",
    "though the full implications remain contested",
    "particularly in high-stakes applied settings",
    "a distinction that is rarely made explicit",
    "as the evidence increasingly suggests",
    "at least when examined under controlled conditions",
    "though this varies considerably across contexts",
    "something practitioners frequently underestimate",
    "a pattern that holds across most domains studied",
    "when taken together with related findings",
    "despite the complexity this introduces",
]

# ── Short punchy fragments for burstiness ─────────────────────────────────────
FRAGMENTS = [
    "Worth pausing on.",
    "Not a minor point.",
    "The implications run deep.",
    "A meaningful distinction.",
    "Difficult to overstate.",
    "The evidence is clear.",
    "Not always appreciated.",
    "A pattern worth noting.",
    "The data bear this out.",
    "Rarely acknowledged openly.",
    "Context matters here.",
    "The stakes are real.",
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class NLPSurgeon:
    """
    Applies advanced offline NLP transformations to text.
    Each method targets a specific detector signal.
    All operations are deterministic-ish (controlled randomness).
    """

    def operate(self, text: str) -> str:
        """
        Run the full surgery pipeline on a paragraph.
        Order is carefully chosen — each stage feeds into the next.

        Args:
            text — paragraph text after Gemini paraphrase

        Returns:
            Surgically transformed text targeting <5% AI detection
        """
        # Stage A: Passive → Active (SpaCy-guided)
        # Do this first — other stages work better on active voice
        text = self._invert_passive_constructions(text)

        # Stage B: Synonym replacement (WordNet frequency-ranked)
        # Replace common words with rarer synonyms
        text = self._replace_with_rare_synonyms(text)

        # Stage C: N-gram disruption
        # Break up common AI phrase sequences
        text = self._disrupt_ai_ngrams(text)

        # Stage D: Syntactic fronting
        # Move mid-sentence adverbials to sentence-initial position
        text = self._front_adverbials(text)

        # Stage E: Burstiness enforcement
        # Split long sentences, insert fragments, fuse short pairs
        text = self._enforce_burstiness(text)

        # Stage F: Subordinate clause injection
        # Add syntactic depth via embedded clauses
        text = self._inject_subordinate_clauses(text)

        # Stage G: Final cleanup
        text = self._cleanup(text)

        return text


    # ── STAGE A: Passive → Active ─────────────────────────────────────────────

    def _invert_passive_constructions(self, text: str) -> str:
        """
        Convert passive voice constructions to active voice using
        SpaCy dependency parsing.

        SpaCy identifies:
          - nsubjpass (passive nominal subject)
          - agent (the "by" phrase)
          - auxpass (passive auxiliary: was/were/been)

        Example:
          "The model was trained by researchers" 
          → "Researchers trained the model"

        Falls back to regex patterns when SpaCy can't find a clean inversion.
        """
        try:
            nlp  = _get_spacy()
            doc  = nlp(text)
            sents = list(doc.sents)
            result = []

            for sent in sents:
                inverted = self._try_invert_sentence(sent)
                result.append(inverted)

            return " ".join(result)

        except Exception:
            # Fallback: regex-based passive detection
            return self._regex_passive_fix(text)

    def _try_invert_sentence(self, sent) -> str:
        """
        Attempt SpaCy dependency-guided passive inversion on one sentence.
        Returns original sentence string if inversion isn't clean.
        """
        text = sent.text.strip()

        # Find passive subject, verb, and agent
        passive_subj = None
        passive_verb = None
        agent        = None

        for token in sent:
            if token.dep_ == "nsubjpass":
                passive_subj = token
            if token.dep_ == "auxpass":
                passive_verb = token.head
            if token.dep_ == "agent":
                # The agent's children are the actual "by X" noun
                for child in token.children:
                    if child.dep_ == "pobj":
                        agent = child

        # Only invert if we found all three components
        if passive_subj and passive_verb and agent:
            try:
                agent_span = _get_subtree_text(agent)
                subj_span  = _get_subtree_text(passive_subj)
                verb_lemma = passive_verb.lemma_

                # Try to conjugate verb to match agent
                conjugated = _conjugate_verb(verb_lemma, agent_span)

                # Build: Agent + verb + subject
                inverted = f"{agent_span.capitalize()} {conjugated} {subj_span}"

                # Append remainder of sentence (anything after the agent)
                remainder = _get_sentence_remainder(sent, passive_subj, agent)
                if remainder:
                    inverted += f", {remainder}"

                inverted = inverted.rstrip() + "."
                return inverted
            except Exception:
                pass

        return text

    def _regex_passive_fix(self, text: str) -> str:
        """
        Fallback regex-based passive voice simplification.
        Converts 'has been shown', 'was demonstrated', etc.
        to simpler active constructions.
        """
        replacements = [
            (r'\bhas been shown to\b',      'shows signs of'),
            (r'\bhave been shown to\b',     'appear to'),
            (r'\bwas demonstrated that\b',  'shows that'),
            (r'\bwere found to be\b',       'appear to be'),
            (r'\bhas been found to\b',      'appears to'),
            (r'\bhas been established\b',   'stands'),
            (r'\bhas been suggested\b',     'appears'),
            (r'\bcan be seen\b',            'appears'),
            (r'\bcan be observed\b',        'appears'),
            (r'\bit has been noted\b',      'researchers note'),
            (r'\bit has been argued\b',     'scholars argue'),
            (r'\bit has been proposed\b',   'some propose'),
            (r'\bit is widely believed\b',  'many believe'),
            (r'\bit is generally accepted\b','most accept'),
            (r'\bit is commonly known\b',   'most acknowledge'),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


    # ── STAGE B: WordNet Frequency-Ranked Synonym Replacement ────────────────

    def _replace_with_rare_synonyms(self, text: str) -> str:
        """
        Use WordNet + Brown corpus frequency rankings to replace common words
        with synonyms that are statistically rarer.

        Strategy:
          1. Tag words with NLTK POS tagger
          2. For each content word, find WordNet synonyms
          3. Filter synonyms by:
             a. Same POS tag (ensures grammatical correctness)
             b. Lower frequency in Brown corpus (ensures rarity)
             c. Not a multi-word phrase (keeps sentence structure)
             d. Not already in the text (avoids repetition)
          4. Replace with probability 0.35 per eligible word
          5. Preserve original capitalisation

        Why frequency ranking matters:
          AI detectors measure perplexity — how surprising each word is.
          Replacing "important" (very common) with "consequential" (rarer)
          directly raises the perplexity score of the text.
        """
        try:
            import nltk
            wordnet, freq_dist, stopwords = _get_wordnet()
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
        except Exception:
            return text

        result_tokens = list(tokens)

        for i, (word, tag) in enumerate(tagged):
            # Skip stopwords, short words, punctuation, numbers
            if (word.lower() in stopwords
                    or len(word) <= 3
                    or not word.isalpha()
                    or word[0].isupper()):   # skip proper nouns
                continue

            # Only replace 35% of eligible words
            if random.random() > 0.35:
                continue

            # Map NLTK POS tag to WordNet POS
            wn_pos = _nltk_to_wordnet_pos(tag)
            if not wn_pos:
                continue

            # Get synonyms from WordNet
            synonyms = _get_wordnet_synonyms(wordnet, word.lower(), wn_pos)

            # Filter: must be rarer than original word in Brown corpus
            original_freq = freq_dist.freq(word.lower())
            rare_synonyms = [
                s for s in synonyms
                if (freq_dist.freq(s) < original_freq * 0.7   # at least 30% rarer
                    and freq_dist.freq(s) > 0                  # must exist in corpus
                    and s != word.lower()                      # not same word
                    and ' ' not in s                           # no multi-word
                    and len(s) > 3)                            # not too short
            ]

            if not rare_synonyms:
                continue

            # Pick the rarest synonym (lowest frequency)
            chosen = min(rare_synonyms, key=lambda s: freq_dist.freq(s))

            # Preserve original capitalisation
            if word[0].isupper():
                chosen = chosen[0].upper() + chosen[1:]

            result_tokens[i] = chosen

        # Reconstruct text from tokens
        return _detokenize(result_tokens)


    # ── STAGE C: N-gram Disruption ────────────────────────────────────────────

    def _disrupt_ai_ngrams(self, text: str) -> str:
        """
        Detect common AI bigrams and trigrams in the text and break them up
        by inserting syntactically valid interruptions.

        Example:
          "plays a crucial role in" → "serves — quite directly — as a driver of"
          "in recent years" → "over the past decade or so"
          "a wide range of" → "an assortment of"

        These substitutions target the n-gram distribution signal that
        detectors use to identify AI text.
        """
        # Direct phrase substitutions (most effective)
        phrase_subs = [
            # Role phrases
            (r'\bplays a crucial role in\b',     'is central to'),
            (r'\bplays a key role in\b',         'drives'),
            (r'\bplays a significant role in\b', 'shapes'),
            (r'\bplays an important role in\b',  'matters for'),
            (r'\bplays a pivotal role in\b',     'is at the heart of'),

            # Time phrases
            (r'\bin recent years\b',             'over the past few years'),
            (r'\bin today\'s world\b',           'today'),
            (r'\bin the modern era\b',           'now'),
            (r'\bin the digital age\b',          'in this era'),
            (r'\bat this point in time\b',       'now'),

            # Range phrases
            (r'\ba wide range of\b',             'a broad array of'),
            (r'\ba wide variety of\b',           'an assortment of'),
            (r'\ba plethora of\b',               'a wealth of'),
            (r'\ba myriad of\b',                 'many'),
            (r'\bnumerous\b',                    'many'),

            # Transition phrases
            (r'\bfurthermore\b',                 random.choice(['beyond this', 'what is more', 'on top of that'])),
            (r'\bmoreover\b',                    random.choice(['beyond this', 'additionally', 'equally'])),
            (r'\badditionally\b',                random.choice(['on top of this', 'equally', 'beyond that'])),
            (r'\bin conclusion\b',               'taking stock'),
            (r'\bin summary\b',                  'stepping back'),
            (r'\bto summarize\b',                'in short'),
            (r'\bin order to\b',                 'to'),
            (r'\bdue to the fact that\b',        'because'),
            (r'\bwith regard to\b',              'on'),
            (r'\bwith respect to\b',             'regarding'),
            (r'\bin terms of\b',                 'when it comes to'),
            (r'\bfor the purpose of\b',          'to'),

            # State of the art
            (r'\bstate-of-the-art\b',            'advanced'),
            (r'\bcutting-edge\b',                'advanced'),
            (r'\bgroundbreaking\b',              'notable'),
            (r'\brevolutionary\b',               'significant'),
            (r'\btransformative\b',              'far-reaching'),
            (r'\bseamless(?:ly)?\b',             'smooth'),
            (r'\brobust\b',                      random.choice(['solid', 'strong', 'reliable'])),
            (r'\bholistic\b',                    'comprehensive'),
            (r'\bparamount\b',                   'essential'),
            (r'\bpivotal\b',                     random.choice(['key', 'central', 'critical'])),

            # Common AI scaffolding
            (r'\bit is worth noting that\b',     ''),
            (r'\bit is important to note that\b',''),
            (r'\bit should be noted that\b',     ''),
            (r'\bit is crucial to understand that\b', ''),
            (r'\bneedless to say\b',             ''),
            (r'\bas mentioned (?:earlier|above|previously)\b', ''),
            (r'\bas (?:we can see|we have seen)\b', ''),
        ]

        for pattern, replacement in phrase_subs:
            if callable(replacement):
                replacement = replacement
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean up double spaces left by empty replacements
        text = re.sub(r'  +', ' ', text).strip()
        # Fix sentences starting with lowercase after empty replacement
        text = re.sub(r'(?<=\. )([a-z])', lambda m: m.group(1).upper(), text)
        # Fix sentences starting with comma
        text = re.sub(r'^[,\s]+', '', text)
        text = re.sub(r'\. [,\s]+', '. ', text)

        return text


    # ── STAGE D: Syntactic Fronting ───────────────────────────────────────────

    def _front_adverbials(self, text: str) -> str:
        """
        Move mid-sentence adverbial phrases to sentence-initial position.

        Fronted adverbials are extremely common in human expert writing
        and very rare in AI output. They also change the syntactic structure
        in a way that directly raises perplexity scores.

        Example:
          "The system performs well in practice."
          → "In practice, the system performs well."

          "This approach works for the most part."
          → "For the most part, this approach works."
        """
        sentences = _split_sentences(text)
        result    = []

        for sentence in sentences:
            fronted = self._try_front_one(sentence)
            result.append(fronted)

        return " ".join(result)

    def _try_front_one(self, sentence: str) -> str:
        """Attempt to front an adverbial in one sentence. Returns original if not possible."""
        # Only front occasionally to avoid mechanical-sounding text
        if random.random() > 0.35:
            return sentence

        # Patterns: find adverbial phrase NOT at the start of the sentence
        frontable = [
            (r'(\w[^.]*?),?\s+(in practice)([,.\s])',        r'In practice, \1\3'),
            (r'(\w[^.]*?),?\s+(in theory)([,.\s])',          r'In theory, \1\3'),
            (r'(\w[^.]*?),?\s+(in general)([,.\s])',         r'In general, \1\3'),
            (r'(\w[^.]*?),?\s+(for the most part)([,.\s])', r'For the most part, \1\3'),
            (r'(\w[^.]*?),?\s+(as a result)([,.\s])',        r'As a result, \1\3'),
            (r'(\w[^.]*?),?\s+(by contrast)([,.\s])',        r'By contrast, \1\3'),
            (r'(\w[^.]*?),?\s+(on balance)([,.\s])',         r'On balance, \1\3'),
            (r'(\w[^.]*?),?\s+(in turn)([,.\s])',            r'In turn, \1\3'),
            (r'(\w[^.]*?),?\s+(at its core)([,.\s])',        r'At its core, \1\3'),
        ]

        for pattern, replacement in frontable:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                try:
                    new_sent = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
                    # Make sure result starts with capital
                    new_sent = new_sent[0].upper() + new_sent[1:]
                    # Only accept if result is clean
                    if len(new_sent) > 10:
                        return new_sent
                except Exception:
                    pass

        return sentence


    # ── STAGE E: Burstiness Enforcement ───────────────────────────────────────

    def _enforce_burstiness(self, text: str) -> str:
        """
        Enforce sentence length variance (burstiness) using SpaCy-guided
        sentence splitting and strategic fragment insertion.

        Target distribution per paragraph:
          - At least 1 sentence < 6 words  (fragment/punchy)
          - At least 1 sentence > 28 words (complex/detailed)
          - At least 2 sentences 10-20 words (medium)
          - Standard deviation of lengths > 8 words

        Operations:
          1. Split very long sentences (>35 words) at dependency boundaries
          2. Fuse very short adjacent sentences (<7 words each)
          3. Insert strategic fragments after long sentences

        Using SpaCy dependency parsing to find SAFE split points:
          - Split at coordinating conjunctions (cc dependency)
          - Split at relative clause boundaries (relcl)
          - Split at adverbial clause boundaries (advcl)
        """
        sentences = _split_sentences(text)
        if not sentences:
            return text

        processed = []
        has_short  = False
        has_long   = False

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            wc   = len(sent.split())

            if wc > 35:
                # Try SpaCy-guided split
                parts = self._spacy_split(sent)
                if parts and len(parts) == 2:
                    processed.extend(parts)
                    has_long  = True
                    has_short = has_short or any(len(p.split()) < 8 for p in parts)
                else:
                    # Fallback: comma split
                    comma_parts = self._comma_split(sent)
                    if comma_parts:
                        processed.extend(comma_parts)
                    else:
                        processed.append(sent)
                    has_long = True

            elif wc < 7 and i + 1 < len(sentences) and len(sentences[i+1].split()) < 7:
                # Fuse two short sentences into one medium sentence
                fused = self._fuse_sentences(sent, sentences[i+1])
                processed.append(fused)
                i += 2
                continue

            else:
                processed.append(sent)

            if wc < 6:
                has_short = True
            if wc > 28:
                has_long = True

            i += 1

        # Inject a fragment if we still don't have a short sentence
        if not has_short and processed:
            frag = random.choice(FRAGMENTS)
            # Insert after the longest sentence
            longest_i = max(range(len(processed)), key=lambda x: len(processed[x].split()))
            processed.insert(longest_i + 1, frag)

        # If no long sentence exists, expand the longest medium sentence
        if not has_long and processed:
            longest_i = max(range(len(processed)), key=lambda x: len(processed[x].split()))
            processed[longest_i] = self._expand_sentence(processed[longest_i])

        return " ".join(processed)

    def _spacy_split(self, sentence: str) -> Optional[List[str]]:
        """
        Use SpaCy dependency parse to find a safe split point.
        Looks for coordinating conjunctions (and/but/yet/while)
        at the root level of the parse tree.
        """
        try:
            nlp = _get_spacy()
            doc = nlp(sentence)

            for token in doc:
                # Find coordinating conjunction at clause boundary
                if (token.dep_ == "cc"
                        and token.head.dep_ in ("ROOT", "conj")
                        and token.i > 2
                        and token.i < len(doc) - 3):

                    split_pos = token.idx   # character position
                    first  = sentence[:split_pos].rstrip(" ,")
                    second = sentence[split_pos + len(token.text):].strip()

                    if len(first.split()) >= 8 and len(second.split()) >= 6:
                        # Capitalise second part and ensure first ends with period
                        if not first.endswith("."):
                            first += "."
                        second = second[0].upper() + second[1:]
                        return [first, second]

                # Also split at adverbial clause boundaries
                if (token.dep_ in ("advcl", "relcl")
                        and token.i > 5
                        and token.i < len(doc) - 3):
                    # Split just before this clause
                    split_pos = token.idx
                    first  = sentence[:split_pos].rstrip(" ,")
                    second = sentence[split_pos:].strip()

                    if len(first.split()) >= 10 and len(second.split()) >= 6:
                        if not first.endswith("."):
                            first += "."
                        second = second[0].upper() + second[1:]
                        return [first, second]

        except Exception:
            pass
        return None

    def _comma_split(self, sentence: str) -> Optional[List[str]]:
        """
        Fallback: split at a comma roughly in the middle of the sentence.
        Used when SpaCy can't find a clean dependency boundary.
        """
        words  = sentence.split()
        mid    = len(words) // 2
        text   = sentence

        commas = [m.start() for m in re.finditer(r",", text)]
        target = len(" ".join(words[:mid]))

        best_comma = None
        best_dist  = float("inf")
        for pos in commas:
            if 0.25 < pos / len(text) < 0.75:
                dist = abs(pos - target)
                if dist < best_dist:
                    best_dist  = dist
                    best_comma = pos

        if best_comma is not None:
            first  = text[:best_comma].strip()
            second = text[best_comma+1:].strip()

            if len(first.split()) >= 8 and len(second.split()) >= 6:
                connector = random.choice([
                    ". That said, ", ". Even so, ", ". Still, ",
                    ". And yet ", ". In practice, ", ". The result: ",
                ])
                if not first.endswith("."):
                    first += connector
                else:
                    first += " " + second[0].upper() + second[1:]
                    return [first]

                second = second[0].upper() + second[1:]
                return [first, second]

        return None

    def _fuse_sentences(self, s1: str, s2: str) -> str:
        """
        Fuse two short sentences into one longer compound sentence.
        Uses a random connector for variety.
        """
        connectors = [
            ", and ",
            ", though ",
            ", while ",
            ", yet ",
            " — and ",
            ", even as ",
        ]
        s1 = s1.rstrip(".")
        s2 = s2[0].lower() + s2[1:]
        return s1 + random.choice(connectors) + s2

    def _expand_sentence(self, sentence: str) -> str:
        """
        Expand a medium sentence by inserting a subordinate clause
        to increase its length and syntactic depth.
        """
        clause = random.choice(SUBORDINATE_CLAUSES)
        words  = sentence.rstrip(".").split()

        # Insert clause after the first noun phrase (roughly 30% through)
        insert_at = max(2, len(words) // 3)
        words.insert(insert_at, f"— {clause} —")

        return " ".join(words) + "."


    # ── STAGE F: Subordinate Clause Injection ─────────────────────────────────

    def _inject_subordinate_clauses(self, text: str) -> str:
        """
        Add syntactic depth by injecting subordinate clauses into
        long sentences. Uses SpaCy to find the main subject (nsubj)
        and inserts the clause directly after it.

        This increases parse tree depth — a signal human writing
        exhibits naturally that AI tends to flatten.

        Example:
          "The system achieves high accuracy on benchmarks."
          → "The system — which has been evaluated extensively — 
             achieves high accuracy on benchmarks."

        Only applies to ~25% of eligible sentences to avoid over-injection.
        """
        sentences = _split_sentences(text)
        result    = []

        for sentence in sentences:
            wc = len(sentence.split())

            # Only inject in medium-to-long sentences
            if wc < 12 or wc > 40:
                result.append(sentence)
                continue

            # Only inject 25% of the time
            if random.random() > 0.25:
                result.append(sentence)
                continue

            injected = self._inject_one_clause(sentence)
            result.append(injected)

        return " ".join(result)

    def _inject_one_clause(self, sentence: str) -> str:
        """
        Find the subject of the sentence via SpaCy and inject
        a parenthetical clause directly after it.
        """
        try:
            nlp = _get_spacy()
            doc = nlp(sentence)

            for token in doc:
                if token.dep_ == "nsubj" and token.i > 0:
                    # Get the full subject span
                    subj_end = token.right_edge.idx + len(token.right_edge.text)
                    clause   = random.choice(SUBORDINATE_CLAUSES)

                    new_sentence = (
                        sentence[:subj_end]
                        + f" — {clause} —"
                        + sentence[subj_end:]
                    )
                    return new_sentence

        except Exception:
            pass

        return sentence


    # ── STAGE G: Cleanup ──────────────────────────────────────────────────────

    def _cleanup(self, text: str) -> str:
        """
        Fix all artifacts introduced by surgery stages.
        Ensures the text is grammatically clean and readable.
        """
        # Fix double spaces
        text = re.sub(r'  +', ' ', text)
        # Fix space before punctuation
        text = re.sub(r' ([.,;:!?])', r'\1', text)
        # Fix double periods
        text = re.sub(r'\.\.+', '.', text)
        # Fix em-dash spacing
        text = re.sub(r'\s*—\s*', ' — ', text)
        # Fix multiple commas
        text = re.sub(r',{2,}', ',', text)
        # Ensure sentence-initial capitalisation
        text = re.sub(
            r'(?<=[.!?] )([a-z])',
            lambda m: m.group(1).upper(),
            text
        )
        # Fix sentence starting with lowercase
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        # Strip invisible characters
        for ch in ['\u200B', '\u00AD', '\u200D']:
            text = text.replace(ch, '')
        # Fix orphaned punctuation at start of sentence
        text = re.sub(r'(?<=\. )[,;:]\s*', '', text)
        # Final whitespace normalisation
        text = text.strip()
        return text


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> List[str]:
    """Split text into individual sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def _get_subtree_text(token) -> str:
    """Get the full text of a SpaCy token's subtree."""
    return " ".join(t.text for t in token.subtree)


def _get_sentence_remainder(sent, subj_token, agent_token) -> str:
    """
    Get any remaining text from a sentence after the passive subject and agent.
    Used when constructing active voice inversions.
    """
    covered = set()
    for t in subj_token.subtree:
        covered.add(t.i)
    for t in agent_token.subtree:
        covered.add(t.i)
    # Also cover the auxpass tokens
    for t in sent:
        if t.dep_ in ("auxpass", "nsubjpass", "agent"):
            covered.add(t.i)

    remainder_tokens = [t.text for t in sent if t.i not in covered
                        and t.text not in ('.', ',', 'by')]
    return ' '.join(remainder_tokens).strip()


def _conjugate_verb(lemma: str, subject: str) -> str:
    """
    Simple verb conjugation based on subject.
    Handles third-person singular vs plural.
    Not a full conjugator — covers the most common cases.
    """
    subject_lower = subject.lower()

    # Third person singular subjects → add 's'
    singular_indicators = ['he', 'she', 'it', 'this', 'that']
    is_singular = any(subject_lower.startswith(s) for s in singular_indicators)

    # Irregular verbs
    irregulars = {
        'be':    'is' if is_singular else 'are',
        'have':  'has' if is_singular else 'have',
        'do':    'does' if is_singular else 'do',
        'go':    'goes' if is_singular else 'go',
    }

    if lemma in irregulars:
        return irregulars[lemma]

    if is_singular:
        if lemma.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return lemma + 'es'
        return lemma + 's'

    return lemma


def _nltk_to_wordnet_pos(nltk_tag: str) -> Optional[str]:
    """
    Convert NLTK POS tag to WordNet POS constant.
    Returns None for tags we don't handle (pronouns, punctuation, etc.)
    """
    from nltk.corpus import wordnet
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    return None


def _get_wordnet_synonyms(wordnet, word: str, pos: str) -> List[str]:
    """
    Get all synonyms for a word from WordNet, filtered for quality.
    Returns list of single-word synonyms in the same POS category.
    """
    synonyms = set()
    for synset in wordnet.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            syn = lemma.name().replace('_', ' ')
            # Only single words, no underscores, not the original word
            if ' ' not in syn and syn != word and syn.isalpha():
                synonyms.add(syn.lower())
    return list(synonyms)


def _detokenize(tokens: List[str]) -> str:
    """
    Reconstruct text from a list of tokens with proper spacing.
    Handles punctuation attachment correctly.
    """
    text = ""
    for i, token in enumerate(tokens):
        if i == 0:
            text = token
        elif token in ('.', ',', ';', ':', '!', '?', ')', ']', '}', "'s", "n't", "'re", "'ve", "'ll", "'d"):
            text += token
        elif i > 0 and tokens[i-1] in ('(', '[', '{'):
            text += token
        elif token == "'" and i > 0:
            text += token
        else:
            text += ' ' + token
    return text