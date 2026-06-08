"""
lib.py - Core Utilities for Semantic Folding Pipeline

This module provides essential utilities for the semantic folding pipeline, including:
- Text normalization and lemmatization with POS-aware processing
- Phrase expansion and filtering strategies
- File I/O operations for phrases, contexts, and fingerprints
- Sparse fingerprint representation handling

The module ensures consistency across all pipeline stages by providing
centralized implementations of common operations like phrase normalization,
word boundary detection, and fingerprint loading.

Key Design Principles:
- Cached lemmatization for performance (@lru_cache)
- POS-aware text processing for semantic accuracy
- Sparse representation support for memory efficiency
- Consistent normalization across all pipeline stages

Author: [Your Name]
Date: 2026-03-18
"""
import spacy
from spacy.tokens import Token
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from scipy.sparse import hstack, csr_matrix, lil_matrix
from rich import print
from loguru import logger
import numpy as np
from functools import lru_cache
import json, os


# nltk.data.path.insert(0, 'C:\\nltk_data')
nltk.data.path.insert(0, "D:\\darsi\\ms\\Thesis\\Dr.Banaie\\code050302\\nltk_data")
os.environ['NLTK_DATA'] = r'D:\\darsi\\ms\\Thesis\\Dr.Banaie\\code050302\\nltk_data'

import re
from hazm import Normalizer, word_tokenize
normalizer = Normalizer()
from typing import List, Set, Tuple, Optional
from functools import lru_cache
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize


import sys
import os
from pathlib import Path
from loguru import logger as _base_logger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _stderr_formatter(record):
    record["extra"].setdefault("step", record["name"])
    return "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[step]}</cyan> | {message}\n"

def get_logger(name: str):
    _base_logger.remove()

    _base_logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=_stderr_formatter,
        colorize=True,
    )

    _base_logger.add(
        LOG_DIR / f"{name}.log",
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        rotation="10 MB",
        retention=7,
        compression="zip",
        encoding="utf-8",
        colorize=False,
    )

    return _base_logger.bind(step=name)

# ---------------------------------------------------------
# Domain-Aware Stopwords
# ---------------------------------------------------------
_BASE_STOP_WORDS = set(stopwords.words('english'))
_STOP_WORD_EXCEPTIONS = {
    'need', 'use', 'used', 'using', 'without', 'across', 
    'between', 'multiple', 'single', 'further', 'new', 
    'own', 'same', 'such', 'most', 'more', 'less'
}
_EXTRA_STOP_WORDS = {
    'also', 'however', 'therefore', 'thus', 'et', 'al', 
    'eg', 'ie', 'etc', 'would', 'could', 'may', 'might', 
    'one', 'two', 'three'
}
en_stop_words = (_BASE_STOP_WORDS - _STOP_WORD_EXCEPTIONS) | _EXTRA_STOP_WORDS

# ---------------------------------------------------------
# Acronyms & Semantic Word Filter
# ---------------------------------------------------------
_DOMAIN_ACRONYMS = {'ai', 'ml', 'nlp', 'iot', 'api', 'p2p', 'qa', 'ui', 'db', 'id', 'os'}

def is_generic_word(word: str, min_length: int = 3) -> bool:
    if word.lower() in _DOMAIN_ACRONYMS:
        return False
    if len(word) < min_length:
        return True
    if word in en_stop_words:
        return True
    if word.isdigit() or not word.isalpha():
        return True
    return False

# ============================================================================
# CORE NLP UTILITIES
# ============================================================================
# ---------------------------------------------------------
# WordNet Mapping & Cached Lemmatization
# ---------------------------------------------------------
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Safely map POS tags, ensuring participles stay adjectival."""
    if treebank_tag.startswith('J') or treebank_tag in ['VBN', 'VBG']: # Added VBG
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

@lru_cache(maxsize=10000)
def lemmatize_token(word: str, pos_tag_str: str) -> str:
    pos = get_wordnet_pos(pos_tag_str)
    return lemmatizer.lemmatize(word.lower(), pos=pos)

def clear_lemma_cache():
    lemmatize_token.cache_clear()


@lru_cache(maxsize=10000)
def lemmatize_token(word: str, pos_tag: str) -> str:
    """
    Lemmatize a single token with POS-aware processing and caching.
    
    Lemmatization reduces words to their base form (lemma) while considering
    their part-of-speech. Caching significantly improves performance for
    repeated tokens across large corpora.
    
    Args:
        word: Input word to lemmatize
        pos_tag: Penn Treebank POS tag for the word
    
    Returns:
        Lemmatized form of the word in lowercase
    
    Examples:
        >>> lemmatize_token('running', 'VBG')
        'run'
        >>> lemmatize_token('better', 'JJR')
        'good'
        >>> lemmatize_token('mice', 'NNS')
        'mouse'
    
    Note:
        The @lru_cache decorator caches up to 10,000 unique (word, pos_tag)
        pairs, providing substantial speedup for corpus-level processing.
    """
    pos = get_wordnet_pos(pos_tag)
    return lemmatizer.lemmatize(word.lower(), pos=pos)


def is_generic_word(word: str, min_length: int = 3) -> bool:
    """
    Determine if a single word is too generic to carry semantic meaning.
    
    Generic words are filtered out during phrase expansion to maintain
    semantic quality. A word is considered generic if it meets any of:
    - Too short (< min_length characters)
    - Common stop word (articles, prepositions, etc.)
    - Pure numeric string
    
    Args:
        word: Input word to evaluate
        min_length: Minimum character length threshold (default: 3)
    
    Returns:
        True if word is generic and should be filtered, False otherwise
    
    Examples:
        >>> is_generic_word('the')
        True  # stop word
        >>> is_generic_word('ai')
        True  # too short (< 3 chars)
        >>> is_generic_word('123')
        True  # numeric
        >>> is_generic_word('algorithm')
        False  # meaningful content word
    """
    if len(word) < min_length:
        return True
    if word in en_stop_words:
        return True
    if word.isdigit():
        return True
    return False

def is_valid_phrase_structure(tagged_tokens: List[Tuple[str, str]]) -> bool:
    if not tagged_tokens:
        return False
    
    pos_tags = [tag for _, tag in tagged_tokens]
    
    # Reject pure functional verbs and pure adverbs
    if all(tag.startswith('V') and tag not in ('VBN', 'VBG') for tag in pos_tags):
        return False
    if all(tag.startswith('RB') for tag in pos_tags):
        return False
    
    has_content = any(tag.startswith(('N', 'J')) or tag in ('VBN', 'VBG') for tag in pos_tags)
    
    if len(tagged_tokens) > 1:
        # Multi-word phrases must contain a noun
        has_noun = any(tag.startswith('N') for tag in pos_tags)
        
        # STRICT RULE: A multi-word noun phrase should generally end in a noun.
        # This prevents trailing adjectives or NLTK guessing errors.
        # Allow 'S' for plurals (NNS) or proper nouns (NNPS)
        ends_with_noun = pos_tags[-1].startswith('N') 
        
        return has_noun and has_content and ends_with_noun
        
    return has_content

# ============================================================================
# TEXT NORMALIZATION
# ============================================================================
def _is_functional_verb(word: str, tag: str, next_tag: Optional[str] = None) -> bool:
    # Always drop finite verbs regardless of context
    if tag in ("VBZ", "VBP", "VBD", "MD", "VB"):
        return True
    if tag == "VBN" and next_tag in ("NN", "NNS", "NNP", "NNPS"):
        return False
    if tag == "VBG" and next_tag in ("NN", "NNS"):
        return False
    if tag in ("VBN", "VBG"):
        return True
    return False
def normalize_adjective(tok: Token) -> str:
    """Return base form for comparative/superlative adjectives."""
    if tok.tag_ in ('JJR', 'JJS'):  # comparative or superlative
        return tok.lemma_
    return tok.text.lower()

@lru_cache(maxsize=2048)
def normalize_phrase(text: str, remove_verbs: bool = True) -> Optional[str]:
    """
    Normalize a raw phrase string into a canonical form suitable for indexing.

    Processing pipeline (in order):
      1. Tokenize with NLTK word_tokenize.
      2. POS-tag the token list.
      3. Per-token filtering loop:
         a. Strip determiners (DT) — carry no indexing value.
         b. Verb handling (VB*):
            - VBN/VBG in non-final position → participial/gerundive adjective (JJ).
              e.g. "decentralized approach", "promising aspect"
            - VBG as sole or final token → nominal gerund head (NN).
              e.g. "understanding", "deep learning"
            - All other verb forms (VBZ, VBD, VBP, VB, VBN at end) → reject whole phrase.
         c. Comparative/superlative adjectives (JJR, JJS, RBR, RBS) → lemmatize as JJ.
         d. JJ tokens ending in -er/-est that NLTK mis-tags → force JJR/JJS lemmatization.
         e. Empty, non-alphabetic, or stopword tokens → skip silently.
         f. Functional verbs (auxiliaries, copulas) → skip if remove_verbs=True.
         g. All remaining tokens → lemmatize and accumulate.
      4. Reject if processed list is empty.
      5. Validate token sequence structure via is_valid_phrase_structure.
      6. Return joined lemma string.

    Args:
        text:         raw phrase string, e.g. "the unique characteristics".
        remove_verbs: if True, functional verbs are filtered via _is_functional_verb.

    Returns:
        Normalized phrase string, or None if the phrase is invalid or filtered out.

    Examples:
        >>> normalize_phrase("the unique characteristics")
        'unique characteristic'
        >>> normalize_phrase("decentralized approach")
        'decentralized approach'
        >>> normalize_phrase("is running")
        None
        >>> normalize_phrase("understanding")
        'understanding'
    """
    logger.debug(f"[NORMALIZE ENTER] text={text!r} remove_verbs={remove_verbs}")

    # ── step 1: tokenize ──────────────────────────────────────────────────────
    tokens = word_tokenize(text)
    if not tokens:
        logger.debug("[NORMALIZE] empty token list after word_tokenize — returning None")
        return None

    # ── step 2: POS-tag ───────────────────────────────────────────────────────
    # NLTK's averaged perceptron tagger; context is limited to the phrase itself,
    # so tags can differ from what a full-sentence tagger would assign.
    tagged_tokens = pos_tag(tokens)
    logger.debug(f"[POS TAGS] {tagged_tokens}")

    processed: list[str] = []           # accumulates final lemmas
    valid_tagged_tokens: list[tuple] = []  # parallel list for structure validation

    # ── step 3: per-token filtering loop ─────────────────────────────────────
    for i, (word, tag) in enumerate(tagged_tokens):
        # Sanitize: lowercase and strip punctuation (keeps hyphens for compound words)
        word_clean = re.sub(r'[^\w\s-]', '', word.lower())

        # ── 3a: strip determiners ─────────────────────────────────────────────
        # "the", "a", "an" add no indexing value; drop unconditionally.
        if tag == 'DT':
            logger.debug(f"[DT SKIP] '{word}' — determiner dropped")
            continue

        # ── 3b: verb handling ─────────────────────────────────────────────────
        if tag.startswith('VB'):
            is_last = (i == len(tagged_tokens) - 1)
            is_only = (len(tagged_tokens) == 1)
            logger.debug(
                f"[VB TAG] word={word!r} tag={tag!r} "
                f"is_last={is_last} is_only={is_only}"
            )

            # Rule 1 — participial / gerundive adjective modifier (non-head position).
            # VBN: "decentralized" in "decentralized approach"
            # VBG: "promising"     in "promising aspect"
            # These modify the head noun; treat as JJ so the phrase is kept.
            if tag in ('VBN', 'VBG') and not is_last:
                lemma = lemmatize_token(word_clean, tag)
                logger.debug(
                    f"[VB ADJMOD] '{word}' ({tag}) in modifier position "
                    f"→ treating as JJ, lemma={lemma!r}"
                )
                processed.append(lemma)
                valid_tagged_tokens.append((lemma, 'JJ'))
                continue

            # Rule 2 — nominal gerund head (sole token or rightmost token).
            # e.g. "understanding", "tampering", "deep learning"
            # The gerund functions as a noun; treat as NN so the phrase is kept.
            if tag == 'VBG' and (is_last or is_only):
                lemma = lemmatize_token(word_clean, tag)
                logger.debug(
                    f"[VBG PASS] '{word}' as nominal gerund head "
                    f"→ treating as NN, lemma={lemma!r}"
                )
                processed.append(lemma)
                valid_tagged_tokens.append((lemma, 'NN'))
                continue

            # Rule 3 — all other verb forms invalidate the whole phrase.
            # Finite verbs (VBZ, VBD, VBP, VB) and VBN in head position
            # indicate a clausal fragment, not a noun phrase.
            logger.debug(f"[VB REJECT] '{word}' ({tag}) is a finite/head verb — phrase rejected")
            return None

        # ── 3c: comparative / superlative adjectives ──────────────────────────
        # JJR ("better"), JJS ("best"), RBR ("faster"), RBS ("fastest")
        # Lemmatize to base adjective form and normalize tag to JJ for consistency.
        if tag in ('JJR', 'JJS', 'RBR', 'RBS'):
            lemma = lemmatize_token(word_clean, tag)
            logger.debug(
                f"[COMPARATIVE/SUPERLATIVE] '{word}' ({tag}) "
                f"→ lemma={lemma!r}, normalized tag → JJ"
            )
            processed.append(lemma)
            valid_tagged_tokens.append((lemma, 'JJ'))
            continue

        # ── 3d: NLTK JJ mis-tags for comparatives ────────────────────────────
        # NLTK sometimes tags "deeper", "wider" as JJ when context is thin.
        # Detect by suffix and force the correct comparative/superlative lemmatization.
        if tag == 'JJ' and word_clean.endswith(('er', 'est')):
            candidate_tag = 'JJR' if word_clean.endswith('er') else 'JJS'
            lemma = lemmatize_token(word_clean, candidate_tag)
            logger.debug(
                f"[JJ COMPARATIVE FIX] '{word}' mis-tagged as JJ, "
                f"re-lemmatized as {candidate_tag} → lemma={lemma!r}"
            )
            processed.append(lemma)
            valid_tagged_tokens.append((lemma, 'JJ'))
            continue

        # ── 3e: skip empty, non-alphabetic, and stopword tokens ───────────────
        if not word_clean or not word_clean.isalpha():
            logger.debug(f"[NON-ALPHA SKIP] '{word}' — empty or non-alphabetic")
            continue
        if word_clean in en_stop_words:
            logger.debug(f"[STOPWORD SKIP] '{word_clean}' — in stopword list")
            continue

        # ── 3f: functional verb filter ────────────────────────────────────────
        # Auxiliaries ("is", "has") and copulas ("be") are dropped when
        # remove_verbs=True. The next token's tag is passed for context
        # (e.g. "is" before VBG is auxiliary, not a content verb).
        next_tag = tagged_tokens[i + 1][1] if i + 1 < len(tagged_tokens) else None
        if remove_verbs and _is_functional_verb(word, tag, next_tag):
            logger.debug(
                f"[FUNCTIONAL VERB SKIP] '{word}' ({tag}) "
                f"next_tag={next_tag!r} — dropped as auxiliary/copula"
            )
            continue

        # ── 3g: lemmatize and accumulate ──────────────────────────────────────
        lemma = lemmatize_token(word_clean, tag)
        logger.debug(f"[LEMMATIZE] '{word}' ({tag}) → '{lemma}'")
        processed.append(lemma)
        valid_tagged_tokens.append((lemma, tag))

    # ── step 4: reject empty result ───────────────────────────────────────────
    if not processed:
        logger.debug(f"[NORMALIZE] no tokens survived filtering for {text!r} — returning None")
        return None

    # ── step 5: structural validation ────────────────────────────────────────
    # Checks POS sequence rules (e.g. no bare adjective phrases, valid head).
    if not is_valid_phrase_structure(valid_tagged_tokens):
        logger.debug(f"[STRUCT REJECT] {valid_tagged_tokens} failed is_valid_phrase_structure")
        return None

    # ── step 6: return joined lemma string ────────────────────────────────────
    result = ' '.join(processed)
    logger.debug(f"[NORMALIZE RESULT] {text!r} → {result!r}")
    return result

# ============================================================================
# PHRASE EXPANSION
# ============================================================================
from typing import List, Optional, Set
# ---------------------------------------------------------
# Boundary Matching & Safe Expansion
# ---------------------------------------------------------
def phrase_exists_in_context(phrase: str, lower_context: str) -> bool:
    """Word-boundary aware check to prevent 'chain' matching inside 'blockchain'."""
    pattern = r'\b' + re.escape(phrase) + r'\b'
    return bool(re.search(pattern, lower_context))

# Compiled once at module level
_HYPHEN_COMPOUND_RE = re.compile(
    r'\b([a-zA-Z]+)-([a-zA-Z]+)\b'
)

def normalize_hyphens(text: str) -> str:
    """
    Replace intra-word hyphens with spaces so hyphenated compounds
    are treated as multi-token phrases by downstream extractors.

    Only replaces hyphens that are surrounded by alphabetic characters
    (word-internal hyphens). Leaves em-dashes, en-dashes, and
    sentence-level punctuation untouched.

    Examples:
        'rule-based programming'   → 'rule based programming'
        'garbage-in, garbage-out'  → 'garbage in, garbage out'
        'high-dimensional data'    → 'high dimensional data'
        'non-linear activation'    → 'non linear activation'
        'state-of-the-art model'   → 'state of the art model'
    """
    return _HYPHEN_COMPOUND_RE.sub(r'\1 \2', text)

def detect_language(text: str) -> str:
    if re.search(r'[\u0600-\u06FF]', text):
        # فارسی
        if re.search(r'[پچژگ]', text):
            return "fa"

        # عربی
        return "ar"
    return "en"

def extract_raw_phrases_ar_fa(text: str) -> Set[str]:

    phrases = set()

    text = normalizer.normalize(text)

    tokens = word_tokenize(text)

    # unigram
    for tok in tokens:
        if len(tok) >= 2:
            phrases.add(tok)

    # bigram
    for i in range(len(tokens)-1):
        phrase = f"{tokens[i]} {tokens[i+1]}"
        phrases.add(phrase)

    # trigram
    for i in range(len(tokens)-2):
        phrase = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        phrases.add(phrase)

    return phrases

def normalize_arabic_phrase(text: str):

    ARABIC_STOPWORDS = {
    "من", "فی", "الی", "على", "علی", "ان", "إن",
    "کان", "قد", "ما", "هو", "هی", "هذا", "هذه",
    "ثم", "او", "أو", "و", "ف", "في", "إلى", "عن",
    "و", "يا", "ذلك", "الذي", "التي"
    }
    # 1. normalize unicode
    text = text.strip()
    text = normalizer.normalize(text)

    # 2. حذف اعراب قرآن
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    


    # 3. یکسان‌سازی حروف
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    # 4. tokenize
    tokens = word_tokenize(text)

    # 5. حذف stopwords
    tokens = [
        t for t in tokens
        if t not in ARABIC_STOPWORDS
    ]

    # 6. حذف token خیلی کوتاه
    tokens = [
        t for t in tokens
        if len(t) >= 2
    ]

    # 7. reject empty
    if not tokens:
        return None
    
    if len(tokens) > 5:
        return None

    return " ".join(tokens)

def expand_phrases(
    phrases: List[str],
    context_text: Optional[str],
    filter_generic: bool = True,
    min_word_length: int = 3,
) -> List[str]:
    """
    Expand raw phrases into all contiguous sub-spans, validate each against
    the source context (if provided), normalize survivors, and optionally filter
    generic single words.

    Processing pipeline (in order):
      1. For each raw phrase, generate all contiguous sub-spans up to MAX_NGRAM
         tokens wide (including the phrase itself).
         e.g. "machine translation model" →
              {"machine", "translation", "model",
               "machine translation", "translation model",
               "machine translation model"}

      2. Context validation — **when context_text is not None**,
         each candidate surface form must appear verbatim (case-insensitive)
         in the source text. This prevents hallucinated or reconstructed
         spans that were never actually written.  **When context_text is
         None, this check is skipped entirely** — the function trusts that
         all provided candidates are legitimate (e.g. for short query
         strings where lemmatised forms may differ from surface forms).

      3. Normalization — pass each surviving candidate through normalize_phrase.
         Candidates that produce None (invalid structure, bare verb, etc.) are
         dropped here.

      4. Generic-word filter (single-word phrases only, when filter_generic=True):
         Single-token results that are high-frequency / low-signal (e.g. "use",
         "new", "system") are dropped via is_generic_word. Multi-word phrases
         are never filtered here regardless of their tokens.

      5. Accumulate unique normalized forms in a set (automatic deduplication),
         then return as a sorted list for deterministic downstream processing.

    Args:
        phrases:          raw (un-normalised) phrases from the extractor.
        context_text:     original context string to validate surface forms
                          against, or **None** to skip validation.
        filter_generic:   drop single-word results that are generic/low-signal.
        min_word_length:  minimum character length for single-word phrases.

    Returns:
        Sorted list of normalised, validated phrases.

    Notes:
        - MAX_NGRAM=5 caps sub-span width; longer phrases are kept whole but not
          further sub-divided beyond 5 tokens.
        - Deduplication is by normalized form, so "translations" and "translation"
          both collapse to "translation" if the lemmatizer agrees.
        - Context validation (when active) uses phrase_exists_in_context which
          handles basic boundary checks; see that function for exact matching
          semantics.
        - Passing context_text=None is the intended mode for query processing,
          where the "document" is a short user‑written string and the normalised
          lemmatised forms carry the semantic intent.
    """
    logger.debug(
        f"[EXPAND ENTER] {len(phrases)} raw phrases | "
        f"filter_generic={filter_generic} min_word_length={min_word_length}"
    )

    expanded_and_validated: set[str] = set()
    lower_context = context_text.lower() if context_text is not None else ""
    MAX_NGRAM = 5

    # ── step 1: iterate over each raw phrase ─────────────────────────────────
    # count = 0
    # countAr = 0
    # countArC = 0
    # countArFi = 0
    # countArNorm = 0

    for raw_phrase in phrases:
        # count = count +1
        # if detect_language(raw_phrase) != "en":
        #     countAr = countAr +1

        words = raw_phrase.split()
        n = len(words)

        # ── step 1a: generate all contiguous sub-spans ────────────────────────
        # Always include the full phrase itself, then add all sub-spans up to
        # MAX_NGRAM tokens wide. Using a set avoids duplicate candidates when
        # the phrase is shorter than MAX_NGRAM (full phrase == a sub-span).
        candidates: set[str] = {raw_phrase}
        for size in range(1, min(n, MAX_NGRAM) + 1):
            for i in range(n - size + 1):
                candidates.add(' '.join(words[i:i + size]))

        logger.debug(
            f"[EXPAND] '{raw_phrase}' ({n} tokens) "
            f"→ {len(candidates)} candidate sub-spans generated"
        )

        # ── steps 2–4: validate, normalize, filter each candidate ─────────────
        
        for candidate in candidates:
            # if detect_language(raw_phrase) != "en":
                # countArC = countArC +1
            
            # ── step 2: context validation (optional) ─────────────────────────
            # When context_text is None, the surface‑form check is skipped.
            # This is the intended behaviour for short query strings where
            # lemmatised forms (e.g. ‘emotion’ from ‘emotions’) may not appear
            # verbatim.  For document indexing (context_text is a full paragraph)
            # the check remains active to avoid spurious sub‑spans.
            #######
            if context_text is not None:
                if not phrase_exists_in_context(candidate.lower(), lower_context):
                    logger.debug(
                        f"  [CONTEXT MISS] '{candidate}' — not found in source text"
                    )
                    continue
            else:
                logger.debug(
                    f"  [CONTEXT SKIP] context_text=None — keeping '{candidate}'"
                )

            # ── step 3: normalization ─────────────────────────────────────────
            # normalize_phrase handles POS filtering, lemmatization, and
            # structural validation. None means the candidate is not a valid
            # noun phrase (e.g. bare verb, failed structure check).
            norm = normalize_phrase(candidate, remove_verbs=False)
            
            if not norm:
                logger.debug(
                    f"  [NORM DROP] '{candidate}' — normalize_phrase returned None"
                )
                continue

            # ── step 4: generic single-word filter ───────────────────────────
            # Only applied to single-token normalized results. Multi-word phrases
            # are never dropped here, even if they contain generic tokens.

            
            if ' ' not in norm and filter_generic and is_generic_word(norm, min_word_length):
                logger.debug(
                    f"  [GENERIC DROP] '{norm}' (from '{candidate}') "
                    f"— flagged as generic/low-signal single word"
                )
                continue

            # ── step 5: accumulate ────────────────────────────────────────────

            if detect_language(norm) != "en":
                # countArFi = countArFi +1
                norm = normalize_arabic_phrase(candidate)
                if not norm:
                    # countArNorm = countArNorm + 1
                    continue
            
            logger.debug(f"  [KEEP] '{candidate}' → normalized='{norm}'")
            expanded_and_validated.add(norm)

    # print(count, countAr, countArC, countArFi, countArNorm, "\n###########")

    # ── final: sort and return ────────────────────────────────────────────────
    # Sorting ensures deterministic output order for downstream deduplication
    # and CSV/JSON serialization.
    result = sorted(expanded_and_validated)
    logger.debug(
        f"[EXPAND RESULT] {len(result)} unique normalized phrases kept "
        f"from {len(phrases)} raw inputs"
    )

    return result
# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def load_phrases(phrases_path: Path, min_freq: int = 0) -> List[Tuple[str, int]]:
    """
    Load phrases with frequencies from phrase inventory file.
    
    Expected file format (one phrase per line):
        phrase_text:frequency
    
    Example:
        machine learning:150
        neural network:89
        deep learning:203
    
    Args:
        phrases_path: Path to phrases file
        min_freq: Minimum frequency threshold (default: 0, no filtering)
    
    Returns:
        List of (phrase, frequency) tuples for phrases meeting threshold
    
    Raises:
        FileNotFoundError: If phrases_path does not exist
        ValueError: If file format is invalid
    
    Note:
        Phrases are NOT normalized during loading. Normalization should
        be applied separately using normalize_phrase() when needed.
    """
    logger.info(f"Loading phrases from: {phrases_path}")
    
    phrases = []
    with open(phrases_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                phrase, freq_str = line.split(':', 1)
                phrase = phrase.strip()
                try:
                    freq = int(freq_str.strip())
                    if freq >= min_freq and phrase:
                        phrases.append((phrase, freq))
                except ValueError:
                    logger.warning(f"Invalid frequency for phrase: '{line}'")
                    continue
    
    logger.success(f"Loaded {len(phrases)} phrases from: {phrases_path}")
    return phrases


def find_phrase_occurrences(text: str, phrase: str, 
                           use_word_boundaries: bool = True) -> int:
    """
    Count phrase occurrences in text with proper word boundary detection.
    
    Word boundary detection ensures accurate matching by preventing
    false positives from substring matches (e.g., 'cat' should not
    match 'concatenate').
    
    Args:
        text: Input text to search
        phrase: Phrase to search for
        use_word_boundaries: If True, only match complete words (default: True)
    
    Returns:
        Number of occurrences found
    
    Examples:
        >>> find_phrase_occurrences('the cat and the cats', 'cat', True)
        1  # matches 'cat' but not 'cats'
        >>> find_phrase_occurrences('the cat and the cats', 'cat', False)
        2  # matches both 'cat' and 'cats' (substring)
    
    Note:
        Always use word boundaries (use_word_boundaries=True) for accurate
        phrase matching in semantic contexts.
    """
    import re
    
    if use_word_boundaries:
        # Escape special regex characters in phrase
        escaped_phrase = re.escape(phrase)
        # Use word boundaries for accurate matching
        pattern = r'\b' + escaped_phrase + r'\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(matches)
    else:
        # Fallback to simple substring matching
        return text.lower().count(phrase.lower())


def load_contexts(corpus_path: Path) -> List[Tuple[str, str]]:
    """
    Load contexts from corpus file with normalization.
    
    Expected file format (CSV):
        context_id,context_text
    
    Example:
        ctx_0,Machine learning is a subset of artificial intelligence
        ctx_1,Neural networks are inspired by biological neurons
    
    Args:
        corpus_path: Path to corpus file
    
    Returns:
        List of (context_id, normalized_context_text) tuples
    
    Note:
        Context text is normalized using normalize_phrase(remove_verbs=False)
        to preserve verbs, which can be important for context understanding.
    """
    logger.info(f"Loading contexts from: {corpus_path}")
    
    contexts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            
            context_id, context_text = line.split(',', 1)
            context_id = context_id.strip()
            context_text = context_text.strip()
            
            # Normalize context text (keep verbs for context)
            normalized_text = normalize_phrase(context_text, remove_verbs=False)
            if normalized_text:
                contexts.append((context_id, normalized_text))
    
    logger.success(f"Loaded {len(contexts)} contexts from: {corpus_path}")
    return contexts


def load_contexts_dict(corpus_path: Path) -> Dict[str, str]:
    """
    Load context texts as dictionary mapping context_id to text.
    
    Expected file format (CSV):
        context_id,context_text
    
    Args:
        corpus_path: Path to corpus file
    
    Returns:
        Dictionary mapping context_id -> context_text (not normalized)
    
    Note:
        Unlike load_contexts(), this function does NOT normalize text.
        Use this when you need the original context text.
    """
    logger.info(f"Loading context texts from: {corpus_path}")
    
    contexts = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            
            context_id, context_text = line.split(',', 1)
            contexts[context_id.strip()] = context_text.strip()
    
    logger.success(f"Loaded {len(contexts)} context texts from: {corpus_path}")
    return contexts


# ============================================================================
# FINGERPRINT LOADING UTILITIES
# ============================================================================
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fingerprint Loaders
#  Used by: query_processing.py (Step 6)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _load_doc_fingerprint_matrix(
    npz_path  : Path,
    meta_path : Path,
    npz_key   : str = "fingerprints",
) -> Tuple[np.ndarray, Dict[str, int], bool, int]:
    """
    Load document fingerprint matrix and metadata.

    Reads the .npz matrix and the new-style meta JSON that contains:
        - "doc_to_row" : {doc_id: row_index}
        - "use_morton" : bool
        - "grid_size"  : int

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file.
    meta_path : Path
        Path to the doc_fingerprints_meta.json file.
    npz_key : str
        Key inside the .npz archive holding the matrix.

    Returns
    -------
    matrix       : np.ndarray (n_docs, grid_size²)
    doc_index    : Dict[str, int]  mapping doc_id → row index
    use_morton   : bool
    grid_size    : int

    Raises
    ------
    FileNotFoundError  if either file missing.
    KeyError           if npz_key missing or meta missing required keys.
    ValueError         if row/index mismatch.
    """
    for p in (npz_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    # Load matrix
    archive = np.load(str(npz_path))
    if npz_key not in archive:
        raise KeyError(f"Key '{npz_key}' not in {npz_path.name}. Available: {list(archive.keys())}")
    matrix = archive[npz_key]
    n_docs, vector_size = matrix.shape
    logger.info(f"Document matrix shape: {matrix.shape} (n_docs={n_docs}, vec={vector_size})")

    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # Extract mapping and flags
    try:
        doc_to_row = meta["doc_to_row"]
        use_morton = meta["use_morton"]
        grid_size  = meta["grid_size"]
    except KeyError as e:
        raise KeyError(f"Missing required key in {meta_path.name}: {e}")

    if len(doc_to_row) != n_docs:
        logger.warning(
            f"Index map has {len(doc_to_row)} entries but matrix has {n_docs} rows "
            f"— possible misalignment."
        )

    # Validate grid_size consistency
    expected_cols = grid_size * grid_size
    if vector_size != expected_cols:
        raise ValueError(
            f"Matrix has {vector_size} columns but grid_size={grid_size} "
            f"implies {expected_cols} columns."
        )

    return matrix, doc_to_row, use_morton, grid_size

def _load_fingerprint_matrix(
    npz_path    : Path,
    index_path  : Path,
    npz_key     : str,
    label       : str,                    # "phrase" or "document" — for log messages
    grid_size   : Optional[int] = None,   # if given, column count is validated
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Shared low-level loader for any fingerprint .npz + index-map pair.

    Parameters
    ----------
    npz_path:
        Path to the .npz file containing the dense float32 matrix.
    index_path:
        Path to the JSON file containing {entity_string: row_index}.
    npz_key:
        Key inside the .npz archive that holds the matrix (e.g. "fingerprints").
    label:
        Human-readable entity type used in log/error messages.
    grid_size:
        If provided, validates that matrix columns == grid_size².

    Returns
    -------
    matrix     : np.ndarray  — shape (n_entities, vector_size)
    index_map  : Dict[str, int]

    Raises
    ------
    FileNotFoundError  — if either file is missing.
    KeyError           — if npz_key is absent from the archive.
    ValueError         — if grid_size is given and column count mismatches.
    """
    # ── Validate files exist ─────────────────────────────────────────────────
    for p in (npz_path, index_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {label} fingerprint file not found: {p}"
            )

    # ── Load matrix ──────────────────────────────────────────────────────────
    logger.info(f"Loading {label} fingerprint matrix from: {npz_path}")
    archive = np.load(str(npz_path))

    if npz_key not in archive:
        raise KeyError(
            f"Key '{npz_key}' not found in {npz_path.name}. "
            f"Available keys: {list(archive.keys())}"
        )

    matrix: np.ndarray = archive[npz_key]          # (n_entities, vector_size)
    n_entities, vector_size = matrix.shape
    logger.info(
        f"{label.capitalize()} matrix shape: {matrix.shape} "
        f"(n={n_entities}, vec={vector_size})"
    )

    # ── Optional column-count validation ─────────────────────────────────────
    if grid_size is not None:
        expected = grid_size * grid_size
        if vector_size != expected:
            raise ValueError(
                f"{label.capitalize()} matrix has {vector_size} columns but "
                f"grid_size={grid_size} implies {expected} columns. "
                f"Did you pass the correct --grid-size?"
            )

    # ── Load entity → row-index map ──────────────────────────────────────────
    logger.info(f"Loading {label} index map from: {index_path}")
    with open(index_path, "r", encoding="utf-8") as fh:
        index_map: Dict[str, int] = json.load(fh)

    # ── Row-count sanity check ────────────────────────────────────────────────
    if len(index_map) != n_entities:
        logger.warning(
            f"{label.capitalize()} index map has {len(index_map)} entries "
            f"but matrix has {n_entities} rows — possible misalignment."
        )

    return matrix, index_map

# ─────────────────────────────────────────────────────────────────────────────

def load_phrase_fingerprints_sparse(
    fingerprints_dir : Path,
    grid_size        : int,
) -> Dict[str, np.ndarray]:
    """
    Load phrase fingerprints produced by Step 4 (phrase_fingerprints.py).

    Expected files in fingerprints_dir:
        phrase_fingerprints.npz        — dense float32 matrix,
                                         key "fingerprints",
                                         shape (n_phrases, grid_size²)
        phrase_fingerprints_meta.json  — {phrase_string: row_index}

    Parameters
    ----------
    fingerprints_dir:
        Step 4 output directory (e.g. outputs/run/phrase_fingerprints/).
    grid_size:
        Grid side-length; used to validate matrix column count.

    Returns
    -------
    Dict[str, np.ndarray]
        phrase_string  →  float32 vector of length grid_size².

    Raises
    ------
    FileNotFoundError  — if either expected file is missing.
    ValueError         — if column count != grid_size²,
                         or if index_map references out-of-bound rows.
    """
    fingerprints_dir = Path(fingerprints_dir)

    matrix, index_map = _load_fingerprint_matrix(
        npz_path   = fingerprints_dir / "phrase_fingerprints.npz",
        index_path = fingerprints_dir / "phrase_fingerprints_meta.json",
        npz_key    = "fingerprints",
        label      = "phrase",
        grid_size  = grid_size,
    )

    n_rows = matrix.shape[0]
    n_keys = len(index_map)

    # ── alignment audit ──────────────────────────────────────────────────────
    if n_keys != n_rows:
        logger.warning(
            f"index_map has {n_keys} entries but matrix has {n_rows} rows "
            f"— index map and matrix may be misaligned. "
            f"Only mapped entries will be used."
        )

    # ── out-of-bound row check ────────────────────────────────────────────────
    bad_phrases = {
        phrase: idx
        for phrase, idx in index_map.items()
        if idx < 0 or idx >= n_rows
    }
    if bad_phrases:
        raise ValueError(
            f"index_map contains {len(bad_phrases)} out-of-bound row "
            f"reference(s) for matrix with {n_rows} rows. "
            f"Examples: { {k: v for k, v in list(bad_phrases.items())[:5]} }"
        )

    # ── build output dict — only rows that are mapped ─────────────────────────
    phrase_fps: Dict[str, np.ndarray] = {
        phrase: matrix[idx].astype(np.float32)
        for phrase, idx in index_map.items()
    }

    # ── report unmapped rows (matrix rows with no phrase key) ─────────────────
    mapped_row_indices = set(index_map.values())
    unmapped_rows = [i for i in range(n_rows) if i not in mapped_row_indices]
    if unmapped_rows:
        logger.warning(
            f"{len(unmapped_rows)} matrix row(s) have no corresponding phrase "
            f"key in the index map and will be ignored. "
            f"First few unmapped row indices: {unmapped_rows[:10]}"
        )

    logger.success(
        f"Loaded {len(phrase_fps)} phrase fingerprints "
        f"(grid_size={grid_size}, vector_size={grid_size**2}, "
        f"matrix_rows={n_rows}, mapped={len(phrase_fps)}, "
        f"unmapped_rows={len(unmapped_rows)})."
    )
    return phrase_fps


# ─────────────────────────────────────────────────────────────────────────────

def load_document_fingerprints(
    doc_fp_dir : Path,
) -> Tuple[Dict[str, "csr_matrix"], Dict]:
    """
    Load document fingerprints produced by Step 5.

    Expected files in doc_fp_dir:
        doc_fingerprints.npz
        doc_fingerprints_meta.json  (format: {"doc_to_row": ..., "use_morton": ..., "grid_size": ...})
        doc_fingerprints_stats.json

    Parameters
    ----------
    doc_fp_dir : Path
        Step 5 output directory.

    Returns
    -------
    doc_fingerprints : Dict[str, csr_matrix]
        doc_id → sparse row-vector of length grid_size².
    combined_metadata : Dict
        All fields from stats.json plus "grid_size", "num_docs", "use_morton".

    Raises
    ------
    FileNotFoundError  if required files missing.
    KeyError           if meta/stats structure incorrect.
    """
    from scipy.sparse import csr_matrix

    doc_fp_dir = Path(doc_fp_dir)

    stats_path = doc_fp_dir / "doc_fingerprints_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    # Load stats to obtain any extra info (optional)
    with open(stats_path, "r", encoding="utf-8") as fh:
        stats = json.load(fh)

    # Use the new document‑specific loader
    matrix, doc_index, use_morton, grid_size = _load_doc_fingerprint_matrix(
        npz_path  = doc_fp_dir / "doc_fingerprints.npz",
        meta_path = doc_fp_dir / "doc_fingerprints_meta.json",
    )

    # Build doc_id → sparse row vector
    doc_fingerprints = {
        doc_id: csr_matrix(matrix[row_idx].reshape(1, -1))
        for doc_id, row_idx in doc_index.items()
    }

    combined_metadata = {
        **stats,
        "grid_size"  : grid_size,
        "num_docs"   : len(doc_fingerprints),
        "use_morton" : use_morton,
    }

    logger.success(
        f"Loaded {len(doc_fingerprints)} document fingerprints "
        f"(grid_size={grid_size}, use_morton={use_morton})."
    )
    return doc_fingerprints, combined_metadata
def load_phrase_fingerprints_sparse(
    fingerprints_dir : Path,
    grid_size        : int,
) -> Dict[str, "csr_matrix"]:
    """
    Load phrase fingerprints (Step 4 output) as sparse CSR matrices.

    Step 4 writes two files into its output directory:
        phrase_fingerprints.npz        – dense float32 matrix, key "fingerprints",
                                          shape (n_phrases, grid_size * grid_size)
        phrase_fingerprints_meta.json  – metadata, either:
            * nested (new): { "phrase_to_row": {...}, "use_morton": bool, "grid_size": int }
            * flat (legacy): { "phrase": row_index, ... }

    The function detects the format automatically and returns a mapping from phrase string
    to a sparse row vector (csr_matrix of shape (1, grid_size²)).

    Parameters
    ----------
    fingerprints_dir : Path
        Directory containing phrase_fingerprints.npz and phrase_fingerprints_meta.json.
    grid_size : int
        Expected grid side length; used to validate the matrix column count.

    Returns
    -------
    Dict[str, csr_matrix]
        Mapping from normalised phrase string to its sparse fingerprint vector.

    Raises
    ------
    FileNotFoundError
        If either the .npz or meta file is missing.
    ValueError
        If the matrix column count does not match grid_size².
    """
    from scipy.sparse import csr_matrix

    npz_path  = fingerprints_dir / "phrase_fingerprints.npz"
    meta_path = fingerprints_dir / "phrase_fingerprints_meta.json"

    # ── Validate files exist ─────────────────────────────────────────────────
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Fingerprint matrix not found: {npz_path}\n"
            f"Expected Step 4 output inside: {fingerprints_dir}"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Phrase index map not found: {meta_path}\n"
            f"Expected Step 4 output inside: {fingerprints_dir}"
        )

    # ── Load matrix ──────────────────────────────────────────────────────────
    logger.info(f"Loading fingerprint matrix from: {npz_path}")
    data   = np.load(str(npz_path))
    matrix = data["fingerprints"]                    # shape (n_phrases, vector_size)
    n_phrases, vector_size = matrix.shape

    expected_cols = grid_size * grid_size
    if vector_size != expected_cols:
        raise ValueError(
            f"Matrix has {vector_size} columns but "
            f"grid_size={grid_size} implies {expected_cols} columns. "
            f"Did you pass the correct --grid-size?"
        )

    logger.info(
        f"Matrix shape: {matrix.shape} "
        f"(n_phrases={n_phrases}, vector_size={vector_size})"
    )

    # ── Load phrase → row-index map ──────────────────────────────────────────
    logger.info(f"Loading phrase index map from: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # Detect format: nested (new) or flat (legacy)
    if "phrase_to_row" in meta:
        token_map = meta["phrase_to_row"]                # nested mapping
        use_morton = meta.get("use_morton", True)
        meta_grid_size = meta.get("grid_size", None)
        logger.info(
            f"Loaded nested metadata: {len(token_map)} phrases, "
            f"use_morton={use_morton}, grid_size={meta_grid_size}"
        )
    else:
        token_map = meta                                 # flat mapping
        logger.info(f"Loaded flat metadata: {len(token_map)} phrases")

    # Sanity check
    if len(token_map) != n_phrases:
        logger.warning(
            f"Token map has {len(token_map)} entries but matrix has "
            f"{n_phrases} rows – possible misalignment."
        )

    # ── Build phrase → sparse CSR vector dict ─────────────────────────────────
    phrase_fps: Dict[str, "csr_matrix"] = {}
    for phrase, idx in token_map.items():
        idx = int(idx)
        if idx >= n_phrases:
            logger.warning(f"Skipping phrase '{phrase}' with out-of-range index {idx}")
            continue
        # Create a sparse row vector (1, vector_size)
        row_dense = matrix[idx].astype(np.float32)
        phrase_fps[phrase] = csr_matrix(row_dense.reshape(1, -1))

    logger.success(f"Loaded {len(phrase_fps)} phrase fingerprints (sparse format).")
    return phrase_fps


def load_fingerprint_cache(
    cache_path: Path,
    grid_size: int
) -> Dict[str, csr_matrix]:
    """
    Load document fingerprints from cache file into sparse matrix format.
    
    Expected file format (JSON):
        {
            "doc_id_1": {
                "coordinates": [[x1, y1], [x2, y2], ...],
                "values": [v1, v2, ...]
            },
            "doc_id_2": { ... }
        }
    
    The sparse CSR (Compressed Sparse Row) format is optimal for:
    - Memory efficiency with high-dimensional sparse data
    - Fast row slicing and matrix-vector operations
    - Efficient similarity computations
    
    Args:
        cache_path: Path to fingerprint cache JSON file
        grid_size: Size of the semantic grid (determines matrix dimensions)
    
    Returns:
        Dictionary mapping doc_id -> csr_matrix of shape (1, grid_size²)
    
    Example:
        >>> cache = load_fingerprint_cache(Path('doc_fps.json'), 128)
        >>> cache['doc_1'].shape
        (1, 16384)  # 128 * 128
        >>> cache['doc_1'].nnz
        47  # number of active bits
    
    Note:
        Each fingerprint is stored as a row vector (1, grid_size²) for
        compatibility with similarity computation functions.
    """
    logger.info(f"Loading fingerprint cache from: {cache_path}")
    
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    fingerprints = {}
    total_dims = grid_size * grid_size
    
    for doc_id, fp_data in cache_data.items():
        coords = fp_data.get('coordinates', [])
        values = fp_data.get('values', [])
        
        if not coords:
            logger.warning(f"Empty fingerprint for document: '{doc_id}'")
            continue
        
        # Convert 2D coordinates to 1D indices
        indices = [x * grid_size + y for x, y in coords]
        
        # Create sparse matrix (row vector)
        row = np.zeros(1, dtype=int)
        col = np.array(indices, dtype=int)
        data = np.array(values, dtype=float)
        
        # Build CSR matrix
        sparse_fp = csr_matrix((data, (row, col)), shape=(1, total_dims))
        fingerprints[doc_id] = sparse_fp
    
    logger.success(f"Loaded {len(fingerprints)} document fingerprints from: {cache_path}")
    return fingerprints


def save_fingerprint_cache(
    fingerprints: Dict[str, csr_matrix],
    cache_path: Path,
    grid_size: int
) -> None:
    """
    Save document fingerprints to cache file in JSON format.
    
    Converts sparse CSR matrices to JSON-serializable format with
    explicit coordinate and value storage.
    
    Args:
        fingerprints: Dictionary mapping doc_id -> csr_matrix
        cache_path: Path to output cache JSON file
        grid_size: Size of the semantic grid
    
    Example:
        >>> fps = {'doc_1': csr_matrix(...), 'doc_2': csr_matrix(...)}
        >>> save_fingerprint_cache(fps, Path('cache.json'), 128)
    
    Note:
        The cache file can be loaded back using load_fingerprint_cache()
        for fast retrieval without recomputation.
    """
    logger.info(f"Saving fingerprint cache to: {cache_path}")
    
    cache_data = {}
    
    for doc_id, sparse_fp in fingerprints.items():
        # Convert sparse matrix to coordinates and values
        sparse_fp = sparse_fp.tocoo()  # Convert to COO for easy iteration
        
        coords = []
        values = []
        
        for i, j, v in zip(sparse_fp.row, sparse_fp.col, sparse_fp.data):
            # Convert 1D index back to 2D coordinates
            x = j // grid_size
            y = j % grid_size
            coords.append([int(x), int(y)])
            values.append(float(v))
        
        cache_data[doc_id] = {
            'coordinates': coords,
            'values': values
        }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    
    logger.success(f"Saved {len(cache_data)} fingerprints to: {cache_path}")


# ============================================================================
# COORDINATE UTILITIES
# ============================================================================

def load_context_coordinates(coords_path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Load context coordinates from semantic space mapping file.
    
    Expected file format (CSV):
        context_id,x,y
        ctx_0,45,67
        ctx_1,23,89
    
    These coordinates represent the position of each context in the
    discretized semantic space grid, generated by semantic_space.py.
    
    Args:
        coords_path: Path to context coordinates CSV file
    
    Returns:
        Dictionary mapping context_id -> (x, y) grid coordinates
    
    Example:
        >>> coords = load_context_coordinates(Path('context_coords.csv'))
        >>> coords['ctx_0']
        (45, 67)
    
    Note:
        This file is generated by semantic_space.py and is required for
        phrase fingerprint generation in phrase_fingerprints.py.
    """
    logger.info(f"Loading context coordinates from: {coords_path}")
    
    coordinates = {}
    
    with open(coords_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) != 3:
                logger.warning(f"Invalid coordinate line format: '{line}'")
                continue
            
            context_id, x_str, y_str = parts
            context_id = context_id.strip()
            
            try:
                x = int(x_str.strip())
                y = int(y_str.strip())
                coordinates[context_id] = (x, y)
            except ValueError:
                logger.warning(
                    f"Invalid coordinates for '{context_id}': "
                    f"x='{x_str.strip()}', y='{y_str.strip()}'"
                )
                continue
    
    logger.success(f"Loaded coordinates for {len(coordinates)} contexts from: {coords_path}")
    return coordinates

# ============================================================================
# IDF COMPUTATION
# ============================================================================

def compute_idf_weights(
    phrases: List[str],
    contexts: List[str]
) -> Dict[str, float]:
    """
    Compute IDF (Inverse Document Frequency) weights for phrases.
    
    IDF weights measure the discriminative power of phrases across contexts.
    Rare phrases receive higher weights, while common phrases receive lower
    weights, following the formula:
    
        IDF(phrase) = log(N / df(phrase))
    
    where N is the total number of contexts and df(phrase) is the number
    of contexts containing the phrase.
    
    Args:
        phrases: List of phrases to compute IDF for
        contexts: List of context texts
    
    Returns:
        Dictionary mapping phrase -> IDF weight
    
    Example:
        >>> phrases = ['machine learning', 'the', 'neural network']
        >>> contexts = ['machine learning is...', 'the neural network...']
        >>> idf = compute_idf_weights(phrases, contexts)
        >>> idf['machine learning'] > idf['the']
        True  # 'machine learning' is more discriminative
    
    Note:
        IDF weights are used in doc_fingerprints.py and query_processing.py
        to emphasize discriminative phrases in document representations.
    """
    logger.info(f"Computing IDF weights for {len(phrases)} phrases across {len(contexts)} contexts")
    
    # Count document frequency for each phrase
    df = defaultdict(int)
    
    for context in contexts:
        context_lower = context.lower()
        seen_phrases = set()
        
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower not in seen_phrases:
                if find_phrase_occurrences(context_lower, phrase_lower, use_word_boundaries=True) > 0:
                    df[phrase_lower] += 1
                    seen_phrases.add(phrase_lower)
    
    # Compute IDF weights
    N = len(contexts)
    idf_weights = {}
    
    for phrase in phrases:
        phrase_lower = phrase.lower()
        doc_freq = df.get(phrase_lower, 0)
        
        if doc_freq > 0:
            idf_weights[phrase] = np.log(N / doc_freq)
        else:
            # Assign maximum IDF for phrases not found in any context
            idf_weights[phrase] = np.log(N)
    
    logger.success(f"Computed IDF weights for {len(idf_weights)} phrases")
    return idf_weights


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================

def compute_cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical), with 0 indicating orthogonality.
    
    Formula:
        cos(θ) = (A · B) / (||A|| × ||B||)
    
    Args:
        vec1: First vector (numpy array or sparse matrix)
        vec2: Second vector (numpy array or sparse matrix)
    
    Returns:
        Cosine similarity score in range [-1, 1]
    
    Examples:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([1, 0, 1, 0])
        >>> compute_cosine_similarity(v1, v2)
        1.0  # identical vectors
        
        >>> v3 = np.array([1, 0, 0, 0])
        >>> v4 = np.array([0, 1, 0, 0])
        >>> compute_cosine_similarity(v3, v4)
        0.0  # orthogonal vectors
    
    Note:
        Handles both dense numpy arrays and sparse scipy matrices.
        Returns 0.0 if either vector has zero magnitude.
    """
    # Convert sparse matrices to dense if needed
    if hasattr(vec1, 'toarray'):
        vec1 = vec1.toarray().flatten()
    if hasattr(vec2, 'toarray'):
        vec2 = vec2.toarray().flatten()
    
    # Compute norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    
    return float(similarity)


def compute_jaccard_similarity(
    set1: Set,
    set2: Set
) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Jaccard similarity measures the overlap between two sets as the ratio
    of intersection to union:
    
        J(A, B) = |A ∩ B| / |A ∪ B|
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        Jaccard similarity score in range [0, 1]
    
    Examples:
        >>> s1 = {1, 2, 3, 4}
        >>> s2 = {3, 4, 5, 6}
        >>> compute_jaccard_similarity(s1, s2)
        0.333...  # 2 common / 6 total
        
        >>> s3 = {1, 2, 3}
        >>> s4 = {1, 2, 3}
        >>> compute_jaccard_similarity(s3, s4)
        1.0  # identical sets
    
    Note:
        Returns 0.0 if both sets are empty.
        Useful for comparing sparse fingerprint coordinate sets.
    """
    if not set1 and not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union

# ============================================================================
# Z-ORDER CURVE UTILITIES
# ============================================================================

def xy_to_morton(x: int, y: int) -> int:
    """
    Convert 2D coordinates to Morton code (Z-order curve index).
    
    Morton codes interleave the binary representations of x and y coordinates,
    creating a space-filling curve that preserves spatial locality. Points
    close in 2D space tend to have similar Morton codes.
    
    Algorithm:
        For x=5 (binary: 101) and y=3 (binary: 011):
        Interleave: y1 x1 y0 x0 y2 x2 → 100111 = 39
    
    Args:
        x: X coordinate (non-negative integer)
        y: Y coordinate (non-negative integer)
    
    Returns:
        Morton code (Z-order index)
    
    Examples:
        >>> xy_to_morton(0, 0)
        0
        >>> xy_to_morton(1, 0)
        1
        >>> xy_to_morton(0, 1)
        2
        >>> xy_to_morton(1, 1)
        3
        >>> xy_to_morton(5, 3)
        39
    
    Note:
        Used in phrase_fingerprints.py and doc_fingerprints.py for
        Z-order curve thresholding, which preserves spatial structure
        when selecting top-k bits.
    """
    def part1by1(n: int) -> int:
        """Spread bits of n by inserting a 0 between each bit"""
        n &= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n
    
    return (part1by1(y) << 1) + part1by1(x)


def morton_to_xy(morton: int) -> Tuple[int, int]:
    """
    Convert Morton code back to 2D coordinates.
    
    Inverse operation of xy_to_morton(), extracting the interleaved
    x and y coordinates from the Morton code.
    
    Args:
        morton: Morton code (Z-order index)
    
    Returns:
        Tuple of (x, y) coordinates
    
    Examples:
        >>> morton_to_xy(0)
        (0, 0)
        >>> morton_to_xy(1)
        (1, 0)
        >>> morton_to_xy(2)
        (0, 1)
        >>> morton_to_xy(3)
        (1, 1)
        >>> morton_to_xy(39)
        (5, 3)
    
    Note:
        Useful for debugging and visualization of Z-order traversal.
    """
    def compact1by1(n: int) -> int:
        """Extract every other bit"""
        n &= 0x55555555
        n = (n ^ (n >> 1)) & 0x33333333
        n = (n ^ (n >> 2)) & 0x0F0F0F0F
        n = (n ^ (n >> 4)) & 0x00FF00FF
        n = (n ^ (n >> 8)) & 0x0000FFFF
        return n
    
    x = compact1by1(morton)
    y = compact1by1(morton >> 1)
    return (x, y)


def get_zorder_neighbors(
    x: int,
    y: int,
    grid_size: int,
    radius: int = 1
) -> List[Tuple[int, int]]:
    """
    Get neighboring coordinates within a given radius in Z-order space.
    
    Returns all valid grid coordinates within Manhattan distance 'radius'
    from the given point, useful for spreading activation in semantic space.
    
    Args:
        x: Center X coordinate
        y: Center Y coordinate
        grid_size: Size of the grid (for boundary checking)
        radius: Manhattan distance radius (default: 1)
    
    Returns:
        List of (x, y) coordinate tuples within radius
    
    Examples:
        >>> get_zorder_neighbors(5, 5, 10, radius=1)
        [(4, 5), (6, 5), (5, 4), (5, 6), (4, 4), (4, 6), (6, 4), (6, 6)]
        
        >>> get_zorder_neighbors(0, 0, 10, radius=1)
        [(1, 0), (0, 1), (1, 1)]  # boundary-aware
    
    Note:
        Used in query_processing.py for spreading query fingerprints
        to improve recall by activating nearby semantic regions.
    """
    neighbors = []
    
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            # Skip center point
            if dx == 0 and dy == 0:
                continue
            
            nx = x + dx
            ny = y + dy
            
            # Check boundaries
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbors.append((nx, ny))
    
    return neighbors


# ============================================================================
# FINGERPRINT MANIPULATION
# ============================================================================

def normalize_fingerprint(
    fingerprint: csr_matrix,
    method: str = 'l2'
) -> csr_matrix:
    """
    Normalize a sparse fingerprint vector.
    
    Normalization methods:
    - 'l2': L2 normalization (unit vector), preserves direction
    - 'l1': L1 normalization (sum to 1), preserves relative magnitudes
    - 'binary': Binarize (all non-zero values → 1)
    
    Args:
        fingerprint: Sparse fingerprint matrix (shape: 1 × D)
        method: Normalization method ('l2', 'l1', or 'binary')
    
    Returns:
        Normalized sparse fingerprint matrix
    
    Examples:
        >>> fp = csr_matrix([[1, 2, 0, 3]])
        >>> normalize_fingerprint(fp, 'l2')
        # Returns unit vector with same direction
        
        >>> normalize_fingerprint(fp, 'binary')
        # Returns [[1, 1, 0, 1]]
    
    Raises:
        ValueError: If method is not recognized
    
    Note:
        L2 normalization is standard for cosine similarity computation.
        Binary normalization is useful for pure overlap-based matching.
    """
    if method == 'l2':
        # L2 normalization
        norm = np.sqrt(fingerprint.multiply(fingerprint).sum())
        if norm > 0:
            return fingerprint / norm
        return fingerprint
    
    elif method == 'l1':
        # L1 normalization
        norm = np.abs(fingerprint).sum()
        if norm > 0:
            return fingerprint / norm
        return fingerprint
    
    elif method == 'binary':
        # Binarize
        fp_copy = fingerprint.copy()
        fp_copy.data = np.ones_like(fp_copy.data)
        return fp_copy
    
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def merge_fingerprints(
    fingerprints: List[csr_matrix],
    weights: Optional[List[float]] = None
) -> csr_matrix:
    """
    Merge multiple fingerprints with optional weighting.
    
    Combines multiple sparse fingerprints into a single representation
    by weighted summation. Useful for:
    - Combining phrase fingerprints into document fingerprints
    - Merging multi-query representations
    - Creating composite semantic representations
    
    Args:
        fingerprints: List of sparse fingerprint matrices (same shape)
        weights: Optional list of weights (default: uniform weighting)
    
    Returns:
        Merged sparse fingerprint matrix
    
    Examples:
        >>> fp1 = csr_matrix([[1, 0, 1, 0]])
        >>> fp2 = csr_matrix([[0, 1, 1, 0]])
        >>> merge_fingerprints([fp1, fp2])
        # Returns [[1, 1, 2, 0]]
        
        >>> merge_fingerprints([fp1, fp2], weights=[0.7, 0.3])
        # Returns weighted combination
    
    Raises:
        ValueError: If fingerprints have different shapes
        ValueError: If weights length doesn't match fingerprints length
    
    Note:
        All fingerprints must have the same shape.
        Result is NOT automatically normalized.
    """
    if not fingerprints:
        raise ValueError("Cannot merge empty fingerprint list")
    
    # Validate shapes
    shape = fingerprints[0].shape
    for fp in fingerprints[1:]:
        if fp.shape != shape:
            raise ValueError(f"Shape mismatch: {fp.shape} != {shape}")
    
    # Set uniform weights if not provided
    if weights is None:
        weights = [1.0] * len(fingerprints)
    
    if len(weights) != len(fingerprints):
        raise ValueError(
            f"Weights length {len(weights)} != fingerprints length {len(fingerprints)}"
        )
    
    # Weighted sum
    merged = weights[0] * fingerprints[0]
    for w, fp in zip(weights[1:], fingerprints[1:]):
        merged = merged + w * fp
    
    return merged


def sparsify_fingerprint(
    fingerprint: csr_matrix,
    top_k: int,
    use_zorder: bool = False,
    grid_size: Optional[int] = None,
) -> csr_matrix:
    """
    Sparsify a fingerprint by keeping only the top-k active bits.
    
    Reduces fingerprint density by retaining only the highest-value
    entries, which corresponds to the most strongly activated semantic
    regions. Two selection strategies are supported:
    
    - Standard: Select top-k by value (highest activation first)
    - Z-order:  Select top-k by Morton code order (spatially coherent)
    
    Args:
        fingerprint: Input sparse fingerprint matrix (shape: 1 × D)
        top_k: Number of bits to retain
        use_zorder: If True, use Z-order curve ordering (default: False)
        grid_size: Required when use_zorder=True for coordinate conversion
    
    Returns:
        Sparsified fingerprint with at most top_k non-zero entries
    
    Examples:
        >>> fp = csr_matrix([[0.1, 0.9, 0.0, 0.5, 0.3]])
        >>> sparsify_fingerprint(fp, top_k=2).toarray()
        array([[0. , 0.9, 0. , 0.5, 0. ]])
    
    Raises:
        ValueError: If use_zorder=True but grid_size is not provided
    
    Note:
        Z-order sparsification preserves spatial coherence in the
        semantic grid, which can improve retrieval quality.
    """
    if use_zorder and grid_size is None:
        raise ValueError("grid_size is required when use_zorder=True")

    # Convert to dense for processing
    dense = fingerprint.toarray().flatten()
    nonzero_indices = np.nonzero(dense)[0]

    if len(nonzero_indices) <= top_k:
        return fingerprint  # Already sparse enough

    if use_zorder:
        # Sort nonzero indices by Morton code (Z-order)
        morton_codes = [
            (xy_to_morton(int(idx // grid_size), int(idx % grid_size)), idx)
            for idx in nonzero_indices
        ]
        morton_codes.sort(key=lambda x: x[0])
        selected_indices = [idx for _, idx in morton_codes[:top_k]]
    else:
        # Sort by activation value (descending), keep top_k
        sorted_indices = nonzero_indices[np.argsort(dense[nonzero_indices])[::-1]]
        selected_indices = sorted_indices[:top_k]

    # Build new sparse matrix with only selected indices
    new_dense = np.zeros_like(dense)
    new_dense[selected_indices] = dense[selected_indices]

    return csr_matrix(new_dense.reshape(1, -1))

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_fingerprint(
    fingerprint: csr_matrix,
    grid_size: int,
    min_active: int = 1,
    max_active: Optional[int] = None
) -> bool:
    """
    Validate fingerprint properties.
    
    Checks:
    - Correct shape (1 × grid_size²)
    - Minimum number of active bits
    - Maximum number of active bits (if specified)
    - All values are non-negative
    
    Args:
        fingerprint: Sparse fingerprint matrix
        grid_size: Expected grid size
        min_active: Minimum number of active bits (default: 1)
        max_active: Maximum number of active bits (optional)
    
    Returns:
        True if fingerprint is valid, False otherwise
    
    Examples:
        >>> fp = csr_matrix([[1, 0, 1, 0]])
        >>> validate_fingerprint(fp, grid_size=2, min_active=1)
        True
        
        >>> validate_fingerprint(fp, grid_size=2, min_active=5)
        False  # not enough active bits
    
    Note:
        Use this for quality control in fingerprint generation pipelines.
    """
    expected_dims = grid_size * grid_size
    
    # Check shape
    if fingerprint.shape != (1, expected_dims):
        logger.warning(f"Invalid shape: {fingerprint.shape}, expected (1, {expected_dims})")
        return False
    
    # Check number of active bits
    n_active = fingerprint.nnz
    
    if n_active < min_active:
        logger.warning(f"Too few active bits: {n_active} < {min_active}")
        return False
    
    if max_active is not None and n_active > max_active:
        logger.warning(f"Too many active bits: {n_active} > {max_active}")
        return False
    
    # Check for negative values
    if hasattr(fingerprint, 'data'):
        if np.any(fingerprint.data < 0):
            logger.warning("Fingerprint contains negative values")
            return False
    
    return True


def compute_fingerprint_stats(
    fingerprints: Dict[str, csr_matrix]
) -> Dict[str, float]:
    """
    Compute statistics for a collection of fingerprints.
    
    Computed metrics:
    - Mean sparsity (percentage of zero values)
    - Mean number of active bits
    - Standard deviation of active bits
    - Min/max active bits
    
    Args:
        fingerprints: Dictionary mapping ID -> fingerprint matrix
    
    Returns:
        Dictionary of statistics
    
    Example:
        >>> fps = {'doc1': csr_matrix(...), 'doc2': csr_matrix(...)}
        >>> stats = compute_fingerprint_stats(fps)
        >>> print(stats['mean_active_bits'])
        47.3
    
    Note:
        Useful for quality assessment and hyperparameter tuning.
    """
    if not fingerprints:
        return {}
    
    active_bits = [fp.nnz for fp in fingerprints.values()]
    total_dims = list(fingerprints.values())[0].shape[1]
    
    stats = {
        'n_fingerprints': len(fingerprints),
        'total_dimensions': total_dims,
        'mean_active_bits': np.mean(active_bits),
        'std_active_bits': np.std(active_bits),
        'min_active_bits': np.min(active_bits),
        'max_active_bits': np.max(active_bits),
        'mean_sparsity': 1.0 - (np.mean(active_bits) / total_dims)
    }
    
    return stats


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Ensure NLTK data is available
def _ensure_nltk_data():
    """Download required NLTK data if not present"""
    required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    
    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' else f'corpora/{data_name}')
        except LookupError:
            logger.info(f"Downloading NLTK data: {data_name}")
            nltk.download(data_name, quiet=True)

# Initialize on module import
# _ensure_nltk_data()

def batch_compute_similarities(
    query_fp: csr_matrix,
    doc_fps: List[csr_matrix]
) -> np.ndarray:
    """
    Compute cosine similarities between query and multiple documents efficiently.
    
    Args:
        query_fp: Query fingerprint (1 × N sparse matrix)
        doc_fps: List of document fingerprints (each 1 × N sparse matrix)
        
    Returns:
        Array of similarity scores, one per document
    """
    from scipy.sparse import vstack
    
    # Stack documents into (num_docs, N) matrix
    doc_matrix = vstack(doc_fps)
    
    # Convert query to dense for computation
    query_dense = query_fp.toarray().flatten()
    query_norm = np.linalg.norm(query_dense)
    
    if query_norm == 0:
        return np.zeros(len(doc_fps))
    
    # Compute dot products: (num_docs, N) @ (N,) → (num_docs,)
    dot_products = doc_matrix.dot(query_dense)
    
    # Compute document norms: sqrt of sum of squares per row
    doc_norms = np.sqrt(np.array(doc_matrix.multiply(doc_matrix).sum(axis=1)).flatten())
    
    # Avoid division by zero
    doc_norms[doc_norms == 0] = 1e-10
    
    # Cosine similarity: dot / (norm_q * norm_d)
    similarities = dot_products / (query_norm * doc_norms)
    
    return similarities


def get_fingerprint_overlap(
    fp1: csr_matrix,
    fp2: csr_matrix
) -> Tuple[int, int, int]:
    """
    Compute overlap statistics between two fingerprints.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Tuple of (intersection_size, fp1_only, fp2_only)
    """
    # Get active indices
    indices1 = set(fp1.indices)
    indices2 = set(fp2.indices)
    
    intersection = len(indices1 & indices2)
    fp1_only = len(indices1 - indices2)
    fp2_only = len(indices2 - indices1)
    
    return intersection, fp1_only, fp2_only


def visualize_fingerprint(
    fingerprint: csr_matrix,
    grid_size: int,
    title: str = "Fingerprint",
    output_path: Optional[Path] = None
) -> None:
    """
    Create heatmap visualization of fingerprint.
    
    Args:
        fingerprint: Sparse fingerprint matrix (1 × N)
        grid_size: Grid dimension
        title: Plot title
        output_path: Optional path to save figure
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert to dense 2D
        dense_fp = fingerprint.toarray().reshape(grid_size, grid_size)
        
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            dense_fp,
            annot=False,
            cmap='YlOrRd',
            cbar=True,
            square=True,
            cbar_kws={'label': 'Activation'}
        )
        
        plt.title(title, fontsize=12, pad=10)
        plt.xlabel('Grid X', fontsize=10)
        plt.ylabel('Grid Y', fontsize=10)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.warning("Matplotlib not available for visualization")
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")


def export_fingerprints_to_numpy(
    fingerprints: Dict[str, csr_matrix],
    output_path: Path,
    grid_size: int
) -> None:
    """
    Export fingerprints to dense numpy format for analysis.
    
    Args:
        fingerprints: Dictionary of sparse fingerprints
        output_path: Output .npz file path
        grid_size: Grid dimension
    """
    dense_fps = {}
    for key, fp in fingerprints.items():
        dense_fps[key] = fp.toarray().reshape(grid_size, grid_size)
    
    np.savez_compressed(output_path, **dense_fps)
    logger.success(f"Exported {len(fingerprints)} fingerprints to {output_path}")


def compute_fingerprint_diversity(
    fingerprints: Dict[str, csr_matrix],
    sample_size: int = 100
) -> Dict[str, float]:
    """
    Compute diversity metrics for a set of fingerprints.
    
    Args:
        fingerprints: Dictionary of fingerprints
        sample_size: Number of pairs to sample for diversity computation
        
    Returns:
        Dictionary of diversity metrics
    """
    import random
    
    if len(fingerprints) < 2:
        return {'avg_similarity': 0.0, 'diversity_score': 1.0}
    
    fp_list = list(fingerprints.values())
    similarities = []
    
    # Sample pairs
    num_samples = min(sample_size, len(fp_list) * (len(fp_list) - 1) // 2)
    
    for _ in range(num_samples):
        i, j = random.sample(range(len(fp_list)), 2)
        sim = compute_cosine_similarity(fp_list[i], fp_list[j])
        similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    diversity = 1 - avg_sim
    
    return {
        'avg_similarity': float(avg_sim),
        'diversity_score': float(diversity),
        'num_samples': num_samples
    }

if __name__ == "__main__":

    # Quick sanity check — run this once in a REPL or a test
    print(lemmatize_token("deeper", "JJR"))   # should print: deep
    print(lemmatize_token("deepest", "JJS"))  # should print: deep
    print(lemmatize_token("better", "JJR"))   # should print: good
