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
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk import pos_tag
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix, lil_matrix
from rich import print
from loguru import logger
from loguru import logger as _base_logger
import numpy as np
from functools import lru_cache
import json, os
import sys


# Set up NLTK data path relative to project root
_nltk_path = Path(__file__).resolve().parent.parent / "nltk_data"
if _nltk_path.exists():
    # nltk.data.path.insert(0, "D:\\darsi\\ms\\Thesis\\Dr.Banaie\\code050302\\nltk_data")
    nltk.data.path.insert(0, str(_nltk_path))
    os.environ['NLTK_DATA'] = str(_nltk_path)

import re
_ARABIC_SCRIPT = re.compile(r'[\u0600-\u06FF]')
from hazm import Normalizer, word_tokenize as hazm_word_tokenize
normalizer = Normalizer()

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
    """
    Determine if a single word is too generic to carry semantic meaning.

    Generic words are filtered out during phrase expansion to maintain
    semantic quality. A word is considered generic if it meets any of:
    - Too short (< min_length characters)
    - Common stop word (articles, prepositions, etc.)
    - Pure numeric string or contains non-alpha characters
    - Domain acronyms (ai, ml, nlp, etc.) are preserved.

    Args:
        word: Input word to evaluate
        min_length: Minimum character length threshold (default: 3)

    Returns:
        True if word is generic and should be filtered, False otherwise
    """
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

@lru_cache(maxsize=50000)
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


def split_arabic_english(text: str):
    ar_positions = [m.start() for m in _ARABIC_SCRIPT.finditer(text)]
    if not ar_positions:
        return "", text.strip()

    last_ar = max(ar_positions)
    ar_raw = text[:last_ar + 1]
    en_raw = text[last_ar + 1:]

    arabic_text = ar_raw.rstrip(',').strip().strip('"').strip()
    english_text = en_raw.strip().strip('"').strip()

    return arabic_text, english_text

def split_id_arabic_english(line: str):
    comma1 = line.index(',')
    ctx_id = line[:comma1].strip()
    rest = line[comma1 + 1:]

    ar_positions = [m.start() for m in _ARABIC_SCRIPT.finditer(rest)]
    if not ar_positions:
        return ctx_id, "", rest.strip()

    last_ar = max(ar_positions)
    ar_raw = rest[:last_ar + 1]
    en_raw = rest[last_ar + 1:]

    arabic_text = ar_raw.rstrip(',').strip().strip('"').strip()
    english_text = en_raw.strip().strip('"').strip()

    return ctx_id, arabic_text, english_text

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

@lru_cache(maxsize=32768)
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
    tokens = nltk_word_tokenize(text)
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

def _norm_ar(t: str) -> str:
    """Fast character normalization to match _AR_FUNCTION_WORDS."""
    t = t.replace("\u064b", "").replace("\u064c", "").replace("\u064d", "")
    t = t.replace("\u064e", "").replace("\u064f", "").replace("\u0650", "")
    t = t.replace("\u0651", "").replace("\u0652", "").replace("\u0670", "")
    t = t.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
    t = t.replace("\u0649", "\u064a")  # alif-maqsura → ya
    t = t.replace("\u0629", "\u0647")  # ta-marbuta → ha
    t = t.replace("\u06cc", "\u064a")  # farsi yeh → ya
    t = t.replace("\u06a9", "\u0643")  # keheh → kaf
    t = t.replace("\u06af", "\u0643")  # gaf → kaf
    t = t.replace("\u06c0", "\u0647")  # heh yeh → ha
    t = t.replace("\u0671", "\u0627")  # alif-wasla → alif
    t = t.replace("\u0626", "\u064a")  # yeh-with-hamza → ya (so اولئك→اوليك matches اوليك)
    return t


def extract_raw_phrases_ar_fa(text: str) -> Set[str]:
    phrases = set()

    text = normalizer.normalize(text)
    # tokens = hazm_word_tokenize(text)
    # tokens = text.split()
    tokens = nltk_word_tokenize(text)
    if not tokens:
        return phrases

    # Pre-normalize each token to match _AR_FUNCTION_WORDS codepoints
    normed = [_norm_ar(t) for t in tokens]

    # unigram — only keep tokens ≥ 2 chars that are NOT in the function-word list
    for i, tok in enumerate(tokens):
        nt = normed[i]
        if len(nt) >= 2 and nt not in _AR_FUNCTION_WORDS:
            phrases.add(nt)

    # POS-tag filtered bigram — only accept Noun/Adj/Noun-like + Noun patterns,
    # mirroring the English fallback extractor's strategy.
    # Also reject if either normalized token is a known function word (safety
    # net for NLTK's English-trained tagger which sometimes tags Arabic
    # function words / verbs as nouns).
    NOUN_LIKE = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VBN'}
    tagged = nltk.pos_tag(tokens)
    for i in range(len(tagged) - 1):
        w1, t1 = tagged[i]
        w2, t2 = tagged[i + 1]
        n1, n2 = normed[i], normed[i + 1]
        if t1 in NOUN_LIKE and t2.startswith('N'):
            if n1 not in _AR_FUNCTION_WORDS and n2 not in _AR_FUNCTION_WORDS:
                phrases.add(f"{n1} {n2}")

    # trigram — all 3 tokens must be noun/adjective-like (strict POS filter)
    for i in range(len(tagged) - 2):
        w1, t1 = tagged[i]
        w2, t2 = tagged[i + 1]
        w3, t3 = tagged[i + 2]
        n1, n2, n3 = normed[i], normed[i + 1], normed[i + 2]
        if (t1 in NOUN_LIKE and t2 in NOUN_LIKE and t3 in NOUN_LIKE
            and all(n not in _AR_FUNCTION_WORDS for n in (n1, n2, n3))
            and all(len(n) >= 2 for n in (n1, n2, n3))):
            phrases.add(f"{n1} {n2} {n3}")

    return phrases


# ---------------------------------------------------------
# Comprehensive Quranic Arabic function-word list
# Forms are in normalized script (أإآ → ا, ة → ه, ى → ي)
# ---------------------------------------------------------
_AR_FUNCTION_WORDS = {
    # single-letter clitics (will only match when hazm isolates them)
    "و", "ف", "ب", "ل", "ك", "س",
    # pronouns (personal)
    "هو", "هي", "هم", "هن", "هما", "انا", "نحن", "انت", "انتم",
    "انتن", "انتو", "انتما", "انتي", "اياي", 'اياك', 'اياه',
    # demonstratives
    "هذا", "هذه", "هذان", "هاتان", "هولاء", "ذلك", "تلك",
    "اوليك", "هنا", "هناك", "هكذا", "كذلك",
    # relative pronouns
    "الذي", "التي", "الذين", "الذان", "اللتان", "اللاتي",
    # clitic + relative pronoun compounds (hazm does NOT split these)
    "والذي", "والتي", "والذين", "والذان", "فالذي", "فالتي", "فالذين",
    "بالذي", "بالتي", "بالذين", "للذين",
    # prepositions
    "من", "في", "الي", "الى", "على", "علي", "عن", "مع",
    "حتى", "منذ", "دون", "بين", "فوق", "تحت", "خلف", "وراء",
    "قدام", "امام", "عند", "لدى", "لدي", "حيال",
    # conjunctions
    "ثم", "او", "ام", "بل", "لكن", "لکن", "لعل", "كي",
    "کي", "لاجل", "کی", "لکی",
    # clitic + conjunction compounds
    "ولكن", "ولکن", "فلكن", "فيلك", "وبل", "واما", "فاما",
    # negation
    "لا", "لم", "لن", "لما", "ليس", "غير", "الا", "سوى",
    "عدا", "خلا", "حاشا", "لست", "لستم", "ليسا", "ليسوا",
    # negation / relative / interrogative particle
    "ما",
    # particles
    "قد", "هل", "س", "سوف", "إن", "ان", "کأن", "كان",
    "کان", "لقد", "انما", "انّ", "ان", "فان", "وان",
    # vocative / address particles
    "ايها", "ايتها", "ايها", "یاایها", "ياايها", "يايها",
    "یایها", "يایها",
    # interrogatives
    "ماذا", "كيف", "اين", "متى", "ايان", "كم", "اي", "اى",
    "لماذا", "اينما", "حيثما", "كيفما",
    # conditionals
    "اذا", "لو", "لولا", "لئن", "کلما", "كلما", "مهما",
    "اذ", "إذا", "لما", "حيث",
    # time / place adverbs (function-adjacent)
    "حين", "حیث", "عندما", "بعد", "قبل", "قط", "ذات",
    "حینما", "بينما", "بعدما", "قبلما",
    # reporting verbs (extremely high frequency, low semantic value in n-grams)
    "قال", "قل", "قالوا", "قالت", "قيل", "يقول", "يقولون",
    "قلنا", "قلتم", "يقال", "فقال", "وقال",
    # common clitic+stopword compounds seen in Quran vocab
    "فما", "فلا", "فمن", "فهل", "فلن", "فلم", "فلما", "فان",
    "وما", "ولا", "ومن", "وله", "ولم", "ولن", "وهو", "وهي",
    "وهم", "وله", "ولها", "ولهم", "ولكم", "ولنا", "ولي",
    "واذا", "فاذا", "ولقد", "ولئن", "ولما",
    "بما", "لمن", "ممن", "مما", "فيم", "فیم", "فبما", "وبما",
    "ومما", "وفیما", "وفيما", "ففی", "ففي",
    "واذ", "فاذ",
    "وانتم", "وأنتم", "فانتم", "فأنتم",
    "لعلکم", "لذلك", "ولذلك",
    # "to be" verb conjugations (function-level)
    "كانوا", "کانوا", "كنتم", "كنت", "كن", "كنا", "كن", "كون",
    "يكون", "تكون", "اكون", "اكن", "نكون",
    "تكاد", "يكاد", "تظل", "يظل", "تزال", "يزال",
    # preposition+pronoun compounds
    "به", "له", "لها", "لهم", "لكم", "لك", "لنا", "لي",
    "بك", "بكم", "بنا", "بي", "بهم", "بها", "بكن",
    "فيه", "فيها", "فيهم", "فينا", "فيكم",
    "منه", "منها", "منهم", "منكم", "منك", "منا", "مهن",
    "عنه", "عنها", "عنهم", "عنكم", "عنك",
    "اليه", "اليها", "اليهم", "اليكم", "اليك",
    "عليه", "عليها", "عليهم", "عليكم", "عليك", "علينا",
    "عنده", "عندها", "عندهم", "عندكم", "عندنا",
    "دونه", "دونها", "دونهم", "دونكم",
    "بينهم", "بينكم", "بيننا", "بينهما",
    "معه", "معها", "معهم", "معكم", "معنا",
    # conjunction+pronoun compounds
    "وبه", "وله", "ولهم", "ولها", "ولكم", "ولنا", "ولي",
    "وفيه", "وفيها", "وفيهم",
    "وعليه", "وعليها", "وعليهم",
    # pronoun+verb / particle+pronoun compounds
    "اني", "انه", "انها", "انهم", "انكم", "انك", "انا", "اننا",
    "کانه", "کانهم", "كانما",
    # intensifiers / quantifiers (function level)
    "کل", "كل", "كلا", "کلا", "جميع", "اجمع", "معا",
}

# Auto-add normalized variants so that normalizer output always matches
_AR_FUNCTION_WORDS = _AR_FUNCTION_WORDS | {
    w.replace('\u0622','\u0627').replace('\u0625','\u0627').replace('\u0623','\u0627')
     .replace('\u0629','\u0647').replace('\u0649','\u064A')
    for w in _AR_FUNCTION_WORDS
}

_AR_CLITICS = {"ال", "و", "ف", "ب", "ل", "ك", "ک", "س", "بال", "فل", "ول", "فب"}

# ---------------------------------------------------------
# Quranic important-term whitelist
# Phrases in this set bypass the min_freq filter in Step 1
# so that rare but semantically/theologically important
# Quranic vocabulary is always preserved.
# Forms must be pre-normalized with normalize_arabic_phrase.
# ---------------------------------------------------------
_QURANIC_KEEP: Set[str] = {
    # 2349 and 3706
    "طه", "يس",
    "يسٓ",
    "یس", "یسٓ",
    # # Verse 5774
    # "كرام", "برره", "كرام برره",
    # # Verse 5799
    # "ترهقها", "قتره", "ترهقها قتره",
    # # Verse 5826
    # "تذهبون", "فاين", "فاين تذهبون",
    # # Verse 5845
    # "بغايبين",
    # # Verse 5915
    # "قعود",
    # # Verse 5931
    # "لوح", "محفوظ", "لوح محفوظ",
    # # Verse 5934
    # "النجم", "الثاقب", "النجم الثاقب",
    # # Verse 6055
    # "انبعث", "اشقيها", "انبعث اشقيها",
    # # Verse 6147
    # "ضبحا", "والعديت", "والعديت ضبحا",
    # # Verse 6224
    # "يلد", "يولد", "يلد يولد",

    # # English equivalents (normalized by expand_phrases → normalize_phrase)
    # # Verse 5799
    # "blackness",
    # # Verse 5931
    # "preserved", "preserved slate", "slate",
    # # Verse 5934
    # "piercing", "piercing star", "star",
    # # Verse 6147
    # "panting", "racer",
    # # Verse 6224
    # "begets",
}

# Also build a set of trie-like prefixes for fast lookup of clitic-attached forms
_AR_CLITIC_PREFIXES = tuple(sorted(_AR_CLITICS, key=len, reverse=True))


def normalize_arabic_phrase(text: str):
    ARABIC_STOPWORDS = _AR_FUNCTION_WORDS
    # 1. normalize unicode (same as _norm_ar — no hazm double-normalization)
    text = text.strip()
    # remove diacritics and superscript alef
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # alef normalization
    text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064a")  # alif-maqsura → ya
    text = text.replace("\u0629", "\u0647")  # ta-marbuta → ha
    text = text.replace("\u06cc", "\u064a")  # farsi yeh → ya
    text = text.replace("\u06a9", "\u0643")  # keheh → kaf
    text = text.replace("\u06af", "\u0643")  # gaf → kaf
    text = text.replace("\u06c0", "\u0647")  # heh-yeh → ha
    text = text.replace("\u0671", "\u0627")  # alif-wasla → alif
    text = text.replace("\u0626", "\u064a")  # yeh-with-hamza → ya

    # 2. tokenize
    # tokens = hazm_word_tokenize(text)
    # tokens = text.split()
    tokens = nltk_word_tokenize(text)

    # 3. strip clitic prefixes from each token
    #    Try clitics longest-first; only accept if the stem is >= 2 chars
    #    AND is not itself a function word (prevents over-stripping like
    #    الله → له which is a stopword).
    stripped = []
    for t in tokens:
        stem = t
        for cl in _AR_CLITIC_PREFIXES:
            if t.startswith(cl) and len(t) > len(cl):
                s = t[len(cl):]
                if len(s) >= 2 and s not in ARABIC_STOPWORDS:
                    stem = s
                    break
        stripped.append(stem)
    tokens = stripped

    # 4. stopword removal
    tokens = [
        t for t in tokens
        if t not in ARABIC_STOPWORDS
    ]

    # 5. remove very short tokens
    tokens = [
        t for t in tokens
        if len(t) >= 2
    ]

    # 6. reject empty
    if not tokens:
        return None

    # 7. reject too long
    if len(tokens) > 5:
        return None

    # 8. structural validation:
    #    multi-word phrases must contain at least one content word
    #    (≥ 3 chars AND not in function-word list)
    if len(tokens) > 1:
        has_content = any(
            len(t) >= 3 and t not in ARABIC_STOPWORDS
            for t in tokens
        )
        if not has_content:
            return None

    # 9. single-token phrases: ensure it's not a known compound stopword
    if len(tokens) == 1:
        t = tokens[0]
        if t in ARABIC_STOPWORDS:
            return None
        # reject clitic+single-char combinations (pure function words:
        # "به" = بـ + ه, "له" = لـ + ه, "لك" = لـ + ك, etc.)
        for cl in _AR_CLITICS:
            if t.startswith(cl) and len(t) - len(cl) <= 1:
                return None

    return " ".join(tokens)

def expand_phrases(
    phrases: List[str],
    context_text: Optional[str],
    remove_verbs  : bool = False,
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
            norm = normalize_phrase(candidate, remove_verbs=remove_verbs)
            
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
        This function does NOT normalize text.
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
#  Used by: query_processing.py (Step 7)
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
    return phrase_fps #csr_matrix


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


# ============================================================================
# Z-ORDER CURVE UTILITIES
# ============================================================================

def _spread_bits(value: int) -> int:
    """Spread bits of a 16-bit integer by inserting a 0 between each bit."""
    value &= 0x0000FFFF
    value = (value | (value << 8))  & 0x00FF00FF
    value = (value | (value << 4))  & 0x0F0F0F0F
    value = (value | (value << 2))  & 0x33333333
    value = (value | (value << 1))  & 0x55555555
    return value


def _compact_bits(value: int) -> int:
    """Extract every other bit from a 32-bit integer (inverse of _spread_bits)."""
    value &= 0x55555555
    value = (value ^ (value >> 1)) & 0x33333333
    value = (value ^ (value >> 2)) & 0x0F0F0F0F
    value = (value ^ (value >> 4)) & 0x00FF00FF
    value = (value ^ (value >> 8)) & 0x0000FFFF
    return value


def xy_to_morton(x: int, y: int, grid_size: int = None) -> int:
    """
    Convert 2D coordinates to Morton code (Z-order curve index).
    
    Morton codes interleave the binary representations of x and y coordinates,
    creating a space-filling curve that preserves spatial locality.
    
    Args:
        x: X coordinate (non-negative integer)
        y: Y coordinate (non-negative integer)
        grid_size: Ignored; kept for API compatibility with Step 3-6 callers.
    
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
    """
    return _spread_bits(x) | (_spread_bits(y) << 1)


def xy_to_morton_vectorized(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Vectorized Morton encoding for arrays of coordinates.

    Args:
        xs: 1D array of X coordinates (non-negative integers)
        ys: 1D array of Y coordinates (non-negative integers)

    Returns:
        1D array of Morton codes (same length as input)
    """
    def _spread_bits_vec(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint32) & 0x0000FFFF
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    return _spread_bits_vec(xs) | (_spread_bits_vec(ys) << 1)


def morton_to_xy(index: int, grid_size: int = None) -> Tuple[int, int]:
    """
    Convert Morton (Z-order) index back to 2D coordinates.
    
    Inverse operation of xy_to_morton().
    
    Args:
        index: Morton code (Z-order index)
        grid_size: Ignored; kept for API compatibility with visualizer callers.
    
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
    """
    x = _compact_bits(index)
    y = _compact_bits(index >> 1)
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
