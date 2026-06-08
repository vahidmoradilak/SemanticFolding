#!/usr/bin/env python3
"""
query_processing.py — Step 6 of the Semantic Folding Pipeline

Processes user queries by extracting phrases, constructing a sparse
query fingerprint, optionally applying Z-order spreading, and ranking
documents by cosine similarity against Step-5 document fingerprints.

Pipeline position
-----------------
Step 1  phrase_extractor.py    → phrases.txt
Step 2  term_context.py        → phrase_context_matrix.*
Step 3  semantic_space.py      → grid layout
Step 4  phrase_fingerprints.py → phrase_fingerprints/
Step 5  doc_fingerprints.py    → doc_fingerprints/
Step 6  query_processing.py    → ranked results          ← THIS FILE

Consistency guarantee
---------------------
Query phrase extraction follows the **identical** three-stage pipeline
used in Steps 1 and 5:

    raw query text
        └─ extract_raw_phrases_*()     # spaCy noun chunks / NLTK n-grams
                └─ normalize_phrase()  # lowercase → stopwords → lemmatize
                        └─ expand_phrases()   # sub-phrase generation
                                └─ vocab filter (keys of phrase_fingerprints)

The ``--remove-verbs``, ``--no-spacy``, ``--no-filter-generic``, and
``--min-word-length`` flags **must** be set identically to the values used
in Step 1 (``phrase_extractor.py``) so that query normalisation produces
the same token sequences as those stored in the vocabulary.

Usage
-----
    python query_processing.py \\
        --query "machine learning algorithms" \\
        --phrase-fp-dir  outputs/run/phrase_fingerprints/ \\
        --doc-fp-dir     outputs/run/doc_fingerprints/ \\
        --grid-size      16 \\
        --top-k          10 \\
        --weighting      idf \\
        --normalization  l2 \\
        --spreading-steps 1
"""

from __future__ import annotations

import hashlib, argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from scipy.sparse import csr_matrix

from phrase_extractor import (
    SPACY_AVAILABLE,
    extract_raw_phrases_fallback,
    extract_raw_phrases_spacy,
)
from lib import (
    expand_phrases,
    get_zorder_neighbors,
    load_document_fingerprints,
    load_phrase_fingerprints_sparse,
    normalize_fingerprint,
    normalize_phrase,
)
SPARCITY_GAURD=0.005
# ─────────────────────────────────────────────────────────────────────────────
# Logging — level driven by LOG_LEVEL env var (default: INFO)
# ─────────────────────────────────────────────────────────────────────────────
from lib import get_logger
logger = get_logger("query_processing")

# ── spaCy bootstrap ───────────────────────────────────────────────────────────
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.debug("spaCy imported successfully")
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.success("spaCy model 'en_core_web_sm' loaded")
    except OSError:
        logger.warning("spaCy model not found — run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spaCy not installed — falling back to NLTK extraction")
    SPACY_AVAILABLE = False

# NLTK only needed when spaCy is unavailable
if not SPACY_AVAILABLE:
    logger.debug("Importing NLTK fallback tokenizer and POS tagger")
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag


# ─────────────────────────────────────────────────────────────────────────────
# QUERY EXPANSION: Map OOV query terms to nearest in-vocabulary phrases
# ─────────────────────────────────────────────────────────────────────────────


def build_vocab_fingerprint_index(
    phrase_fp_dir: str,
    phrase_vocab: Set[str],
) -> Dict[str, np.ndarray]:
    """
    Load all phrase fingerprints into memory as dense float32 vectors.

    Iterates over every phrase in ``phrase_vocab``, reconstructs the
    filename that Step 4 (``phrase_fingerprints.py``) used when saving
    (spaces → underscores, slashes → hyphens), and converts the stored
    sparse active-bit list into a dense binary vector.

    This index is consumed by :func:`expand_oov_query_terms` to find
    the nearest in-vocabulary neighbours for OOV query terms via batch
    cosine similarity.

    Parameters
    ----------
    phrase_fp_dir : str
        Directory containing ``<phrase>.json`` fingerprint files
        produced by Step 4.
    phrase_vocab : Set[str]
        Complete set of normalised phrase strings (keys of the loaded
        phrase fingerprint dictionary).

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of ``{phrase: dense_vector}`` where each vector has
        shape ``(grid_size²,)`` and dtype ``float32``.  Phrases whose
        JSON file is missing on disk are silently skipped.

    Notes
    -----
    - Loading the full vocabulary into RAM can be expensive for large
      corpora.  Consider caching the result if called repeatedly.
    - The ``grid_size`` is read from each JSON file individually, so
      mixed-size corpora are handled correctly (though unusual).
    """
    index: Dict[str, np.ndarray] = {}
    skipped = 0

    logger.debug(
        f"Building vocab fingerprint index from '{phrase_fp_dir}' "
        f"({len(phrase_vocab)} phrases in vocab)"
    )

    for phrase in phrase_vocab:
        # Reconstruct the filename exactly as Step 4 saved it
        safe_name = phrase.replace(" ", "_").replace("/", "-")
        fp_path   = os.path.join(phrase_fp_dir, f"{safe_name}.json")

        if not os.path.exists(fp_path):
            # Missing file is expected for low-frequency phrases pruned
            # before Step 4 ran — not an error, just skip silently
            skipped += 1
            continue

        with open(fp_path, "r") as f:
            data = json.load(f)

        grid_size   = data.get("grid_size", 128)
        active_bits = data.get("active_bits", [])
        dim         = grid_size * grid_size

        # Convert sparse active-bit indices → dense binary vector
        vec              = np.zeros(dim, dtype=np.float32)
        vec[active_bits] = 1.0
        index[phrase]    = vec

        logger.debug(
            f"  [LOADED] '{phrase}' — {len(active_bits)} active bits "
            f"/ {dim} dims ({len(active_bits)/dim*100:.2f}% density)"
        )

    logger.debug(
        f"Vocab fingerprint index built: {len(index)} loaded, "
        f"{skipped} skipped (file not found)"
    )
    return index

def expand_oov_query_terms(
    oov_terms         : List[str],
    vocab_fp_index    : Dict[str, np.ndarray],
    phrase_fp_dir     : str,
    grid_size         : int   = 128,
    top_k_per_term    : int   = 5,
    min_similarity    : float = 0.2,
    penalize_generic  : bool  = True,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find the top-k most similar in-vocabulary phrases for each OOV term.

    For each term in ``oov_terms`` (phrases extracted from the query that
    are absent from the phrase fingerprint vocabulary), an anchor vector is
    built by summing the fingerprints of any in-vocabulary sub-tokens found
    in the OOV phrase.

    Anchor quality is judged by how many of the OOV term's tokens were
    found in the vocabulary.  Three cases are handled:

    1. **No sub-tokens in vocab** → fall back to a character n-gram
       pseudo-fingerprint (:func:`_pseudo_fingerprint`).  Carries no
       semantic information but may still catch morphological neighbours.

    2. **Too few sub-tokens** → skip this OOV term entirely.
       The minimum required depends on phrase length:

       - Single-token OOV : ``min_required = 1``  (the one token must match)
       - Multi-token OOV  : ``min_required = max(2, len(tokens) // 2)``

       Requiring at least **2** matched tokens for multi-token OOVs prevents
       a single-word anchor (e.g. ``brain`` from ``brain facilitate``) from
       dominating cosine similarity and surfacing wholly unrelated vocabulary
       phrases — the root cause of the ``sim=1.000`` noise observed in the
       debug log (Bug #4).

    3. **Enough sub-tokens** → use the summed real fingerprints as a
       semantically grounded anchor and proceed with batch cosine similarity.

    Before entering the per-term loop, OOV terms are **deduplicated by
    anchor key** (Bug Fix #3).  Two OOV terms that map to the same
    frozenset of in-vocabulary sub-tokens would produce identical anchor
    vectors and therefore identical results.  Only the first such term
    (the *winner*) is processed; the rest are *aliased* to it and receive
    the winner's result list after the loop, at zero extra cost.

    Example (from the debug log)::

        'human brain'            → anchor_key = frozenset({'human', 'brain'})
        'human brain facilitate' → anchor_key = frozenset({'human', 'brain'})
                                   # 'facilitate' is itself OOV → dropped
        ⟹  'human brain facilitate' is aliased to 'human brain'

    Parameters
    ----------
    oov_terms : List[str]
        Normalised phrases from the query that are absent from the
        phrase fingerprint vocabulary.
    vocab_fp_index : Dict[str, np.ndarray]
        Dense vector index built by :func:`build_vocab_fingerprint_index`.
        Keys are vocabulary phrases; values are flat ``float32`` SDR
        vectors of length ``grid_size ** 2``.
        An empty dict causes an immediate early return.
    phrase_fp_dir : str
        Fingerprint directory (passed through for potential future use;
        not read directly in this function).
    grid_size : int, optional
        SDR grid side length.  Must match the value used throughout the
        pipeline (default: ``128``).
    top_k_per_term : int, optional
        Maximum number of vocabulary matches to return per OOV term
        (default: ``5``).
    min_similarity : float, optional
        Cosine similarity threshold below which matches are discarded
        (default: ``0.2``).
    penalize_generic : bool, optional
        When ``True``, halve the similarity score for single-word matches
        that appear in the built-in ``GENERIC_TERMS`` set, reducing noise
        from high-frequency function-like words (default: ``True``).

    Returns
    -------
    Dict[str, List[Tuple[str, float]]]
        Mapping of ``{oov_term: [(vocab_phrase, similarity), ...]}``
        sorted descending by similarity.  OOV terms with no match above
        ``min_similarity`` are omitted from the result.  Aliased duplicate
        OOV terms share the winner's result list (same object reference).

    Notes
    -----
    - Vocab matrix and L2 norms are precomputed **once** before the loop
      — O(V) instead of O(V × T).
    - The deduplication pre-pass (Fix #3) runs in O(T) and eliminates
      redundant cosine-similarity sweeps over the full vocab matrix.
    - The minimum-2-token anchor guard (Fix #4) eliminates the
      ``sim=1.000`` sparse-collision noise observed when a single generic
      token (e.g. ``brain``) was the sole anchor for a multi-token OOV
      phrase (e.g. ``brain facilitate``).
    - Single-word OOV terms are **never** expanded to other single-word
      phrases (mono→mono guard) to suppress generic high-frequency noise.
    - The redundancy filter (skip phrase == oov_term or phrase ∈ tokens)
      is applied **before** the ``top_k`` counter is incremented, so
      filtered-out phrases do not consume result slots.
    - Generic-term penalty is re-checked against ``min_similarity`` after
      the score is halved, so borderline generic matches are cleanly
      dropped rather than kept at a sub-threshold score.

    Raises
    ------
    ValueError
        Not raised explicitly, but a ``grid_size`` inconsistent with the
        vectors stored in ``vocab_fp_index`` will cause a shape mismatch
        in the matrix multiply at STEP 3.
    """
    # ── High-frequency words whose fingerprints are too generic to be
    #    useful as expansion targets.  Matches are kept but score-halved.
    GENERIC_TERMS = {
        "social", "structure", "network", "system", "process",
        "area", "group", "level", "form", "type", "factor",
        "impact", "effect", "result", "change", "increase",
    }

    # ─────────────────────────────────────────────────────────────────────
    # Guard: nothing to do without a vocab index
    # ─────────────────────────────────────────────────────────────────────
    if not vocab_fp_index:
        logger.warning("Vocab fingerprint index is empty — skipping OOV expansion")
        return {}

    logger.debug(
        f"OOV expansion: {len(oov_terms)} OOV terms against "
        f"{len(vocab_fp_index)}-phrase index "
        f"(top_k={top_k_per_term}, min_sim={min_similarity})"
    )

    # ─────────────────────────────────────────────────────────────────────
    # Build vocab matrix once — reused for every OOV term
    #   vocab_matrix : (V, D)  float32
    #   vocab_norms  : (V, 1)  float32  — precomputed to avoid O(V×T) norm calls
    # ─────────────────────────────────────────────────────────────────────
    vocab_phrases = list(vocab_fp_index.keys())
    vocab_matrix  = np.stack(
        [vocab_fp_index[p] for p in vocab_phrases], axis=0
    )                                                   # shape: (V, D)

    # vocab_norms = np.linalg.norm(vocab_matrix, axis=1, keepdims=True) 
    #####
    # --------------------------------------------------
    # reduce memory usage
    # --------------------------------------------------

    vocab_matrix = vocab_matrix.astype(np.float16)

    # --------------------------------------------------
    # chunked norm computation
    # --------------------------------------------------

    batch_size = 5000

    all_norms = []

    for i in range(0, vocab_matrix.shape[0], batch_size):

        batch = vocab_matrix[i:i + batch_size]

        batch_norms = np.linalg.norm(
            batch,
            axis=1,
            keepdims=True
        )

        all_norms.append(batch_norms)

    vocab_norms = np.vstack(all_norms)

    ####

    vocab_norms = np.where(vocab_norms == 0, 1e-9, vocab_norms)    # shape: (V, 1)

    dim = grid_size * grid_size
    expansions: Dict[str, List[Tuple[str, float]]] = {}

    # ═════════════════════════════════════════════════════════════════════
    # FIX #3 — Deduplicate OOV terms that resolve to identical anchor sets
    # ═════════════════════════════════════════════════════════════════════
    #
    # Motivation
    # ----------
    # The anchor vector for an OOV term is built by summing the vocab
    # fingerprints of whichever of its sub-tokens ARE in the vocabulary.
    # Sub-tokens that are themselves OOV contribute nothing.
    #
    # Consequence: two OOV phrases whose in-vocab sub-token sets are
    # identical produce byte-for-byte identical anchor vectors, and
    # therefore identical cosine-similarity rankings — processing both
    # wastes O(V) dot products for zero new information.
    #
    # Observed case (debug log)
    # --------------------------
    #   'human brain'            → in-vocab sub-tokens = {'human', 'brain'}
    #   'human brain facilitate' → in-vocab sub-tokens = {'human', 'brain'}
    #                              (facilitate is OOV → contributes nothing)
    #   anchor_key for both = frozenset({'human', 'brain'})
    #   ⟹  second term is a duplicate; alias it to the first (winner).
    #
    # Implementation
    # --------------
    # unique_anchor_map : frozenset → first OOV term seen for that key (winner)
    # alias_map         : duplicate OOV term → its winner
    # deduplicated_terms: ordered list of winners (no duplicates)
    #
    # After the main loop the alias resolution block copies the winner's
    # result list to every duplicate, so the returned dict contains a key
    # for every originally submitted OOV term.
    # ═════════════════════════════════════════════════════════════════════
    unique_anchor_map : Dict[frozenset, str] = {}   # anchor_key  → winner term
    alias_map         : Dict[str, str]       = {}   # duplicate   → winner term
    deduplicated_terms: List[str]            = []

    for oov_term in oov_terms:
        # anchor_key: only the sub-tokens that actually exist in the vocab
        anchor_key = frozenset(
            t for t in oov_term.split() if t in vocab_fp_index
        )

        if anchor_key not in unique_anchor_map:
            # First time we see this anchor set — this term is the winner
            unique_anchor_map[anchor_key] = oov_term
            deduplicated_terms.append(oov_term)
            logger.debug(
                f"  [OOV DEDUP] '{oov_term}' → new anchor key {set(anchor_key)}"
            )
        else:
            # Same anchor set seen before → alias this term to the winner
            winner = unique_anchor_map[anchor_key]
            alias_map[oov_term] = winner
            logger.debug(
                f"  [OOV DEDUP SKIP] '{oov_term}' → same anchor as "
                f"'{winner}' {set(anchor_key)}, will reuse results"
            )

    logger.debug(
        f"  [OOV DEDUP] {len(oov_terms)} terms → "
        f"{len(deduplicated_terms)} unique anchors "
        f"({len(alias_map)} duplicates suppressed)"
    )

    # ─────────────────────────────────────────────────────────────────────
    # Main expansion loop — only over deduplicated (winner) terms
    # ─────────────────────────────────────────────────────────────────────
    for oov_term in deduplicated_terms:
        logger.debug(f"  [OOV] Processing term: '{oov_term}'")

        # ── STEP 1: Build anchor vector ───────────────────────────────────
        #
        # Sum the real fingerprints of every sub-token that exists in the
        # vocabulary.  This gives the anchor genuine semantic coordinates.
        tokens         = oov_term.split()
        anchor_vec     = np.zeros(dim, dtype=np.float32)
        matched_tokens = 0

        for token in tokens:
            if token in vocab_fp_index:
                anchor_vec     += vocab_fp_index[token]
                matched_tokens += 1

        # ═════════════════════════════════════════════════════════════════
        # FIX #4 — Raise the minimum anchor quality bar for multi-token OOVs
        # ═════════════════════════════════════════════════════════════════
        #
        # Original rule
        # -------------
        #   min_required = max(1, len(tokens) // 2)
        #
        # Problem
        # -------
        # For a 2-token OOV (e.g. 'brain facilitate'):
        #   len(tokens) // 2 = 1  →  max(1, 1) = 1
        # So 'brain' alone (1/2 tokens) was enough to pass Case 2 and build
        # an anchor.  That anchor was IDENTICAL to the single-word fingerprint
        # of 'brain', causing cosine similarity to return sim=1.000 for
        # 'neuroplasticity', 'pain', 'pain management', etc. — i.e. whatever
        # happens to be closest to 'brain' with no contribution from
        # 'facilitate' at all.  This is the sparse-collision noise flagged as
        # Bug #4 in the debug report.
        #
        # Fix
        # ---
        # Single-token OOV : min_required = 1  (must match that one token)
        # Multi-token OOV  : min_required = max(2, len(tokens) // 2)
        #
        # Effect on observed cases
        # ------------------------
        #   'brain facilitate'   (2 tokens, 1 matched) : 1 < 2 → SKIP (was EXPAND)
        #   'human brain'        (2 tokens, 2 matched) : 2 ≥ 2 → EXPAND (unchanged)
        #   'recovery function'  (2 tokens, 0 matched) : char n-gram fallback
        #   'some four word oov' (4 tokens, 2 matched) : 2 ≥ 2 → EXPAND (unchanged)
        #   'some four word oov' (4 tokens, 1 matched) : 1 < 2 → SKIP (tightened)
        #
        # MIN_MULTI_TOKEN_ANCHOR is defined as a named constant for clarity
        # and to make future tuning a single-line change.
        # ═════════════════════════════════════════════════════════════════
        MIN_MULTI_TOKEN_ANCHOR = 2  # minimum in-vocab sub-tokens for multi-word OOV

        if len(tokens) == 1:
            # Single-word OOV: the one token must be in vocab to form an anchor;
            # if it were, the phrase would have been IV in the first place —
            # so in practice this almost always falls through to Case 1.
            min_required = 1
        else:
            # Multi-word OOV: require at least MIN_MULTI_TOKEN_ANCHOR matched
            # tokens, or half of all tokens if that is larger (e.g. a 6-token
            # phrase should still need at least 3, not just 2).
            min_required = max(MIN_MULTI_TOKEN_ANCHOR, len(tokens) // 2)

        logger.debug(
            f"  [OOV ANCHOR CHECK] '{oov_term}' → "
            f"{matched_tokens}/{len(tokens)} sub-tokens in vocab "
            f"(need ≥{min_required})"
        )

        if matched_tokens == 0:
            # ── Case 1: no sub-tokens at all → character n-gram fallback ──
            #
            # The pseudo-fingerprint carries no semantic information but
            # may still surface morphologically similar vocab phrases.
            anchor_vec = _pseudo_fingerprint(oov_term, grid_size=grid_size)
            logger.debug(
                f"  [OOV PSEUDO-FP] '{oov_term}' → "
                f"no sub-tokens in vocab, using char n-gram fallback"
            )

        elif matched_tokens < min_required:
            # ── Case 2: too few sub-tokens → anchor is unreliable, skip ──
            #
            # With Fix #4 this now catches 2-token OOVs where only 1 token
            # matched — previously these slipped through and caused
            # single-word anchor collisions (sim=1.000 noise).
            logger.debug(
                f"  [OOV WEAK ANCHOR SKIP] '{oov_term}' → "
                f"only {matched_tokens}/{len(tokens)} sub-tokens in vocab "
                f"(need ≥{min_required}), skipping"
            )
            continue  # move to next OOV term

        else:
            # ── Case 3: enough sub-tokens → use summed real fingerprints ──
            logger.debug(
                f"  [OOV ANCHOR] '{oov_term}' → "
                f"built from {matched_tokens}/{len(tokens)} in-vocab sub-tokens"
            )

        # ── STEP 2: Degenerate-anchor guard ──────────────────────────────
        #
        # A zero-norm vector means every contributing fingerprint was also
        # zero (extremely unlikely but possible with sparse SDRs).
        # Cosine similarity would be undefined — skip rather than divide by 0.
        anchor_norm = np.linalg.norm(anchor_vec)
        if anchor_norm < 1e-9:
            logger.debug(f"  [OOV SKIP] '{oov_term}' → zero-norm anchor vector")
            continue

        # ── STEP 3: Batch cosine similarity against entire vocab ──────────
        #
        # dot(vocab_matrix, anchor) / (vocab_norms * anchor_norm)
        # Shapes: (V,D) @ (D,) → (V,) / ((V,1).squeeze() * scalar) → (V,)
        #
        # vocab_norms were precomputed before the loop (O(V) once).
        similarities = (vocab_matrix @ anchor_vec) / (
            vocab_norms.squeeze() * anchor_norm
        )

        # Sort indices descending; iterate until threshold or top_k is reached
        top_indices = np.argsort(similarities)[::-1]

        matches: List[Tuple[str, float]] = []
        for idx in top_indices:
            phrase = vocab_phrases[idx]
            score  = float(similarities[idx])

            # ── Early exit: sorted descending, nothing below can qualify ──
            if score < min_similarity:
                logger.debug(
                    f"  [OOV THRESH] '{oov_term}': score dropped to "
                    f"{score:.3f} < {min_similarity} — stopping"
                )
                break

            # ── Redundancy filter (applied BEFORE top_k counter) ─────────
            #
            # These checks must come before appending to `matches` so that
            # filtered phrases do NOT consume one of the top_k result slots.

            # Skip the OOV term itself (exact match is trivially unhelpful)
            if phrase == oov_term:
                continue

            # Skip constituent tokens — they were already used to build the
            # anchor and are likely already present in matched_phrases
            if phrase in tokens:
                continue

            # Mono→mono guard: expanding a single-word OOV to another
            # single-word phrase is almost always noisy (high-frequency
            # generic terms dominate cosine similarity in SDR space).
            if len(tokens) == 1 and len(phrase.split()) == 1:
                logger.debug(
                    f"  [OOV SKIP MONO→MONO] '{oov_term}' → '{phrase}' "
                    f"(single-word to single-word expansion suppressed)"
                )
                continue

            # ── Generic-term score penalty ────────────────────────────────
            #
            # High-frequency words like "system" or "process" tend to have
            # dense, generic fingerprints that score well against almost
            # anything.  Halving their score reduces their ranking priority
            # without removing them entirely.
            if penalize_generic and phrase in GENERIC_TERMS:
                original_score = score
                score *= 0.5
                logger.debug(
                    f"  [OOV GENERIC] '{phrase}' penalised "
                    f"{original_score:.3f} → {score:.3f}"
                )
                # Re-check threshold after penalty
                if score < min_similarity:
                    logger.debug(
                        f"  [OOV GENERIC THRESH] '{phrase}' dropped below "
                        f"threshold after penalty — skipping"
                    )
                    continue

            # ── Accept this match ─────────────────────────────────────────
            matches.append((phrase, score))
            logger.debug(
                f"  [OOV MATCH] '{oov_term}' → '{phrase}' (sim={score:.3f})"
            )

            # Stop once we have enough matches for this OOV term
            if len(matches) >= top_k_per_term:
                break

        # ── STEP 4: Record results ────────────────────────────────────────
        if matches:
            expansions[oov_term] = matches
            logger.debug(
                f"  [OOV RESULT] '{oov_term}' → "
                + ", ".join(f"'{p}'({s:.3f})" for p, s in matches)
            )
        else:
            logger.debug(
                f"  [OOV RESULT] '{oov_term}' → no matches "
                f"above threshold {min_similarity}"
            )

    # ═════════════════════════════════════════════════════════════════════
    # FIX #3 (cont.) — Propagate winner results to aliased duplicate terms
    # ═════════════════════════════════════════════════════════════════════
    #
    # Every caller that submitted an OOV term expects a key in the returned
    # dict (or absence means no match — both are valid).  For aliased
    # duplicates we copy the winner's list reference so the returned dict
    # is complete without re-running any similarity computation.
    #
    # Note: if the winner itself had no matches (empty / skipped), the
    # alias is simply not added — consistent with the no-match convention.
    # ═════════════════════════════════════════════════════════════════════
    for duplicate_term, winner_term in alias_map.items():
        if winner_term in expansions:
            expansions[duplicate_term] = expansions[winner_term]
            logger.debug(
                f"  [OOV ALIAS] '{duplicate_term}' ← reusing "
                f"'{winner_term}' results: "
                + ", ".join(f"'{p}'({s:.3f})" for p, s in expansions[winner_term])
            )
        else:
            # Winner had no matches above threshold — alias also gets nothing
            logger.debug(
                f"  [OOV ALIAS] '{duplicate_term}' → winner '{winner_term}' "
                f"had no results; alias also empty"
            )

    logger.debug(
        f"OOV expansion complete: {len(expansions)}/{len(oov_terms)} "
        f"terms expanded "
        f"({len(alias_map)} resolved via alias, "
        f"{len(deduplicated_terms)} unique anchors processed)"
    )
    return expansions


def _pseudo_fingerprint(term: str, grid_size: int = 128) -> np.ndarray:
    """
    Generate a deterministic pseudo-fingerprint for an OOV term.

    Uses character n-gram hashing (unigrams, bigrams, trigrams) to
    produce a dense float32 vector of dimension ``grid_size²``.  Two
    independent MD5-derived hash functions are applied per n-gram to
    reduce collision clustering.

    This is **not** a semantic fingerprint — it carries no distributional
    meaning.  Its sole purpose is to provide a rough proximity signal for
    :func:`expand_oov_query_terms` when no trained fingerprint exists.

    Parameters
    ----------
    term : str
        The OOV term string to fingerprint.
    grid_size : int, optional
        SDR grid side length (default: ``128``).  The output vector has
        ``grid_size * grid_size`` dimensions.

    Returns
    -------
    np.ndarray
        Shape ``(grid_size²,)``, dtype ``float32``.  Values are
        non-negative counts of hash collisions per bucket.
    """
    dim = grid_size * grid_size
    vec = np.zeros(dim, dtype=np.float32)

    # Unigrams, bigrams, trigrams — wider n-gram range improves coverage
    # for multi-syllable technical terms
    for n in range(1, 4):
        ngrams = [term[i : i + n] for i in range(len(term) - n + 1)]
        for ng in ngrams:
            # Two independent hash seeds per n-gram for better distribution
            h1 = int(hashlib.md5(f"A{ng}".encode()).hexdigest(), 16) % dim
            h2 = int(hashlib.md5(f"B{ng}".encode()).hexdigest(), 16) % dim
            vec[h1] += 1.0
            vec[h2] += 1.0

    logger.debug(
        f"  [PSEUDO-FP] '{term}' → {int(np.count_nonzero(vec))} "
        f"non-zero buckets / {dim} dims"
    )
    return vec

def merge_expanded_phrases(
    matched_phrases  : List[str],
    oov_expansions   : Dict[str, List[Tuple[str, float]]],
    idf_weights      : Optional[Dict[str, float]] = None,
    expansion_weight : float = 0.6,
) -> Dict[str, float]:
    """
    Merge direct vocabulary matches with OOV expansion results.

    Produces a single ``{phrase: weight}`` dict that
    :func:`construct_query_fingerprint` can consume directly.

    Weight assignment rules
    -----------------------
    - **Direct matches** (phrases already in vocabulary): weight is the
      IDF score from ``idf_weights`` if provided, otherwise ``1.0``.

    - **Expanded matches** (OOV proxies): weight is
      ``idf_score × similarity² × expansion_weight``.

      Similarity is **squared** (vs. the previous linear ``sim × discount``)
      so that weak neighbors are penalised quadratically rather than
      linearly.  Concretely:

      +-----------+------------+------------------+
      | sim       | old weight | new weight (sim²)|
      +===========+============+==================+
      | 0.90      | 0.540      | 0.486            |
      | 0.70      | 0.420      | 0.294            |
      | 0.55      | 0.330      | 0.182  ← −45 %   |
      | 0.40      | 0.240      | 0.096  ← −60 %   |
      +-----------+------------+------------------+

      This directly addresses the observed semantic-drift bug where
      ``'language'`` (sim=0.541) was injected as a high-weight proxy for
      ``'human brain'``, incorrectly boosting Context 17 (Semantics) over
      Context 12 (Bilingualism).

    - When the same phrase appears as both a direct match and an
      expansion target, the **maximum** of the two weights is kept.

    Parameters
    ----------
    matched_phrases : List[str]
        Phrases that passed the vocabulary filter in
        :func:`extract_query_phrases`.
    oov_expansions : Dict[str, List[Tuple[str, float]]]
        Output of :func:`expand_oov_query_terms`.  Each key is an OOV
        anchor term; each value is a list of ``(vocab_phrase, similarity)``
        pairs sorted by descending similarity.
    idf_weights : Dict[str, float] or None, optional
        ``{phrase: idf_score}`` mapping.  When ``None``, all base
        weights default to ``1.0``.
    expansion_weight : float, optional
        Discount factor applied to expanded phrase weights (default:
        ``0.6``).  Combined with the squared similarity this gives the
        full formula::

            w = IDF(t) × sim(anchor, t)² × expansion_weight

        Set to ``1.0`` to remove the flat discount and rely solely on
        the quadratic similarity penalty.

    Returns
    -------
    Dict[str, float]
        ``{phrase: weight}`` mapping ready for fingerprint construction.

    Notes
    -----
    **Why sim² instead of sim?**

    With the old linear formula a neighbour at sim=0.54 retained
    ``0.54 × 0.6 = 32 %`` of its IDF weight — enough to meaningfully
    shift the query fingerprint toward unrelated contexts.  Squaring
    reduces that retention to ``0.54² × 0.6 = 17 %``, and the effect
    compounds for every additional weak neighbour that would otherwise
    accumulate.  High-confidence neighbours (sim ≥ 0.85) lose less than
    10 % of their effective weight, so true semantic proxies are largely
    unaffected.
    """
    phrase_weights: Dict[str, float] = {}

    logger.debug(
        f"Merging {len(matched_phrases)} direct matches + "
        f"{len(oov_expansions)} OOV expansion groups "
        f"(expansion_weight={expansion_weight}, sim_power=2)"
    )

    # ------------------------------------------------------------------
    # Direct matches — full IDF weight (or 1.0 if no IDF table provided)
    # ------------------------------------------------------------------
    for phrase in matched_phrases:
        base_weight = idf_weights.get(phrase, 1.0) if idf_weights else 1.0
        phrase_weights[phrase] = base_weight
        logger.debug(f"  [DIRECT] '{phrase}' → weight={base_weight:.4f}")

    # ------------------------------------------------------------------
    # Expanded matches — quadratically discounted by sim² × expansion_weight
    #
    # FIX #4: changed  idf × sim × discount
    #                → idf × sim² × discount
    #
    # Rationale: low-similarity OOV neighbours (sim ≈ 0.54–0.63) were
    # retaining enough weight under the linear formula to introduce
    # spurious context signal (e.g. 'language' boosting Context 17).
    # Squaring sim collapses their contribution without harming
    # high-confidence neighbours (sim ≥ 0.85).
    # ------------------------------------------------------------------
    for oov_term, matches in oov_expansions.items():
        for vocab_phrase, sim_score in matches:
            base_weight   = idf_weights.get(vocab_phrase, 1.0) if idf_weights else 1.0
            sim_sq        = sim_score * sim_score          # sim²
            merged_weight = base_weight * sim_sq * expansion_weight

            existing = phrase_weights.get(vocab_phrase, 0.0)
            if merged_weight > existing:
                phrase_weights[vocab_phrase] = merged_weight
                logger.debug(
                    f"  [EXPAND] '{oov_term}' → '{vocab_phrase}' "
                    f"weight={merged_weight:.4f} "
                    f"(idf={base_weight:.3f} × sim²={sim_sq:.3f} × "
                    f"discount={expansion_weight})"
                )
            else:
                logger.debug(
                    f"  [EXPAND SKIP] '{vocab_phrase}' kept existing "
                    f"weight={existing:.4f} > merged={merged_weight:.4f}"
                )

    logger.info(
        f"Phrase merge: {len(matched_phrases)} direct + "
        f"{sum(len(v) for v in oov_expansions.values())} expanded "
        f"→ {len(phrase_weights)} unique weighted phrases"
    )
    return phrase_weights



# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _infer_vector_size(phrase_fingerprints: dict) -> int:
    """
    Return the flat fingerprint dimension from the first dictionary entry.

    Handles both dense ``np.ndarray`` and sparse ``csr_matrix`` values so
    the function is robust to whichever storage format Step 4 produced.

    Parameters
    ----------
    phrase_fingerprints : dict
        Non-empty mapping of ``{phrase: vector}`` as returned by
        ``lib.load_phrase_fingerprints_sparse``.

    Returns
    -------
    int
        Number of elements in the flat (1-D) fingerprint vector,
        i.e. ``grid_size * grid_size``.

    Raises
    ------
    StopIteration
        If ``phrase_fingerprints`` is empty (caller's responsibility to
        guard against this).
    """
    first = next(iter(phrase_fingerprints.values()))

    if hasattr(first, "toarray"):
        # Sparse matrix (csr_matrix, lil_matrix, etc.)
        size = first.toarray().ravel().shape[0]
    else:
        # Dense array or list
        size = np.asarray(first).ravel().shape[0]

    logger.debug(f"Inferred fingerprint vector size: {size} ({int(size**0.5)}²)")
    return size


# ─────────────────────────────────────────────────────────────────────────────
# Phrase extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_query_phrases(
    query           : str,
    phrase_vocab    : Set[str],
    use_spacy       : bool = True,
    remove_verbs    : bool = False,
    filter_generic  : bool = True,
    min_word_length : int  = 3,
) -> List[str]:
    """
    Extract vocabulary-matched phrases from a raw query string.

    Applies the **identical** three-stage pipeline used in Steps 1 and 5
    to guarantee that query phrases map to the same vocabulary entries
    built during indexing.

    Stage 1 — Extraction + normalisation
        ``extract_raw_phrases_spacy`` (or the NLTK fallback) detects noun
        chunks, named entities, and compound nouns.  Each candidate is
        piped through ``lib.normalize_phrase`` (lowercase → stop-word
        removal → optional verb removal → lemmatisation → POS validation).

    Stage 2 — Sub-phrase expansion
        ``lib.expand_phrases`` generates all meaningful sub-phrases from
        each normalised phrase (bigrams and trigrams from longer phrases,
        individual tokens from shorter ones), mirroring the expansion
        logic in ``process_corpus_with_expansion``.

        **This stage is critical for recall.**  Without it a query
        containing "deep neural network" would not activate the
        fingerprints for "neural network" or "neural", even though those
        sub-phrases exist in the vocabulary built during Step 1.

    Stage 3 — Vocabulary filter
        Only phrases present as keys in ``phrase_vocab`` are retained,
        exactly mirroring the ``min_freq`` filter applied in Step 1.

    Parameters
    ----------
    query : str
        Raw query string entered by the user.
    phrase_vocab : Set[str]
        Complete set of normalised phrase strings from the loaded phrase
        fingerprint dictionary (``set(phrase_fingerprints.keys())``).
    use_spacy : bool, optional
        When ``True`` (default), attempt spaCy noun-chunk extraction.
        **Must match the flag used during Step 1.**
    remove_verbs : bool, optional
        Strip verb tokens before lemmatisation (default: ``True``).
        **Must match the flag used during Step 1.**
    filter_generic : bool, optional
        Remove generic single-word tokens during expansion (default:
        ``True``).  **Must match the flag used during Step 1.**
    min_word_length : int, optional
        Minimum character length for single-word tokens kept after
        expansion (default: ``3``).  **Must match the flag used during
        Step 1.**

    Returns
    -------
    List[str]
        Vocabulary-matched, normalised phrase strings.  Duplicates are
        preserved so that term-frequency weighting can be applied by the
        caller.  Returns an empty list when no phrases match the
        vocabulary.

    Notes
    -----
    - If spaCy is requested but unavailable a WARNING is emitted and the
      NLTK fallback is used automatically.
    - The function is stateless and safe to call from parallel workers.

    Examples
    --------
    Suppose the vocabulary contains ``{"neural network", "deep neural",
    "neural", "network"}``.

    >>> phrases = extract_query_phrases(
    ...     "The model uses a deep neural network.",
    ...     phrase_vocab=vocab,
    ...     use_spacy=True,
    ...     remove_verbs=True,
    ... )
    >>> sorted(set(phrases))
    ['deep neural', 'neural', 'neural network', 'network']
    """
    logger.debug(
        f"extract_query_phrases: query={query!r} "
        f"use_spacy={use_spacy} remove_verbs={remove_verbs} "
        f"filter_generic={filter_generic} min_word_length={min_word_length} "
        f"vocab_size={len(phrase_vocab)}"
    )

    # ── Stage 1: extraction + normalisation ──────────────────────────────────
    if use_spacy and SPACY_AVAILABLE:
        logger.debug("Stage 1: using spaCy extractor")
        doc = nlp(query)
        raw_phrases = extract_raw_phrases_spacy(doc)
    else:
        if use_spacy and not SPACY_AVAILABLE:
            logger.warning(
                "spaCy requested but unavailable — using NLTK fallback. "
                "Verify this matches the setting used in Step 1."
            )
        else:
            logger.debug("Stage 1: using NLTK fallback extractor")
        raw_phrases = extract_raw_phrases_fallback(query, max_ngram=4)

    logger.debug(f"  [EXTRACT] {len(raw_phrases)} raw phrases: {raw_phrases}")

    candidates: List[str] = []
    for phrase in raw_phrases:
        norm = normalize_phrase(phrase, remove_verbs=remove_verbs)
        if norm:
            candidates.append(norm)
            logger.debug(f"  [NORM OK] '{phrase}' → '{norm}'")
        else:
            logger.debug(f"  [NORM DROP] '{phrase}' → empty after normalisation")

    if not candidates:
        logger.debug(f"No candidates after normalisation for query: {query!r}")
        return []

    logger.debug(f"  Stage 1 complete: {len(candidates)} normalised candidates")

    # ── Stage 2: sub-phrase expansion ────────────────────────────────────────
    logger.debug("Stage 2: expanding candidates into sub-phrases")
    expanded: List[str] = expand_phrases(
        candidates,
        context_text    = query,   # raw query text used for context validation
        filter_generic  = filter_generic,
        min_word_length = min_word_length,
    )
    logger.debug(
        f"  Stage 2 complete: {len(candidates)} candidates → "
        f"{len(expanded)} expanded phrases"
    )
    if expanded:
        logger.debug(f"  [EXPANDED] {expanded}")

    # ── Stage 3: vocabulary filter ────────────────────────────────────────────
    logger.debug("Stage 3: filtering against phrase vocabulary")
    matched: List[str] = []
    oov:     List[str] = []
    for p in expanded:
        if p in phrase_vocab:
            matched.append(p)
        else:
            oov.append(p)

    if oov:
        logger.debug(f"  [OOV] {len(oov)} phrases not in vocab: {oov}")
    if matched:
        logger.debug(f"  [MATCHED] {matched}")

    logger.info(
        f"Query phrase extraction: {len(raw_phrases)} raw → "
        f"{len(candidates)} normalised → "
        f"{len(expanded)} expanded → "
        f"{len(matched)} vocab hits"
    )

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint construction
# ─────────────────────────────────────────────────────────────────────────────


def construct_query_fingerprint(
    query_phrases       : List[str],
    phrase_fingerprints : Dict[str, csr_matrix],
    weighting           : str                        = "idf",
    idf_weights         : Optional[Dict[str, float]] = None,
    normalization       : str                        = "l2",
) -> Tuple[Optional[csr_matrix], Dict]:
    """
    Aggregate per-phrase fingerprints into a single query fingerprint vector.

    Each phrase in ``query_phrases`` that is found in
    ``phrase_fingerprints`` is weighted and accumulated into a dense
    float32 array before being converted to a sparse row vector.  The
    aggregated vector is optionally normalised before return.

    The function is intentionally conservative: it returns
    ``(None, metadata)`` in every failure mode rather than raising,
    allowing the caller to decide whether to skip or abort.

    Weighting modes
    ---------------
    ``"uniform"``
        Every unique phrase contributes weight ``1.0``, regardless of
        frequency.
    ``"frequency"``
        Weight equals the term frequency (TF) of the phrase in
        ``query_phrases`` (promotes repeated terms).
    ``"idf"``
        Weight is TF × IDF score from ``idf_weights`` (falls back to
        TF × 1.0 for unknown phrases).  Requires ``idf_weights``; degrades
        to TF-only if ``idf_weights`` is ``None``.

    Implementation notes
    --------------------
    - Accumulation is performed on a dense ``np.zeros`` float32 array for
      O(1) additions, bypassing the ``lil_matrix`` bottleneck.  Converted
      to ``csr_matrix`` only at the end.
    - Uses ``collections.Counter`` to group unique phrases, fixing an O(N²)
      bottleneck from ``list.count()`` and a logical bug where iterating
      over duplicate phrases effectively squared TF contributions.

    Parameters
    ----------
    query_phrases : List[str]
        Ordered list of normalised phrase strings from
        :func:`extract_query_phrases`.  An empty list causes an immediate
        ``(None, {"error": "no_phrases"})`` return.
    phrase_fingerprints : Dict[str, csr_matrix]
        Mapping of ``{phrase: fingerprint_vector}`` from Step 4.  Each
        value must be a ``(1, grid_size²)`` sparse row vector.
    weighting : str, optional
        Weighting scheme — ``"uniform"``, ``"frequency"``, or ``"idf"``
        (default: ``"uniform"``).
    idf_weights : Dict[str, float] or None, optional
        ``{phrase: idf_score}`` mapping.  Only consulted when
        ``weighting="idf"``.
        normalization : str, optional
        Normalisation applied to the aggregated vector before return.
        Supported values: ``"l2"`` (default), ``"l1"``, ``"binary"``,
        ``"none"``.

    Returns
    -------
    fingerprint : csr_matrix or None
        Sparse ``(1, grid_size²)`` query vector.  ``None`` on failure.
    metadata : Dict
        Always present; contains an ``"error"`` key on failure or
        detailed statistics on success:

        ``num_phrases``
            Total phrase tokens in ``query_phrases`` (with duplicates).
        ``num_matched``
            Unique phrases found in ``phrase_fingerprints``.
        ``num_missing``
            Unique phrases absent from ``phrase_fingerprints``.
        ``missing_phrases``
            List of OOV phrase strings.
        ``phrase_weights``
            ``{phrase: weight}`` actually applied during accumulation.
        ``active_bits_pre_norm``
            Non-zero count before normalisation.
        ``active_bits``
            Non-zero count after normalisation.
        ``sparsity``
            ``active_bits / grid_size²``.
        ``weighting``
            The weighting scheme used.
        ``normalization``
            The normalisation method used.
    """
    if not query_phrases:
        logger.warning("No query phrases provided to construct_query_fingerprint")
        return None, {"error": "no_phrases"}

    if not phrase_fingerprints:
        logger.error("Phrase fingerprints dictionary is empty")
        return None, {"error": "empty_phrase_vocabulary"}

    grid_size_sq = _infer_vector_size(phrase_fingerprints)

    logger.debug(
        f"construct_query_fingerprint: {len(query_phrases)} phrase tokens, "
        f"weighting='{weighting}', normalization='{normalization}', "
        f"grid_size_sq={grid_size_sq}"
    )

    # Dense accumulator — O(1) per addition, converted to CSR at the end
    acc = np.zeros(grid_size_sq, dtype=np.float32)

    phrase_weights_used: Dict[str, float] = {}
    missing_phrases:     List[str]        = []

    # Counter collapses duplicates and gives TF in one pass — O(N)
    phrase_counts = Counter(query_phrases)
    logger.debug(f"  [TF] phrase counts: {dict(phrase_counts)}")

    for phrase, tf in phrase_counts.items():
        if phrase not in phrase_fingerprints:
            logger.debug(f"  [MISSING] '{phrase}' not in fingerprint vocab")
            missing_phrases.append(phrase)
            continue

        phrase_fp = phrase_fingerprints[phrase]

        # Compute scalar weight for this phrase
        if weighting == "idf" and idf_weights:
            idf_score = float(idf_weights.get(phrase, 1.0))
            weight    = idf_score * float(tf)
            logger.debug(
                f"  [WEIGHT IDF] '{phrase}' tf={tf} × idf={idf_score:.4f} "
                f"→ weight={weight:.4f}"
            )
        elif weighting == "frequency":
            weight = float(tf)
            logger.debug(f"  [WEIGHT TF] '{phrase}' tf={tf} → weight={weight:.4f}")
        else:  # "uniform"
            weight = 1.0
            logger.debug(f"  [WEIGHT UNIFORM] '{phrase}' → weight=1.0")

        # Flatten fingerprint to 1-D and accumulate
        if hasattr(phrase_fp, "toarray"):
            fp_array = phrase_fp.toarray().ravel()
        else:
            fp_array = np.asarray(phrase_fp).ravel()

        pre_nnz = int(np.count_nonzero(acc))
        acc    += weight * fp_array
        post_nnz = int(np.count_nonzero(acc))

        logger.debug(
            f"  [ACCUM] '{phrase}' — fp_nnz={int(np.count_nonzero(fp_array))}, "
            f"acc nnz: {pre_nnz} → {post_nnz}"
        )
        phrase_weights_used[phrase] = weight

    # Guard: accumulator still all-zero after processing all phrases
    if not np.any(acc):
        if missing_phrases:
            logger.error(
                f"All {len(missing_phrases)} unique query phrase(s) are "
                f"out-of-vocabulary: {missing_phrases}"
            )
        else:
            logger.error("Query fingerprint is empty after aggregation")
        return None, {
            "error":           "empty_fingerprint",
            "missing_phrases": missing_phrases,
        }

    logger.debug(
        f"  [PRE-NORM] accumulator nnz={int(np.count_nonzero(acc))}, "
        f"min={acc.min():.4f}, max={acc.max():.4f}"
    )

    # Convert dense accumulator → sparse row vector
    acc_csr      = csr_matrix(acc.reshape(1, -1))
    pre_norm_nnz = acc_csr.nnz

    # Apply normalisation (preserves float weights — no binarisation)
    if normalization and normalization != "none":
        acc_csr = normalize_fingerprint(acc_csr, method=normalization)
        logger.debug(
            f"  [NORM] '{normalization}' applied — "
            f"nnz before={pre_norm_nnz}, after={acc_csr.nnz}"
        )
    else:
        logger.debug("  [NORM] skipped (normalization='none')")

    post_norm_nnz = acc_csr.nnz

    metadata = {
        "num_phrases":          len(query_phrases),
        "num_matched":          len(phrase_weights_used),
        "num_missing":          len(missing_phrases),
        "missing_phrases":      missing_phrases,
        "phrase_weights":       phrase_weights_used,
        "active_bits_pre_norm": pre_norm_nnz,
        "active_bits":          post_norm_nnz,
        "sparsity":             post_norm_nnz / grid_size_sq,
        "weighting":            weighting,
        "normalization":        normalization,
    }

    logger.debug(
        f"  [RESULT] matched={len(phrase_weights_used)}, "
        f"missing={len(missing_phrases)}, "
        f"active_bits={post_norm_nnz}, "
        f"sparsity={post_norm_nnz/grid_size_sq:.4f}"
    )
    logger.success(
        f"Query fingerprint: {post_norm_nnz} active elements from "
        f"{len(phrase_weights_used)} phrases (weighting='{weighting}')"
    )

    return acc_csr, metadata

# ─────────────────────────────────────────────────────────────────────────────
# Spreading
# ─────────────────────────────────────────────────────────────────────────────

def apply_spreading(
    fingerprint     : csr_matrix,
    grid_size       : int,
    radius          : int   = 1,
    decay           : float = 0.5,
    normalize_after : bool  = True,
) -> Tuple[csr_matrix, Dict]:
    r"""
    Apply Z-order neighbourhood spreading to a query fingerprint.

    Spreading propagates activation from each active bit outward to its
    Z-order spatial neighbours, attenuated by a decay factor per unit of
    Chebyshev distance.  This soft-expands the query's representational
    footprint so that documents sharing *nearby* rather than *identical*
    grid coordinates can still contribute to similarity.

    Parameters
    ----------
    fingerprint : csr_matrix
        Sparse ``(1, grid_size²)`` query fingerprint.
    grid_size : int
        Side length of the square Z-order grid.
    radius : int, optional
        Maximum Chebyshev distance to which activation is spread.
        ``0`` is a no-op (returns the input unchanged).  Default: ``1``.
    decay : float, optional
        Multiplicative attenuation per unit Chebyshev distance.  Must be
        in ``(0.0, 1.0]``.  Default: ``0.5``.
    normalize_after : bool, optional
        When ``True`` (default), L2-normalise the spread fingerprint
        before return to keep it on the same unit sphere as document
        fingerprints.

    Returns
    -------
    result : csr_matrix
        Spread (and optionally normalised) ``(1, grid_size²)`` fingerprint.
    metadata : Dict
        ``spreading_applied`` — ``False`` when ``radius=0``.
        ``radius``, ``decay``, ``active_bits_before``,
        ``active_bits_after``, ``bits_added``.

    Notes
    -----
    - The sparsity guard uses ``fingerprint.shape[1]`` (number of cells)
      rather than ``fingerprint.shape[0]`` (always 1) as the denominator.
      The threshold is ``0.02`` so that short queries on a 16×16 grid are
      not unconditionally suppressed.
    - Neighbour coordinates are obtained via ``lib.get_zorder_neighbors``,
      which respects grid boundaries.
    - The intermediate computation uses a dense ``(grid_size, grid_size)``
      NumPy array; for grids larger than 64×64 consider a sparse
      spreading implementation.
    """
    logger.debug(
        f"apply_spreading: radius={radius}, decay={decay}, "
        f"normalize_after={normalize_after}, input_nnz={fingerprint.nnz}"
    )

    if radius == 0:
        logger.debug("  [SPREAD SKIP] radius=0 — returning fingerprint unchanged")
        return fingerprint, {"spreading_applied": False}

    n_cells  = fingerprint.shape[1]
    sparsity = fingerprint.nnz / n_cells

    logger.debug(
        f"  [SPREAD GUARD] sparsity={sparsity:.4f} "
        f"({fingerprint.nnz} bits / {n_cells} cells), threshold={SPARCITY_GAURD}"
    )

    if sparsity < SPARCITY_GAURD:
        logger.warning(
            f"Fingerprint very sparse ({sparsity:.4f}, {fingerprint.nnz} bits "
            f"/ {n_cells} cells) — spreading skipped to avoid score collapse."
        )
        return fingerprint, {
            "spreading_applied": False,
            "reason":            "sparsity_too_low",
            "sparsity":          sparsity,
        }

    original_nnz  = fingerprint.nnz
    dense_fp      = fingerprint.toarray().reshape(grid_size, grid_size)
    spread_fp     = dense_fp.copy()
    active_coords = np.argwhere(dense_fp > 0)

    logger.debug(
        f"  [SPREAD] {len(active_coords)} active coords to propagate, "
        f"grid={grid_size}×{grid_size}"
    )

    for y, x in active_coords:
        value     = dense_fp[y, x]
        neighbors = get_zorder_neighbors(x, y, grid_size, radius)
        for nx, ny in neighbors:
            dist = max(abs(nx - x), abs(ny - y))
            contribution = value * (decay ** dist)
            spread_fp[ny, nx] += contribution
            # logger.debug(
            #     f"    [SPREAD PROP] ({x},{y})→({nx},{ny}) "
            #     f"dist={dist} val={value:.4f} contrib={contribution:.4f}"
            # )

    result = csr_matrix(spread_fp.reshape(1, -1))
    pre_norm_nnz = result.nnz

    logger.debug(
        f"  [SPREAD POST] nnz before norm={pre_norm_nnz}, "
        f"bits_added={pre_norm_nnz - original_nnz}"
    )

    if normalize_after:
        result = normalize_fingerprint(result, method="l2")
        logger.debug(
            f"  [SPREAD NORM] l2 applied — nnz: {pre_norm_nnz} → {result.nnz}"
        )
    else:
        logger.debug("  [SPREAD NORM] skipped (normalize_after=False)")

    metadata = {
        "spreading_applied":  True,
        "radius":             radius,
        "decay":              decay,
        "active_bits_before": original_nnz,
        "active_bits_after":  result.nnz,
        "bits_added":         result.nnz - original_nnz,
    }

    logger.info(
        f"Spreading: {original_nnz} → {result.nnz} active bits "
        f"(+{result.nnz - original_nnz}) | radius={radius} decay={decay:.2f}"
    )
    return result, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_documents(
    query_fp        : csr_matrix,
    doc_fingerprints: Dict[str, csr_matrix],
    top_k           : int   = 10,
    min_similarity  : float = 0.0,
    use_batch       : bool  = True,
    **kwargs,
) -> Tuple[List[Tuple[str, float]], Dict]:
    """
    Rank documents by asymmetric cosine similarity against the query fingerprint.

    Score formula:
        score(q, d) = dot(q, d) / (||q||₂ × sqrt(doc_nnz))

    This is a deliberate asymmetric cosine: the query is fully L2-normalised
    (preserving IDF weighting from phrase scores), while the document side uses
    sqrt(nnz) — a mild length penalty that avoids over-penalising dense SDRs
    compared to full L2 normalisation of binary vectors.

    Parameters
    ----------
    query_fp : csr_matrix
        Query fingerprint vector ``(1, grid_size²)``, float-weighted and
        ideally L2-normalised upstream (process_query Stage 2).
    doc_fingerprints : Dict[str, csr_matrix]
        Mapping of ``{doc_id: fingerprint_vector}``.
    top_k : int
        Number of top results to return.
    min_similarity : float, optional
        Minimum score threshold; documents below this are excluded.
    use_batch : bool, optional
        Accepted for API compatibility; batch path used when corpus > 50 docs.
    **kwargs
        Catches any additional legacy keyword arguments without error.

    Returns
    -------
    results : List[Tuple[str, float]]
        Ranked list of ``(doc_id, score)`` tuples, sorted descending by
        score, truncated to ``top_k``.
    metadata : Dict
        ``total_documents``, ``documents_above_threshold``,
        ``mean_similarity``, ``max_similarity``.
    """
    logger.debug(
        f"rank_documents: query_nnz={query_fp.nnz if query_fp is not None else 0}, "
        f"corpus_size={len(doc_fingerprints)}, top_k={top_k}, "
        f"min_similarity={min_similarity}"
    )

    empty_meta = {
        "total_documents":           0,
        "documents_above_threshold": 0,
        "mean_similarity":           0.0,
        "max_similarity":            0.0,
    }

    if query_fp is None or query_fp.nnz == 0:
        logger.warning("Query fingerprint is empty — returning no results")
        return [], empty_meta

    if not doc_fingerprints:
        logger.warning("No document fingerprints provided — returning no results")
        return [], empty_meta

    # Precompute query L2 norm once — reused for every document
    query_norm = np.sqrt(query_fp.power(2).sum())
    if query_norm < 1e-9:
        logger.warning("Query fingerprint has near-zero norm — returning no results")
        return [], empty_meta

    logger.debug(f"  [RANK] query_norm={query_norm:.4f}")

    all_scores: List[Tuple[str, float]] = []
    skipped_empty = 0

    for doc_id, doc_fp in doc_fingerprints.items():
        if doc_fp.nnz == 0:
            skipped_empty += 1
            logger.debug(f"  [RANK SKIP] '{doc_id}' has zero active bits")
            continue

        raw_dot = float(query_fp.dot(doc_fp.T).toarray()[0, 0])
        # Asymmetric cosine: full query norm, sqrt(nnz) for doc length penalty
        score   = raw_dot / (query_norm * np.sqrt(doc_fp.nnz))

        logger.debug(
            f"  [RANK SCORE] '{doc_id}' raw_dot={raw_dot:.4f}, "
            f"doc_nnz={doc_fp.nnz}, query_norm={query_norm:.4f}, "
            f"score={score:.4f}"
        )
        all_scores.append((doc_id, score))

    if skipped_empty:
        logger.debug(f"  [RANK] skipped {skipped_empty} empty document fingerprints")

    if not all_scores:
        logger.warning("All documents scored zero — returning empty results")
        return [], {
            "total_documents":           len(doc_fingerprints),
            "documents_above_threshold": 0,
            "mean_similarity":           0.0,
            "max_similarity":            0.0,
        }

    raw_scores = [s for _, s in all_scores]
    mean_sim   = float(np.mean(raw_scores))
    max_sim    = float(np.max(raw_scores))

    logger.debug(
        f"  [RANK STATS] scored={len(all_scores)}, "
        f"mean={mean_sim:.4f}, max={max_sim:.4f}, "
        f"threshold={min_similarity}"
    )

    filtered = [
        (doc_id, score) for doc_id, score in all_scores
        if score >= min_similarity
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        f"  [RANK FILTER] {len(all_scores)} → {len(filtered)} docs "
        f"above threshold={min_similarity}, returning top {top_k}"
    )

    metadata = {
        "total_documents":           len(doc_fingerprints),
        "documents_above_threshold": len(filtered),
        "mean_similarity":           mean_sim,
        "max_similarity":            max_sim,
    }

    logger.info(
        f"Ranking: {len(filtered)}/{len(doc_fingerprints)} docs above threshold, "
        f"top score={max_sim:.4f}, mean={mean_sim:.4f}"
    )
    return filtered[:top_k], metadata


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def display_results(
    results          : List[Tuple[str, float]],
    query            : str,
    query_metadata   : Dict,
    ranking_metadata : Dict,
    doc_metadata     : Optional[Dict[str, Dict]] = None,
    verbose          : bool = False,
) -> None:
    """
    Print ranked results to stdout in a human-readable tabular format.

    Parameters
    ----------
    results : List[Tuple[str, float]]
        Ranked ``(doc_id, score)`` pairs.
    query : str
        Original raw query string (used as section heading).
    query_metadata : Dict
        Metadata from :func:`construct_query_fingerprint`.
    ranking_metadata : Dict
        Metadata from :func:`rank_documents`.
    doc_metadata : Dict or None, optional
        Optional per-document annotations for verbose mode.
    verbose : bool, optional
        Print full diagnostic blocks when ``True`` (default: ``False``).
    """
    logger.debug(
        f"display_results: query={query!r}, n_results={len(results)}, "
        f"verbose={verbose}"
    )

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    if verbose:
        print("\nQuery Analysis:")
        print(
            f"  Phrases matched : "
            f"{query_metadata.get('num_matched', 0)}/"
            f"{query_metadata.get('num_phrases', 0)}"
        )
        print(f"  Active bits     : {query_metadata.get('active_bits', 0)}")
        print(f"  Sparsity        : {query_metadata.get('sparsity', 0):.4f}")

        missing = query_metadata.get("missing_phrases", [])
        if missing:
            print(f"  Missing phrases : {', '.join(missing)}")
            logger.debug(f"  [DISPLAY OOV] {missing}")

        print("\nCorpus Statistics:")
        print(f"  Total documents : {ranking_metadata.get('total_documents', 0)}")
        print(f"  Mean similarity : {ranking_metadata.get('mean_similarity', 0):.4f}")
        print(f"  Max similarity  : {ranking_metadata.get('max_similarity', 0):.4f}")

    print(f"\nTop {len(results)} Results:")
    print("-" * 80)

    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"{rank:2d}. {doc_id:50s} | Score: {score:.4f}")
        logger.debug(f"  [DISPLAY RANK {rank}] '{doc_id}' score={score:.4f}")

        if verbose and doc_metadata and doc_id in doc_metadata:
            meta = doc_metadata[doc_id]
            if "matched_phrases" in meta:
                print(f"    Matched phrases : {meta['matched_phrases']}")
            if "coverage" in meta:
                print(f"    Coverage        : {meta['coverage']:.3f}")

    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end single query processor
# ─────────────────────────────────────────────────────────────────────────────
def process_query(
    query               : str,
    phrase_fingerprints : Dict[str, csr_matrix],
    doc_fingerprints    : Dict[str, csr_matrix],
    args                : argparse.Namespace,
    idf_weights         : Optional[Dict[str, float]] = None,
) -> Tuple[List[Tuple[str, float]], Dict]:
    r"""
    Process a single query through the full retrieval pipeline.

    The pipeline has four stages:

    1. **Phrase Extraction + OOV Expansion**
       - Extract matched phrases from the query that exist in
         ``phrase_vocab`` (the keys of ``phrase_fingerprints``).
       - Re-run the raw extraction pipeline (spaCy or fallback n-gram) to
         collect *all* candidate phrases before the vocabulary filter.
       - Compute two disjoint sets from ``all_expanded``:

         * ``vocab_hits_from_raw`` — candidates that *are* in
           ``phrase_vocab`` but were missed by ``extract_query_phrases``
           (typically multi-word phrases).
         * ``oov_terms`` — candidates that are *absent* from
           ``phrase_vocab`` and require fingerprint-similarity expansion.

       - Merge ``matched_phrases`` (from ``extract_query_phrases``) with
         ``vocab_hits_from_raw`` into ``combined_matched`` so that
         multi-word vocabulary phrases receive their IDF weight and
         contribute to the query fingerprint.
       - Build a fingerprint index over the entire phrase vocabulary and
         use it to find, for each OOV term, the most semantically similar
         in-vocabulary phrases (scored by fingerprint cosine similarity).
       - Merge ``combined_matched`` and OOV expansions into a single
         ``phrase_weights`` dict.  Matched phrases receive their
         IDF-derived weight (or 1.0 if no IDF table is provided); OOV
         substitutes receive a discounted weight
         (default ``0.6 × matched phrase weight``).

    2. **Query Fingerprint Construction**
       - Call :func:`construct_query_fingerprint` with
         ``weighting="uniform"`` to obtain construction metadata.
       - Re-accumulate the fingerprint manually using the actual weights
         from ``phrase_weights`` so that IDF and expansion discounts are
         reflected in the final vector.
       - Optionally normalise the result according to ``args.normalization``.

    3. **Spreading Activation** *(optional)*
       - If ``args.spreading_steps > 0``, apply spatial spreading over the
         2-D fingerprint grid with the given radius and decay factor.

    4. **Document Ranking**
       - Compute weighted overlap between the query fingerprint and every
         document fingerprint in ``doc_fingerprints``.
       - Return the top-``k`` results together with ranking metadata.

    Parameters
    ----------
    query : str
        Raw natural-language query string.
    phrase_fingerprints : Dict[str, csr_matrix]
        Mapping from phrase text to its pre-computed sparse fingerprint
        (shape ``(1, grid_size²)``).  Used both as the vocabulary source
        and for fingerprint lookup during accumulation.
    doc_fingerprints : Dict[str, csr_matrix]
        Mapping from document identifier to its pre-computed sparse
        fingerprint (shape ``(1, grid_size²)``).
    args : argparse.Namespace
        Runtime configuration.  Recognised attributes:

        * ``no_spacy``                  – bool, skip spaCy extraction
        * ``remove_verbs``              – bool, strip verb tokens
        * ``filter_generic``            – bool, remove high-frequency stopwords
        * ``min_word_length``           – int, minimum token character length
        * ``normalization``             – str, one of ``"l1"``, ``"l2"``,
          ``"max"``, ``"none"``
        * ``phrase_fp_dir``             – Path, directory containing
          per-phrase fingerprint JSON files
        * ``grid_size``                 – int, square root of fingerprint
          dimension (default 128)
        * ``spreading_steps``           – int, spreading radius in grid cells
        * ``spreading_decay``           – float, per-step decay multiplier
        * ``normalize_after_spreading`` – bool
        * ``top_k``                     – int, number of results to return
        * ``min_similarity``            – float, score cutoff
        * ``use_batch``                 – bool, use batched similarity
    idf_weights : Dict[str, float], optional
        Mapping from phrase text to its IDF weight.  When provided,
        matched phrases are weighted by their IDF score; OOV expansions
        receive ``expansion_weight × IDF``.  When ``None``, all matched
        phrases receive weight 1.0.

    Returns
    -------
    results : List[Tuple[str, float]]
        Ranked list of ``(document_id, score)`` pairs, at most
        ``args.top_k`` entries, all with score ≥ ``args.min_similarity``.
    metadata : Dict
        Diagnostic dictionary with keys:

        * ``"query"``              – the original query string
        * ``"query_construction"`` – metadata from
          :func:`construct_query_fingerprint`
        * ``"spreading"``          – metadata from :func:`apply_spreading`
          (empty dict if spreading was skipped)
        * ``"ranking"``            – metadata from :func:`rank_documents`

        On failure the dict additionally contains an ``"error"`` key with
        a short error code and ``results`` is an empty list.

    Notes
    -----
    * ``extract_query_phrases`` is vocabulary-filtered and typically
      returns only single-token or very common multi-word matches.
      Multi-word phrases like "cognitive skills" that exist in the vocab
      but are not returned by that function are recovered via
      ``vocab_hits_from_raw`` — the intersection of ``all_expanded``
      (raw spaCy/n-gram candidates) and ``phrase_vocab``.
    * :func:`construct_query_fingerprint` is called with
      ``weighting="uniform"`` and its output fingerprint is immediately
      overridden by the manual re-accumulation loop.  The call is used
      only to obtain ``query_metadata``; the weighted accumulation is the
      authoritative fingerprint.
    """
    logger.debug(
        f"process_query: query={query!r}, "
        f"vocab_size={len(phrase_fingerprints)}, "
        f"corpus_size={len(doc_fingerprints)}"
    )

    phrase_vocab    = set(phrase_fingerprints.keys())
    use_spacy       = not getattr(args, "no_spacy",        False)
    remove_verbs    = getattr(args, "remove_verbs",        False)
    filter_generic  = getattr(args, "filter_generic",      True)
    min_word_length = getattr(args, "min_word_length",     3)

    # ── Stage 1: phrase extraction + OOV expansion ───────────────────────────
    logger.debug("  [STAGE 1] phrase extraction + OOV expansion")

    # Primary extraction: vocabulary-filtered, typically returns single tokens
    # and the most common short phrases, but often misses multi-word vocab hits.
    matched_phrases = extract_query_phrases(
        query, phrase_vocab,
        use_spacy=use_spacy, remove_verbs=remove_verbs,
        filter_generic=filter_generic, min_word_length=min_word_length,
    )
    logger.debug(f"  [STAGE 1] matched_phrases={matched_phrases}")

    # Re-run raw extraction to collect every candidate before vocab filter.
    # This is the source of truth for both multi-word vocab hits and OOV terms.
    if use_spacy and SPACY_AVAILABLE:
        doc = nlp(query)
        raw = extract_raw_phrases_spacy(doc)
    else:
        raw = extract_raw_phrases_fallback(query, max_ngram=4)

    logger.debug(f"  [STAGE 1] raw candidates={raw}")

    # Normalise and expand every raw candidate into a flat list
    all_expanded = []
    for p in raw:
        norm_p = normalize_phrase(p, remove_verbs=remove_verbs)
        if norm_p:
            all_expanded.append(norm_p)

    all_expanded = expand_phrases(
        all_expanded,
        context_text=query,
        filter_generic=filter_generic,
        min_word_length=min_word_length,
    )
    logger.debug(f"  [STAGE 1] expanded candidates={all_expanded}")

    # ── Partition all_expanded into vocab hits and OOV terms ─────────────────
    #
    # vocab_hits_from_raw: candidates that ARE in phrase_vocab but were NOT
    #   returned by extract_query_phrases (typically multi-word phrases like
    #   "cognitive skills", "human brain", "new neural connection").
    #   These carry genuine fingerprint coordinates and must be included so
    #   that multi-word concepts contribute to the query vector.
    #
    # oov_terms: candidates absent from phrase_vocab entirely — handled by
    #   expand_oov_query_terms via fingerprint cosine similarity.
    vocab_hits_from_raw = [p for p in all_expanded if p in phrase_vocab]
    oov_terms           = [p for p in all_expanded if p not in phrase_vocab]

    logger.debug(
        f"  [STAGE 1] vocab hits from raw extraction "
        f"({len(vocab_hits_from_raw)}): {vocab_hits_from_raw}"
    )
    logger.debug(f"  [STAGE 1] OOV terms ({len(oov_terms)}): {oov_terms}")
    logger.info(f"OOV terms identified for expansion: {oov_terms}")

    # ── Build combined_matched: merge both sources of vocabulary hits ─────────
    #
    # Priority order (first writer wins for duplicates):
    #   1. matched_phrases  — from extract_query_phrases (may carry pre-computed
    #                         weights if returned as a dict)
    #   2. vocab_hits_from_raw — recovered multi-word hits, base weight 1.0
    #                            (IDF scaling is applied inside merge_expanded_phrases)
    combined_matched: Dict[str, float] = {}

    if isinstance(matched_phrases, dict):
        # extract_query_phrases returned pre-weighted dict — preserve weights
        combined_matched.update(matched_phrases)
    else:
        # extract_query_phrases returned a plain list — assign base weight 1.0
        combined_matched.update({p: 1.0 for p in matched_phrases})

    for p in vocab_hits_from_raw:
        if p not in combined_matched:
            # New multi-word hit not seen by extract_query_phrases
            combined_matched[p] = 1.0   # IDF will rescale this in merge step
            logger.debug(f"  [STAGE 1] added raw vocab hit: '{p}'")
        else:
            logger.debug(
                f"  [STAGE 1] raw vocab hit '{p}' already in combined_matched "
                f"(weight={combined_matched[p]:.4f}) — kept existing weight"
            )

    logger.debug(
        f"  [STAGE 1] combined_matched ({len(combined_matched)}): "
        f"{list(combined_matched.keys())}"
    )

    # ── Build in-memory fingerprint index for OOV expansion ──────────────────
    #
    # Building from phrase_fingerprints avoids redundant disk I/O; the same
    # csr_matrix objects are already loaded in memory.
    vocab_fp_index = {
        p: (fp.toarray().ravel() if hasattr(fp, "toarray")
            else np.asarray(fp).ravel())
        for p, fp in phrase_fingerprints.items()
    }
    logger.debug(
        f"  [STAGE 1] vocab_fp_index built in memory "
        f"({len(vocab_fp_index)} entries)"
    )

    oov_expansions = expand_oov_query_terms(
        oov_terms=oov_terms,
        vocab_fp_index=vocab_fp_index,
        phrase_fp_dir=str(args.phrase_fp_dir),
        grid_size=getattr(args, "grid_size", 128),
    )
    logger.debug(f"  [STAGE 1] oov_expansions={oov_expansions}")

    # ── Merge combined_matched + OOV expansions into final phrase_weights ─────
    #
    # combined_matched (not the original matched_phrases) is passed here so
    # that multi-word vocab hits recovered from raw extraction are included.
    phrase_weights = merge_expanded_phrases(
        matched_phrases=combined_matched,   # Bug 3 fix: was matched_phrases
        oov_expansions=oov_expansions,
        idf_weights=idf_weights,
        expansion_weight=0.6,
    )
    logger.debug(
        f"  [STAGE 1] phrase_weights ({len(phrase_weights)}): {phrase_weights}"
    )

    if not phrase_weights:
        logger.error(f"No valid phrases found in query: {query!r}")
        return [], {
            "query"             : query,
            "error"             : "no_phrases_extracted",
            "query_construction": {},
            "spreading"         : {},
            "ranking"           : {
                "total_documents"          : 0,
                "documents_above_threshold": 0,
            },
        }

    # ── Stage 2: query fingerprint construction ───────────────────────────────
    logger.debug("  [STAGE 2] query fingerprint construction")

    norm = None if args.normalization == "none" else args.normalization

    # construct_query_fingerprint is called for its metadata side-effect only.
    # The returned fingerprint uses uniform weights and is immediately replaced
    # by the manual weighted accumulation below.
    _effective_weighting = getattr(args, "weighting", "uniform")
    query_fp, query_metadata = construct_query_fingerprint(
        query_phrases=list(phrase_weights.keys()),
        phrase_fingerprints=phrase_fingerprints,
        weighting=_effective_weighting,          # ← respects --weighting
        idf_weights=idf_weights,                 # ← respects --idf
        normalization=getattr(args, "normalization", "l2"),
    )

    # Manual re-accumulation: apply per-phrase weights (IDF + expansion discounts)
    # so the final vector correctly reflects the importance of each phrase.
    if query_fp is not None:
        grid_size_sq = query_fp.shape[1]
        acc = np.zeros(grid_size_sq, dtype=np.float32)

        for phrase, weight in phrase_weights.items():
            if phrase in phrase_fingerprints:
                fp     = phrase_fingerprints[phrase]
                fp_arr = (fp.toarray().ravel() if hasattr(fp, "toarray")
                          else np.asarray(fp).ravel())
                pre_nnz = int(np.count_nonzero(acc))
                acc    += weight * fp_arr
                logger.debug(
                    f"  [STAGE 2 ACCUM] '{phrase}' weight={weight:.4f}, "
                    f"acc nnz: {pre_nnz} → {int(np.count_nonzero(acc))}"
                )
            else:
                logger.debug(
                    f"  [STAGE 2 ACCUM SKIP] '{phrase}' not in phrase_fingerprints"
                )

        from scipy.sparse import csr_matrix as _csr
        from lib import normalize_fingerprint
        query_fp = _csr(acc.reshape(1, -1))

        logger.debug(
            f"  [STAGE 2] pre-norm nnz={query_fp.nnz}, "
            f"norm='{norm}'"
        )

        if norm:
            query_fp = normalize_fingerprint(query_fp, method=norm)
            logger.debug(f"  [STAGE 2] post-norm nnz={query_fp.nnz}")

    if query_fp is None:
        error_type = query_metadata.get("error", "unknown")
        logger.error(f"Failed to construct query fingerprint: {error_type}")
        return [], {
            "query"             : query,
            "error"             : error_type,
            "query_construction": query_metadata,
            "spreading"         : {},
            "ranking"           : {
                "total_documents"          : 0,
                "documents_above_threshold": 0,
            },
        }

    logger.debug(
        f"  [STAGE 2] fingerprint ready — nnz={query_fp.nnz}, "
        f"shape={query_fp.shape}"
    )

    # ── Stage 3: spreading activation (optional) ──────────────────────────────
    spreading_metadata: Dict = {}
    spreading_steps = getattr(args, "spreading_steps", 0)

    logger.debug(f"  [STAGE 3] spreading_steps={spreading_steps}")

    if spreading_steps > 0:
        grid_size = int(np.sqrt(query_fp.shape[1]))
        logger.debug(
            f"  [STAGE 3] applying spreading: grid_size={grid_size}, "
            f"radius={spreading_steps}, "
            f"decay={getattr(args, 'spreading_decay', 0.5)}"
        )
        query_fp, spreading_metadata = apply_spreading(
            query_fp, grid_size,
            radius=spreading_steps,
            decay=getattr(args, "spreading_decay", 0.5),
            normalize_after=getattr(args, "normalize_after_spreading", True),
        )
    else:
        logger.debug("  [STAGE 3] spreading skipped (spreading_steps=0)")

    # ── Stage 4: document ranking ─────────────────────────────────────────────
    logger.debug(
        f"  [STAGE 4] ranking {len(doc_fingerprints)} documents, "
        f"top_k={getattr(args, 'top_k', 10)}, "
        f"min_similarity={getattr(args, 'min_similarity', 0.0)}"
    )

    results, ranking_metadata = rank_documents(
        query_fp, doc_fingerprints,
        top_k=getattr(args, "top_k", 10),
        min_similarity=getattr(args, "min_similarity", 0.0),
        use_batch=getattr(args, "use_batch", True),
    )

    top_score = f"{results[0][1]:.4f}" if results else "n/a"
    logger.debug(
        f"  [STAGE 4] returned {len(results)} results, "
        f"top_score={top_score}"
    )

    logger.info(
        f"process_query done: {len(results)} results for {query!r}"
    )

    return results, {
        "query"             : query,
        "query_construction": query_metadata,
        "spreading"         : spreading_metadata,
        "ranking"           : ranking_metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for the Step-6 CLI.

    All phrase-extraction flags (``--no-spacy``, ``--remove-verbs`` /
    ``--keep-verbs``, ``--no-filter-generic``, ``--min-word-length``)
    **must** be set to the same values used in Step 1
    (``phrase_extractor.py``) to guarantee consistent phrase
    representations across the pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace consumed by :func:`main`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 6 — Process queries against document fingerprints "
            "using the Semantic Folding pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required I/O ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query string to process.",
    )
    parser.add_argument(
        "--phrase-fp-dir", dest="phrase_fp_dir", type=Path, required=True,
        help="Step 4 phrase fingerprint directory.",
    )
    parser.add_argument(
        "--doc-fp-dir", dest="doc_fp_dir", type=Path, required=True,
        help="Step 5 document fingerprint directory.",
    )

    # ── Optional inputs ───────────────────────────────────────────────────────
    parser.add_argument(
        "--idf", dest="idf_weights", type=Path, default=None,
        help="IDF weights JSON file (required when --weighting idf).",
    )
    parser.add_argument(
        "--query-file", dest="query_file", type=Path, default=None,
        help="Text file with one query per line (alternative to --query).",
    )

    # ── Grid parameters ───────────────────────────────────────────────────────
    parser.add_argument(
        "--grid-size", dest="grid_size", type=int, default=16,
        help="Side length of the N×N semantic grid. Must match Steps 3–5.",
    )

    # ── Phrase extraction flags (must mirror Step 1 settings) ─────────────────
    parser.add_argument(
        "--no-spacy", dest="no_spacy", action="store_true", default=False,
        help="Force NLTK fallback extraction (use if Step 1 used --no-spacy).",
    )
    parser.add_argument(
        "--keep-verbs", dest="remove_verbs", action="store_false", default=False,
        help="Keep verb forms (default: on, mirrors Step 1 --keep-verbs).",
    )
    parser.add_argument(
        "--no-filter-generic", dest="filter_generic", action="store_false",
        default=True,
        help="Keep generic single words during expansion (mirrors Step 1 flag).",
    )
    parser.add_argument(
        "--min-word-length", dest="min_word_length", type=int, default=3,
        help="Minimum token character length kept after expansion.",
    )

    # ── Weighting / normalisation ──────────────────────────────────────────────
    parser.add_argument(
        "--weighting", type=str, default="uniform",
        choices=["uniform", "frequency", "idf"],
        help="Phrase weighting strategy for fingerprint aggregation.",
    )
    parser.add_argument(
        "--normalization", type=str, default="l2",
        choices=["l2", "l1", "binary", "none"],
        help="Query fingerprint normalisation method.",
    )

    # ── Spreading ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--spreading-steps", dest="spreading_steps", type=int, default=1,
        help="Spreading radius in Z-order grid (0 to disable).",
    )
    parser.add_argument(
        "--spreading-decay", dest="spreading_decay", type=float, default=0.5,
        help="Decay factor per step during spreading.",
    )
    parser.add_argument(
        "--normalize-after-spreading", dest="normalize_after_spreading",
        action="store_true", default=False,
        help="L2-normalise fingerprint after spreading.",
    )

    # ── Ranking ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--top-k", dest="top_k", type=int, default=10,
        help="Maximum number of results to return per query.",
    )
    parser.add_argument(
        "--min-similarity", dest="min_similarity", type=float, default=0.0,
        help="Minimum score threshold for results.",
    )
    parser.add_argument(
        "--use-batch", dest="use_batch", action="store_true", default=True,
        help="Accepted for compatibility; dot-product ranking is always used.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", dest="output_json", type=Path, default=None,
        help="Save all query results to this JSON file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed query analysis and corpus statistics.",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point for Step 6 of the Semantic Folding pipeline.

    Loads phrase and document fingerprints, optionally loads IDF weights,
    collects queries from ``--query`` and/or ``--query-file``, processes
    each query via :func:`process_query`, displays results, and optionally
    saves all results to a JSON file.

    Exit codes
    ----------
    0   — Success (all queries processed; some may have had no results).
    1   — Fatal error (missing input directories or empty fingerprint dicts).
    """
    args = parse_args()

    logger.debug(f"main: args={vars(args)}")

    # ── Validate mutually dependent arguments ─────────────────────────────────
    if not args.query and not args.query_file:
        logger.error("Either --query or --query-file must be provided.")
        sys.exit(1)

    if args.weighting == "idf" and not args.idf_weights:
        logger.warning(
            "IDF weighting requested but --idf not provided → "
            "falling back to uniform."
        )
        args.weighting = "uniform"

    # ── Validate input directories ────────────────────────────────────────────
    if not args.phrase_fp_dir.exists():
        logger.error(f"Phrase fingerprint dir not found: {args.phrase_fp_dir}")
        sys.exit(1)

    if not args.doc_fp_dir.exists():
        logger.error(f"Document fingerprint dir not found: {args.doc_fp_dir}")
        sys.exit(1)

    logger.debug(
        f"  [MAIN] phrase_fp_dir={args.phrase_fp_dir}, "
        f"doc_fp_dir={args.doc_fp_dir}"
    )

    # ── Load phrase fingerprints ───────────────────────────────────────────────
    logger.debug(
        f"  [MAIN LOAD] loading phrase fingerprints from {args.phrase_fp_dir}"
    )
    try:
        phrase_fingerprints = load_phrase_fingerprints_sparse(
            args.phrase_fp_dir, args.grid_size
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Failed to load phrase fingerprints: {exc}")
        sys.exit(1)

    if not phrase_fingerprints:
        logger.error("Phrase fingerprints dict is empty — check Step 4 output.")
        sys.exit(1)

    logger.info(f"Loaded {len(phrase_fingerprints)} phrase fingerprints.")

    # ── Load IDF weights ───────────────────────────────────────────────────────
    idf_weights: Optional[Dict[str, float]] = None
    if args.weighting == "idf":
        if args.idf_weights and args.idf_weights.exists():
            logger.debug(f"  [MAIN LOAD] loading IDF weights from {args.idf_weights}")
            try:
                with open(args.idf_weights, "r", encoding="utf-8") as fh:
                    idf_weights = json.load(fh)
                logger.info(
                    f"Loaded IDF weights for {len(idf_weights)} phrases."
                )
                logger.debug(
                    f"  [MAIN LOAD] IDF range: "
                    f"min={min(idf_weights.values()):.4f}, "
                    f"max={max(idf_weights.values()):.4f}"
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    f"Failed to read IDF weights ({exc}) → "
                    f"falling back to uniform."
                )
                args.weighting = "uniform"
        else:
            logger.warning(
                f"IDF weights file not found: {args.idf_weights} "
                f"→ falling back to uniform."
            )
            args.weighting = "uniform"

    # ── Load document fingerprints ─────────────────────────────────────────────
    logger.debug(
        f"  [MAIN LOAD] loading document fingerprints from {args.doc_fp_dir}"
    )
    try:
        doc_fingerprints, doc_metadata = load_document_fingerprints(args.doc_fp_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Failed to load document fingerprints: {exc}")
        sys.exit(1)

    if not doc_fingerprints:
        logger.error(
            "Document fingerprints dict is empty — check Step 5 output."
        )
        sys.exit(1)

    logger.info(f"Loaded {len(doc_fingerprints)} document fingerprints.")
    logger.debug(
        f"  [MAIN LOAD] doc_metadata keys: {list(doc_metadata.keys())[:5]}..."
        if doc_metadata else "  [MAIN LOAD] no doc_metadata"
    )

    # ── Collect queries ────────────────────────────────────────────────────────
    queries: List[str] = []

    if args.query:
        queries.append(args.query.strip())
        logger.debug(f"  [MAIN QUERIES] inline query: {args.query!r}")

    if args.query_file:
        if not args.query_file.exists():
            logger.error(f"Query file not found: {args.query_file}")
            sys.exit(1)
        try:
            with open(args.query_file, "r", encoding="utf-8") as fh:
                file_queries = [ln.strip() for ln in fh if ln.strip()]
            queries.extend(file_queries)
            logger.info(
                f"Loaded {len(file_queries)} queries from {args.query_file}."
            )
            logger.debug(
                f"  [MAIN QUERIES] first 3 from file: {file_queries[:3]}"
            )
        except OSError as exc:
            logger.error(f"Could not read query file: {exc}")
            sys.exit(1)

    if not queries:
        logger.error("No queries to process.")
        sys.exit(1)

    logger.info(f"Processing {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}.")

    # ── Process queries ────────────────────────────────────────────────────────
    all_results = []

    for i, query in enumerate(queries, 1):
        logger.info(f"[{i}/{len(queries)}] Processing: {query!r}")

        results, metadata = process_query(
            query,
            phrase_fingerprints,
            doc_fingerprints,
            args,
            idf_weights,
        )

        if "error" in metadata:
            logger.error(
                f"Query [{i}] failed — {metadata['error']}: {query!r}"
            )
            missing = (
                metadata
                .get("query_construction", {})
                .get("missing_phrases", [])
            )
            if missing:
                logger.debug(f"  [MAIN ERROR] OOV phrases: {missing}")
        else:
            top_score = f"{results[0][1]:.4f}" if results else "n/a"
            logger.debug(
                f"  [MAIN RESULT] query [{i}] — "
                f"{len(results)} results, top={top_score}"
            )

            display_results(
                results,
                query,
                metadata["query_construction"],
                metadata["ranking"],
                doc_metadata=doc_metadata,
                verbose=args.verbose,
            )

        all_results.append({
            "query":    query,
            "results":  [(doc_id, float(score)) for doc_id, score in results],
            "metadata": metadata,
        })

    logger.debug(
        f"  [MAIN] all {len(queries)} quer"
        f"{'y' if len(queries) == 1 else 'ies'} processed"
    )

    # ── Save output ────────────────────────────────────────────────────────────
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"  [MAIN SAVE] writing results to {args.output_json}")
        try:
            with open(args.output_json, "w", encoding="utf-8") as fh:
                json.dump(all_results, fh, indent=2, ensure_ascii=False)
            logger.success(f"Results saved → {args.output_json}")
        except OSError as exc:
            logger.error(f"Failed to write output file: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
