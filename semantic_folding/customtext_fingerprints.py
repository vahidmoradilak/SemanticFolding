#!/usr/bin/env python3
"""
customtext_fingerprints.py — Step 6 of the Semantic Folding Pipeline

Aggregates phrase-level sparse fingerprints (Step 4) into document-level
Sparse Distributed Representations (SDRs) using TF-IDF weighted union,
then sparsifies via topology-preserving peak detection on 2D semantic grids.

Pipeline position
-----------------
Step 1  phrase_extractor.py        → vocabulary.csv
Step 2  term_context.py            → term_context_matrix.*, idf_weights.json
Step 3  semantic_space.py          → context_coordinates.json
Step 4  phrase_fingerprints.py     → phrase_fingerprints/
Step 5  doc_fingerprints.py        → doc_fingerprints/
Step 6  customtext_fingerprints.py → customtext_fingerprints/          ← THIS FILE
Step 7  query_processing.py        → query results

Consistency guarantee
---------------------
Every phrase extracted from a document in this step goes through the
**identical** normalization + expansion path used in Step 1
(phrase_extractor.py → process_corpus_with_expansion).  Specifically:

    raw text
        └─ extract_and_normalize_phrases()   # spaCy / NLTK + normalize_phrase()
                └─ expand_phrases()          # sub-phrase generation
                        └─ vocab filter      # keep only known phrase_fps keys

This ensures that a document containing "deep neural network" activates
fingerprints for "deep neural", "neural network", "neural", "network", etc.,
exactly mirroring how the vocabulary was built in Step 1.

Usage
-----
    python customtext_fingerprints.py \
        --corpus      data/customtexts.txt \
        --fingerprints outputs/run/phrase_fingerprints \
        --idf-weights outputs/run/term_context/idf_weights.json \
        --output-dir  outputs/run/customtext_fingerprints \
        --grid-size   16 \
        --top-percent 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.ndimage import maximum_filter, gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
from phrase_extractor import (
    extract_raw_phrases_spacy, 
    extract_raw_phrases_fallback, 
    SPACY_AVAILABLE
)

if SPACY_AVAILABLE:
    from phrase_extractor import nlp 
    
from lib import get_logger
logger = get_logger("customtext_fingerprints")

# NLTK only needed when spaCy is unavailable
if not SPACY_AVAILABLE:
    logger.debug("Importing NLTK fallback tokenizer and POS tagger")
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag


from lib import (
    compute_fingerprint_diversity,
    expand_phrases,
    normalize_phrase,
    is_valid_phrase_structure,
    load_contexts_dict,
    normalize_fingerprint,
    sparsify_fingerprint,
    normalize_hyphens,
    extract_raw_phrases_ar_fa,
    normalize_arabic_phrase,
)

from hazm import Normalizer, word_tokenize
normalizer = Normalizer()

# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_outputs(
    fingerprints  : np.ndarray,
    doc_index_map : Dict[str, int],
    stats         : dict,
    output_dir    : Path,
    use_morton    : bool,               # NEW
    grid_size     : int,                # NEW
) -> None:
    """
    Persist Step‑6 outputs to ``output_dir``.

    Writes three files:
    - doc_fingerprints.npz
    - doc_fingerprints_meta.json  (includes encoding info)
    - doc_fingerprints_stats.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path   = output_dir / "doc_fingerprints.npz"
    meta_path  = output_dir / "doc_fingerprints_meta.json"
    stats_path = output_dir / "doc_fingerprints_stats.json"

    # --- fingerprint matrix ---
    np.savez_compressed(str(npz_path), fingerprints=fingerprints)
    logger.info(f"Fingerprint matrix written → {npz_path}  shape={fingerprints.shape}")

    # --- structured metadata ---
    meta_dict = {
        "doc_to_row": doc_index_map,
        "use_morton": use_morton,
        "grid_size": grid_size,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_dict, fh, ensure_ascii=False, indent=2)
    logger.info(
        f"Metadata written → {meta_path}  ({len(doc_index_map)} docs, "
        f"morton={use_morton}, grid={grid_size})"
    )

    # --- run statistics ---
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    logger.info(f"Run statistics written → {stats_path}")

# ---------------------------------------------------------------------------
# Per-document phrase extractor
# ---------------------------------------------------------------------------

import re
_ARABIC_SCRIPT = re.compile(r'[\u0600-\u06FF]')

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

def extract_phrases_from_doc(
    text            : str,
    phrase_vocab    : Set[str],
    use_spacy       : bool = True,
    remove_verbs    : bool = False,
    filter_generic  : bool = True,
    min_word_length : int  = 3,
) -> List[str]:
    """
    Extract vocabulary-matched phrases from a single document string.

    This function is the Step-5 counterpart of
    ``phrase_extractor.process_corpus_with_expansion`` and must follow the
    **same three-stage pipeline** to guarantee that phrase representations
    are identical at index time (Step 1) and at fingerprint-build time
    (Step 5).
    """
    # ── Stage 1: extraction + normalisation ──────────────────────────────────
    arabic_text, english_text = split_arabic_english(text)
    ar_valid = set()
    if arabic_text:
        ar_row = extract_raw_phrases_ar_fa(arabic_text)
        
        logger.debug(f"for {arabic_text} | {len(ar_row)} Ar raw phrases extracted")
        if not ar_row:
            logger.debug(f"for {arabic_text} | no Ar raw phrases — skipping expansion")
            return []

        for p in ar_row:
            norm = normalize_arabic_phrase(p)
            if norm:
                ar_valid.add(norm)
                logger.debug(f"[AR][KEEP] '{p}' → '{norm}'")

    en_valid = set()
    if english_text:
        # ── Hyphen normalization ──────────────────────────────────────────
        # Replace intra-word hyphens (e.g. 'rule-based') with spaces so
        # that compound terms are tokenized as multi-word phrases rather
        # than being split into three tokens: word, '-', word.
        # text_clean is used for ALL downstream processing; text_original
        # is kept only for logging/diagnostics.
        english_clean = normalize_hyphens(english_text)
        english_clean_lower = normalize_hyphens(english_text.lower())

        if english_clean != english_text:
            logger.debug(
                f"for {english_clean} | hyphen normalization applied: "
                f"'{english_text[:60]}' → '{english_clean[:60]}'"
            )
        
        # ── Stage 1: raw candidate extraction ────────────────────────────
        if use_spacy and SPACY_AVAILABLE:
            doc = nlp(english_clean) # spaCy sees hyphen-free text
            en_raw = extract_raw_phrases_spacy(doc)
        else:
            en_raw = extract_raw_phrases_fallback(english_clean_lower)

        logger.debug(f"for {english_clean} | {len(en_raw)} En raw phrases extracted")

        if not en_raw:
            logger.debug(f"[CORPUS] Line {english_clean} | no En raw phrases — skipping expansion")
            return []
        
        # candidates: Set[str] = set()
        # for phrase in en_raw:
        #     norm = normalize_phrase(phrase, remove_verbs=remove_verbs)
        #     if norm:
        #         candidates.add(norm)

        # if not candidates:
        #     logger.debug(f"No phrases extracted from english text snippet: {english_clean[:80]!r}...")
        #     return []

        # # ── Stage 2: sub-phrase expansion ────────────────────────────────────────
        # expanded: List[str] = expand_phrases(
        #     list(candidates),
        #     context_text    = english_clean,             
        #     filter_generic  = filter_generic,
        #     min_word_length = min_word_length,
        # )
        
        # ── Stage 2 & 3: expansion + normalization ────────────────────────
        # expand_phrases receives raw (un-normalized) phrases and handles
        # surface validation before normalizing internally.
        # text_clean is passed so that context validation (substring match)
        # works against the same hyphen-free surface form that the extractor saw.
        en_valid = expand_phrases(
            list(en_raw),
            context_text=english_clean,       # must match what extractor saw
            filter_generic=filter_generic,
            min_word_length=min_word_length,
        )

    valid_sub_phrases = list(ar_valid | set(en_valid))

    if not valid_sub_phrases:
        logger.debug(f"for {text} | no valid sub phrases — skipping expansion")
        return []

    logger.debug(
        f"for {text} | {len(valid_sub_phrases)} phrases "
        f"survived expansion/normalization"
    )

    # ── Stage 4: map to context ID ────────────────────────────────────

    # ── Stage 3: vocabulary filter ────────────────────────────────────────────
    # matched: List[str] = [p for p in expanded if p in phrase_vocab]
    # matched: List[str] = [p for p in candidates if p in phrase_vocab]
    matched: List[str] = [p for p in valid_sub_phrases if p in phrase_vocab]

    if not matched:
        logger.debug(f"No vocabulary matches in text snippet: {text[:80]!r}...")

    return matched


# ---------------------------------------------------------------------------
# Morton encoding helpers
# ---------------------------------------------------------------------------

def _spread_bits(value: int) -> int:
    """
    Spread bits of a 16-bit integer to prepare for Morton encoding.
    Used to interleave x and y coordinates into a single Z-order index.
    """
    value &= 0x0000FFFF
    value = (value | (value << 8))  & 0x00FF00FF
    value = (value | (value << 4))  & 0x0F0F0F0F
    value = (value | (value << 2))  & 0x33333333
    value = (value | (value << 1))  & 0x55555555
    return value


def xy_to_morton(x: int, y: int, grid_size: int) -> int:
    """
    Convert 2D grid coordinates (x, y) to 1D Morton (Z-order) index.
    
    Morton encoding preserves 2D spatial locality when flattening to 1D,
    ensuring that nearby cells in 2D space remain nearby in the 1D array.
    
    Parameters
    ----------
    x, y : int
        Grid coordinates (0-indexed).
    grid_size : int
        Side length of the square grid (unused but kept for API consistency).
    
    Returns
    -------
    int
        Morton-encoded 1D index.
    """
    return _spread_bits(x) | (_spread_bits(y) << 1)


# ---------------------------------------------------------------------------
# Single-document fingerprint builder (2D topology-preserving)
# ---------------------------------------------------------------------------
def morton_to_xy(index: int, grid_size: int) -> Tuple[int, int]:
    """Inverse of xy_to_morton: converts a Morton (Z-order) index back to (x, y)."""
    # Extract bits: Morton = (spread(y) << 1) | spread(x)
    x = 0
    y = 0
    bit = 0
    while index > 0 or (1 << bit) < grid_size * grid_size:
        # Bit for x is in even positions (0,2,4,...), y in odd (1,3,5,...)
        if index & 1:
            x |= (1 << (bit // 2))
        index >>= 1
        bit += 1
        if index & 1:
            y |= (1 << (bit // 2))
        index >>= 1
        bit += 1
    return x, y

def build_index_to_xy_table(grid_size: int, use_morton: bool = True) -> np.ndarray:
    """
    Create a table of shape (grid_size*grid_size, 2) mapping 1D flat index → (y, x).
    If use_morton is True, index is Morton-encoded; else row-major.
    """
    total = grid_size * grid_size
    table = np.zeros((total, 2), dtype=np.int32)
    if use_morton:
        for idx in range(total):
            x, y = morton_to_xy(idx, grid_size)
            if 0 <= x < grid_size and 0 <= y < grid_size:
                table[idx] = (y, x)
            else:
                # Should not happen if grid_size is power of two
                pass
    else:
        ys = np.arange(total) // grid_size
        xs = np.arange(total) % grid_size
        table[:, 0] = ys
        table[:, 1] = xs
    return table
    
def build_document_fingerprint_2d(
    doc_text            : str,
    phrase_fingerprints : np.ndarray,
    phrase_to_row       : Dict[str, int],
    idf_weights         : Dict[str, float],
    grid_size           : int,
    use_morton          : bool,                     # NEW: whether phrase fingerprints are Morton-encoded
    index_to_xy         : np.ndarray,               # NEW: precomputed lookup table (grid_size², 2) -> (y,x)
    use_spacy           : bool = True,
    remove_verbs        : bool = True,
    filter_generic      : bool = True,
    min_word_length     : int  = 3,
) -> Optional[np.ndarray]:
    """
    Build a 2D semantic activation map for a document.

    Accumulates phrase fingerprints (weighted by TF‑IDF) onto a 2D grid,
    preserving the original spatial topology from Step 4. 
    
    **Critical fix**: Step 4 flattens fingerprints using Morton (Z‑order) 
    encoding, which interleaves x and y bits. A naive `reshape(grid_size, grid_size)` 
    scrambles the spatial layout. This function uses a pre‑computed lookup table 
    `index_to_xy` to scatter 1D values back to their correct (y,x) cells.

    Parameters
    ----------
    doc_text : str
        Raw document text.
    phrase_fingerprints : np.ndarray, shape (n_phrases, grid_size²)
        Pre‑computed phrase fingerprints from Step 4.  
        Each row is a 1D vector in Morton (or row‑major) order.
    phrase_to_row : Dict[str, int]
        Maps phrase string → row index in phrase_fingerprints.
    idf_weights : Dict[str, float]
        IDF weights from Step 2 (phrase → IDF score).
    grid_size : int
        Side length of the semantic grid.
    use_morton : bool
        True if phrase fingerprints are Morton‑encoded; False if row‑major.
    index_to_xy : np.ndarray, shape (grid_size², 2), dtype int
        Lookup table mapping linear index → (y, x) coordinate.  
        Generated by `build_index_to_xy_table(grid_size, use_morton)`.
    use_spacy, remove_verbs, filter_generic, min_word_length : 
        See original docstring.

    Returns
    -------
    np.ndarray or None
        2D float32 array (grid_size, grid_size) of weighted activations,
        or None if no phrases matched the vocabulary.
    """
    logger.debug(f"Building 2D fingerprint for document (length={len(doc_text)} chars)")

    # Initialize 2D grid (not flattened!)
    grid_2d = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Extract phrases using the same pipeline as Step 1
    matched_phrases = extract_phrases_from_doc(
        text            = doc_text,
        phrase_vocab    = set(phrase_to_row.keys()),
        use_spacy       = use_spacy,
        remove_verbs    = remove_verbs,
        filter_generic  = filter_generic,
        min_word_length = min_word_length,
    )

    if not matched_phrases:
        logger.debug("  → No matched phrases, returning None")
        return None

    logger.debug(f"  → Matched {len(matched_phrases)} phrases (with duplicates)")

    # Compute term frequencies (TF)
    tf: Dict[str, int] = {}
    for phrase in matched_phrases:
        tf[phrase] = tf.get(phrase, 0) + 1

    logger.debug(f"  → Unique phrases: {len(tf)}")

    # Accumulate phrase fingerprints in 2D space with TF‑IDF weighting
    hits = 0
    total_weight = 0.0

    for phrase, term_freq in tf.items():
        if phrase not in phrase_to_row:
            logger.debug(f"    ⚠ Phrase '{phrase}' not in phrase_to_row (should not happen)")
            continue

        row_index = phrase_to_row[phrase]
        vec_1d = phrase_fingerprints[row_index]  # shape: (grid_size²,)

        # TF‑IDF weight
        idf = idf_weights.get(phrase, 1.0)
        weight = term_freq * idf
        total_weight += weight

        # ── CORRECT 2D MAPPING ──────────────────────────────────────────
        # The lookup table gives the (y, x) for every 1D index.
        # Using advanced indexing, we add the weighted values directly
        # to the cells where they belong.
        # This replaces the previous (erroneous) vec_2d = vec_1d.reshape(grid_size, grid_size)
        # which scrambled Morton‑encoded vectors.
        np.add.at(grid_2d, (index_to_xy[:, 0], index_to_xy[:, 1]), weight * vec_1d)

        logger.debug(
            f"    + '{phrase[:40]}': TF={term_freq}, IDF={idf:.3f}, "
            f"weight={weight:.3f}, nnz_1d={np.count_nonzero(vec_1d)}"
        )
        hits += 1

    if hits == 0:
        logger.debug("  → No phrase fingerprints accumulated (hits=0)")
        return None

    nnz_2d = np.count_nonzero(grid_2d)
    logger.debug(
        f"  → Accumulated {hits} phrases, total_weight={total_weight:.2f}, "
        f"nnz={nnz_2d} ({100.0 * nnz_2d / (grid_size * grid_size):.2f}% dense)"
    )

    return grid_2d

# ---------------------------------------------------------------------------
# Fallback sparsifier (global top-K)
# ---------------------------------------------------------------------------

def _fallback_global_topk(
    grid_2d     : np.ndarray,
    top_percent : float,
    grid_size   : int,
) -> csr_matrix:
    """
    Fallback sparsification using global top-K selection.
    
    Used when peak detection fails (e.g., uniform activation map).
    Simply selects the K highest-valued cells globally, then flattens
    using Morton encoding.
    
    Parameters
    ----------
    grid_2d : np.ndarray, shape (grid_size, grid_size)
        2D semantic activation map.
    top_percent : float
        Fraction of total bits to activate.
    grid_size : int
        Side length of the grid.
    
    Returns
    -------
    csr_matrix
        Sparse 1D fingerprint of shape (1, grid_size²).
    """
    logger.debug("  [Fallback] Using global top-K sparsification")
    
    flat = grid_2d.flatten()
    top_k = max(1, int(round(top_percent * len(flat))))
    
    if len(flat) >= top_k:
        threshold = np.partition(flat, -top_k)[-top_k]
    else:
        threshold = 0
    
    flat[flat < threshold] = 0
    
    logger.debug(f"    → top_k={top_k}, threshold={threshold:.4f}, nnz={np.count_nonzero(flat)}")
    
    return csr_matrix(flat.reshape(1, -1))


# ---------------------------------------------------------------------------
# Topology-preserving SDR sparsifier
# ---------------------------------------------------------------------------

def sparsify_to_sdr_topological(
    grid_2d           : np.ndarray,
    top_percent       : float,
    grid_size         : int,
    min_peak_distance : int   = 4,
    smoothing_sigma   : float = 1.2,
    contrast_boost    : bool  = True,
) -> csr_matrix:
    """
    Sparsify a 2D semantic grid while preserving topological hotspots.
    
    This is the core innovation for semantic-folding-aligned document SDRs.
    Instead of global top-K selection (which destroys spatial structure),
    we detect local peaks in 2D space and activate regions around each peak
    proportionally to their semantic strength.
    
    Algorithm
    ---------
    1. Apply light Gaussian smoothing to merge nearby activations
    2. Detect local maxima (semantic hotspots) using maximum_filter
    3. Allocate bits to each peak proportionally to its strength
    4. For each peak, activate a local region (top-K within a window)
    5. Flatten to 1D using Morton encoding (preserves 2D locality)
    
    Parameters
    ----------
    grid_2d : np.ndarray, shape (grid_size, grid_size)
        2D semantic activation map from build_document_fingerprint_2d().
    top_percent : float
        Target sparsity (fraction of total bits to activate).
    grid_size : int
        Side length of the square grid.
    min_peak_distance : int
        Minimum distance between detected peaks (in grid cells).
        Larger values → fewer, more separated hotspots.
    smoothing_sigma : float
        Gaussian sigma for pre-smoothing. Helps merge nearby contexts
        into coherent hotspots. Set to 0 to disable.
    
    Returns
    -------
    csr_matrix
        Sparse 1D fingerprint of shape (1, grid_size²), Morton-encoded.
    """
    logger.debug(
        f"Sparsifying 2D grid: size={grid_size}, top_percent={top_percent:.3f}, min_peak_dist={min_peak_distance}, sigma={smoothing_sigma:.2f}"
    )
    
    if contrast_boost:
        mean_val = grid_2d.mean()
        grid_2d = np.maximum(grid_2d - 0.5 * mean_val, 0.0)
    # ── 1. Light smoothing to merge nearby semantic regions ──────────────────
    if smoothing_sigma > 0:
        smoothed = gaussian_filter(grid_2d, sigma=smoothing_sigma)
        logger.debug(f"  → Applied Gaussian smoothing (sigma={smoothing_sigma:.2f})")
    else:
        smoothed = grid_2d.copy()
        logger.debug("  → Skipped smoothing (sigma=0)")
    
    # ── 2. Detect local maxima (semantic hotspots) ───────────────────────────
    neighborhood_size = 2 * min_peak_distance + 1
    local_max = maximum_filter(smoothed, size=neighborhood_size)
    peaks = (smoothed == local_max) & (smoothed > 0)
    
    # Get peak coordinates and their strengths
    peak_coords = np.argwhere(peaks)  # shape: (n_peaks, 2)
    peak_values = smoothed[peaks]
    
    logger.debug(
        f"  → Detected {len(peak_coords)} peaks (neighborhood_size={neighborhood_size})"
    )
    
    if len(peak_coords) == 0:
        logger.warning("  ⚠ No peaks detected, falling back to global top-K")
        return _fallback_global_topk(grid_2d, top_percent, grid_size)
    
    # ── 3. Allocate bits proportionally to peak strengths ────────────────────
    total_bits = int(round(top_percent * grid_size * grid_size))
    
    # Sort peaks by strength (descending)
    sorted_indices = np.argsort(-peak_values)
    peak_coords = peak_coords[sorted_indices]
    peak_values = peak_values[sorted_indices]
    
    logger.debug(f"  → Total bits budget: {total_bits}")
    logger.debug(f"  → Peak strengths (top 5): {peak_values[:5]}")
    
    # Normalize peak values to sum to 1
    peak_weights = peak_values / peak_values.sum()
    
    # Allocate bits per peak (ensure at least 1 bit per peak)
    bits_per_peak = np.maximum(1, (peak_weights * total_bits).astype(int))
    
    logger.debug(f"  → Initial bit allocation: {bits_per_peak[:5]}")
    
    # Adjust to match exact total_bits budget
    adjustment_iterations = 0
    while bits_per_peak.sum() > total_bits:
        max_idx = np.argmax(bits_per_peak)
        bits_per_peak[max_idx] -= 1
        adjustment_iterations += 1
    
    while bits_per_peak.sum() < total_bits:
        bits_per_peak[0] += 1
        adjustment_iterations += 1
    
    if adjustment_iterations > 0:
        logger.debug(f"  → Adjusted bit allocation ({adjustment_iterations} iterations)")
    
    logger.debug(f"  → Final bit allocation: {bits_per_peak[:5]} (sum={bits_per_peak.sum()})")
    
    # ── 4. Activate regions around each peak ─────────────────────────────────
    result_2d = np.zeros_like(grid_2d)
    
    for peak_idx, ((y, x), n_bits) in enumerate(zip(peak_coords, bits_per_peak)):
        # Adaptive radius based on allocated bits
        radius = max(2, int(np.sqrt(n_bits)))
        y_min, y_max = max(0, y - radius), min(grid_size, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(grid_size, x + radius + 1)
        
        window = grid_2d[y_min:y_max, x_min:x_max].copy()
        
        logger.debug(
            f"    Peak {peak_idx} at ({y},{x}): strength={peak_values[peak_idx]:.3f}, bits={n_bits}, radius={radius}, window_shape={window.shape}"
        )
        
        # Select top-K within this window
        flat_window = window.flatten()
        if len(flat_window) > n_bits:
            threshold = np.partition(flat_window, -n_bits)[-n_bits]
            window[window < threshold] = 0
            logger.debug(f"      → Applied threshold={threshold:.4f}, kept {np.count_nonzero(window)}/{len(flat_window)} cells")
        else:
            logger.debug(f"      → Window smaller than bit budget, kept all {len(flat_window)} cells")
        
        # Write back to result (accumulate in case of overlapping windows)
        result_2d[y_min:y_max, x_min:x_max] += window
    
    final_nnz_2d = np.count_nonzero(result_2d)
    logger.debug(
        f"  → Activated {final_nnz_2d} cells in 2D (target was {total_bits}, {100.0 * final_nnz_2d / total_bits if total_bits > 0 else 0:.1f}% match)"
    )
    
    # ── 5. Flatten using Morton encoding ─────────────────────────────────────
    result_1d = np.zeros(grid_size * grid_size, dtype=np.float32)
    
    for y in range(grid_size):
        for x in range(grid_size):
            if result_2d[y, x] > 0:
                morton_idx = xy_to_morton(x, y, grid_size)
                result_1d[morton_idx] = result_2d[y, x]
    
    final_nnz_1d = np.count_nonzero(result_1d)
    logger.debug(f"  → Flattened to 1D (Morton): nnz={final_nnz_1d}")
    
    return csr_matrix(result_1d.reshape(1, -1))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_customtext_fingerprints(
    corpus_path       : Path,
    fingerprints_path : Path,
    idf_weights_path  : Optional[Path],
    grid_size         : int   = 16,
    top_percent       : float = 0.05,
    normalize         : bool  = True,
    normalize_method  : str   = "l2",
    use_spacy         : bool  = True,
    remove_verbs      : bool  = True,
    filter_generic    : bool  = True,
    min_word_length   : int   = 3,
    compute_diversity : bool  = False,
    diversity_sample  : int   = 100,
    min_peak_distance : int   = 2,
    smoothing_sigma   : float = 0.5,
    morton_override: bool = True
) -> Tuple[np.ndarray, Dict[str, int], dict]:
    """
    Full Step-6 pipeline: build document SDRs from phrase fingerprints.

    Loads phrase fingerprints produced by Step 4 (phrase_fingerprints.py),
    optionally weights them with IDF scores from Step 2, then builds one
    sparse SDR per document in the corpus using topology-preserving
    sparsification.

    Supports two formats for phrase_fingerprints_meta.json:
      - Nested  (preferred): {"phrase_to_row": {"phrase": row_idx, ...}}
      - Flat    (legacy):    {"phrase": row_idx, ...}

    Parameters
    ----------
    corpus_path       : Path to the corpus JSON file (doc_id → text).
    fingerprints_path : Directory containing phrase_fingerprints.npz
                        and phrase_fingerprints_meta.json (Step 4 outputs).
    idf_weights_path  : Optional path to idf_weights.json (Step 2 output).
                        If None or missing, falls back to TF-only accumulation.
    grid_size         : SDR grid side length; total bits = grid_size².
    top_percent       : Fraction of bits to activate per document SDR.
    normalize         : Whether to normalise each document fingerprint.
    normalize_method  : Normalisation method passed to normalize_fingerprint()
                        (e.g. "l2", "l1", "max").
    use_spacy         : Use spaCy for phrase extraction (falls back to NLTK
                        if spaCy is unavailable at runtime).
    remove_verbs      : Strip verb tokens during phrase extraction.
    filter_generic    : Remove generic/stopword-heavy phrases.
    min_word_length   : Minimum character length for individual tokens.
    compute_diversity : If True, log pairwise diversity metrics after building.
    diversity_sample  : Number of documents sampled for diversity computation.
    min_peak_distance : Minimum distance between semantic hotspots (grid cells).
    smoothing_sigma   : Gaussian sigma for pre-smoothing before peak detection.
    morton_override : bool
            Fallback value for Morton encoding if the phrase_fingerprints_meta.json
            does not contain a `use_morton` field. Typically taken from the CLI
            `--morton-encoded` / `--no-morton-encoded` flag. Default is True.
    Returns
    -------
    fp_matrix     : np.ndarray of shape (n_docs, grid_size²), dtype float32.
    doc_index_map : Dict mapping doc_id → row index in fp_matrix.
    stats         : Dict with run-level statistics (sparsity, diversity, etc.).
    """
    total_bits = grid_size * grid_size
    target_active = int(total_bits * top_percent)

    logger.info("=" * 70)
    logger.info("Step 6: Building Document Fingerprints (Topology-Preserving)")
    logger.info("=" * 70)
    logger.info(f"Corpus:       {corpus_path}")
    logger.info(f"Fingerprints: {fingerprints_path}")
    logger.info(f"IDF weights:  {idf_weights_path or '(none, using TF only)'}")
    logger.info(f"Grid size:    {grid_size} × {grid_size} = {total_bits} bits")
    logger.info(f"Top percent:  {top_percent*100:.1f}% → ~{target_active} active bits")
    logger.info(f"Peak distance: {min_peak_distance} cells")
    logger.info(f"Smoothing σ:   {smoothing_sigma:.2f}")
    logger.info("-" * 70)

    # ── Load corpus ───────────────────────────────────────────────────────────
    logger.info(f"Loading corpus from {corpus_path}...")
    corpus = load_contexts_dict(corpus_path)

    n_docs = len(corpus)
    logger.info(f"  → Loaded {n_docs} documents")

    if n_docs == 0:
        logger.error("Corpus is empty, cannot proceed")
        sys.exit(1)

    # ── Load phrase fingerprints ──────────────────────────────────────────────
    logger.info(f"Loading phrase fingerprints from {fingerprints_path}...")

    npz_path = fingerprints_path / "phrase_fingerprints.npz"
    meta_path = fingerprints_path / "phrase_fingerprints_meta.json"

    if not npz_path.exists():
        logger.error(f"Phrase fingerprints not found: {npz_path}")
        sys.exit(1)

    if not meta_path.exists():
        logger.error(f"Phrase fingerprints metadata not found: {meta_path}")
        sys.exit(1)

    phrase_fingerprints = np.load(str(npz_path))["fingerprints"]
    logger.info(f"  → Loaded phrase fingerprints: shape={phrase_fingerprints.shape}")

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # Handle both nested and flat metadata formats
    if "phrase_to_row" in meta:
        phrase_to_row = meta["phrase_to_row"]
        logger.info(f"  → Loaded phrase_to_row mapping (nested format): {len(phrase_to_row)} phrases")
    else:
        phrase_to_row = meta
        logger.info(f"  → Loaded phrase_to_row mapping (flat format): {len(phrase_to_row)} phrases")

    # ── Determine Morton encoding usage ───────────────────────────────────────
    # The metadata should include a `use_morton` flag (True/False) written by Step 4.
    # If absent, default to True (the previous default in phrase_fingerprints.py).
    if "use_morton" in meta:
        use_morton = meta["use_morton"]
        logger.info(f"  → Morton encoding flag from metadata: {use_morton}")
    else:
        use_morton = morton_override
        logger.info(f"  → Morton encoding flag not in metadata; using override: {use_morton}")
    logger.info(f"  → Final use_morton: {use_morton}")

    logger.info(f"  → Phrase fingerprints use Morton encoding: {use_morton}")
    
    # Build the index‑to‑coordinate lookup table ONCE for all documents.
    # This table is now essential for correct 2D back‑projection.
    index_to_xy_table = build_index_to_xy_table(grid_size, use_morton)
    logger.debug(f"  → Built index_to_xy table: shape={index_to_xy_table.shape}")

    # ── Load IDF weights (optional) ───────────────────────────────────────────
    idf_weights: Dict[str, float] = {}

    if idf_weights_path and idf_weights_path.exists():
        logger.info(f"Loading IDF weights from {idf_weights_path}...")
        with open(idf_weights_path, "r", encoding="utf-8") as fh:
            idf_weights = json.load(fh)
        logger.info(f"  → Loaded {len(idf_weights)} IDF weights")
    else:
        logger.warning("IDF weights not provided or not found, using TF‑only weighting")

    # ── Build document fingerprints ───────────────────────────────────────────
    logger.info("Building document fingerprints...")
    logger.info(f"  Phrase extraction: {'spaCy' if use_spacy else 'NLTK fallback'}")
    logger.info(f"  Remove verbs:      {remove_verbs}")
    logger.info(f"  Filter generic:    {filter_generic}")
    logger.info(f"  Min word length:   {min_word_length}")
    logger.info("-" * 70)

    doc_index_map: Dict[str, int] = {}
    fp_list: List[csr_matrix] = []

    skipped = 0
    processed = 0
    
    for idx, (doc_id, doc_text) in tqdm(enumerate(corpus.items())):
        if (idx + 1) % 10 == 0 or idx == 0:
            progress_pct = 100.0 * (idx + 1) / n_docs
            logger.info(f"Processing document {idx + 1}/{n_docs} ({progress_pct:.1f}%): {doc_id}")

        # Build 2D semantic grid (now correctly back‑projected)
        grid_2d = build_document_fingerprint_2d(
            doc_text            = doc_text,
            phrase_fingerprints = phrase_fingerprints,
            phrase_to_row       = phrase_to_row,
            idf_weights         = idf_weights,
            grid_size           = grid_size,
            use_morton          = use_morton,           # NEW
            index_to_xy         = index_to_xy_table,    # NEW
            use_spacy           = use_spacy,
            remove_verbs        = remove_verbs,
            filter_generic      = filter_generic,
            min_word_length     = min_word_length,
        )

        if grid_2d is None:
            logger.warning(f"  ⚠ Skipping document {doc_id}: no phrases matched")
            skipped += 1
            continue

        # Sparsify using topology‑preserving peak detection
        fp_sparse = sparsify_to_sdr_topological(
            grid_2d           = grid_2d,
            top_percent       = top_percent,
            grid_size         = grid_size,
            min_peak_distance = min_peak_distance,
            smoothing_sigma   = smoothing_sigma,
        )

        # Optional normalization
        if normalize:
            fp_sparse = normalize_fingerprint(fp_sparse, method=normalize_method)

        doc_index_map[doc_id] = len(fp_list)
        fp_list.append(fp_sparse)
        processed += 1

    logger.info("-" * 70)
    logger.info(f"Processed: {processed}/{n_docs} documents")
    logger.info(f"Skipped:   {skipped} documents (no phrase matches)")

    if processed == 0:
        logger.error("No documents were successfully fingerprinted")
        sys.exit(1) ###

    # ── Stack into dense matrix ───────────────────────────────────────────────
    logger.info("Stacking fingerprints into dense matrix...")
    fp_matrix = np.vstack([fp.toarray() for fp in fp_list]).astype(np.float32)
    logger.info(f"  → Final shape: {fp_matrix.shape}")

    # ── Compute statistics ────────────────────────────────────────────────────
    logger.info("Computing statistics...")

    sparsity_per_doc = [np.count_nonzero(row) / total_bits for row in fp_matrix]
    avg_sparsity = np.mean(sparsity_per_doc)
    std_sparsity = np.std(sparsity_per_doc)

    logger.info(f"  Average sparsity: {avg_sparsity*100:.2f}% ± {std_sparsity*100:.2f}%")
    logger.info(f"  Target sparsity:  {top_percent*100:.1f}%")

    stats = {
        "n_documents": processed,
        "n_skipped": skipped,
        "grid_size": grid_size,
        "total_bits": total_bits,
        "top_percent": top_percent,
        "target_active_bits": target_active,
        "avg_sparsity": float(avg_sparsity),
        "std_sparsity": float(std_sparsity),
        "normalize": normalize,
        "normalize_method": normalize_method if normalize else None,
        "min_peak_distance": min_peak_distance,
        "smoothing_sigma": smoothing_sigma,
        "use_spacy": use_spacy,
        "remove_verbs": remove_verbs,
        "filter_generic": filter_generic,
        "min_word_length": min_word_length,
        "use_morton": use_morton,                                     # track in stats
    }

    # ── Optional diversity computation ────────────────────────────────────────
    if compute_diversity and processed > 1:
        logger.info(f"Computing fingerprint diversity (sample size: {diversity_sample})...")

        sample_size = min(diversity_sample, processed)
        sample_indices = np.random.choice(processed, size=sample_size, replace=False)
        sample_fps = fp_matrix[sample_indices]

        diversity_metrics = compute_fingerprint_diversity(sample_fps)

        logger.info(f"  Avg pairwise overlap: {diversity_metrics['avg_overlap']*100:.2f}%")
        logger.info(f"  Avg Jaccard distance: {diversity_metrics['avg_jaccard_dist']:.4f}")
        logger.info(f"  Avg cosine distance:  {diversity_metrics['avg_cosine_dist']:.4f}")

        stats["diversity"] = diversity_metrics

    logger.info("=" * 70)

    return fp_matrix, doc_index_map, stats

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Build customtext-level SDRs from phrase fingerprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus JSON file (doc_id → text)",
    )
    
    parser.add_argument(
        "--fingerprints",
        type=Path,
        required=True,
        help="Directory containing phrase_fingerprints.npz and metadata",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for customtext fingerprints",
    )
    
    parser.add_argument(
        "--idf-weights",
        type=Path,
        default=None,
        help="Path to idf_weights.json (optional, from Step 2)",
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="SDR grid side length (default: 16 → 256 bits)",
    )
    
    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.05,
        help="Fraction of bits to activate (default: 0.1 = 10%%)",
    )
    
    parser.add_argument(
        "--normalize-method",
        type=str,
        default="l2",
        choices=["l1", "l2", "max"],
        help="Normalization method (default: l2)",
    )
    
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip fingerprint normalization",
    )
    
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=3,
        help="Minimum character length for tokens (default: 3)",
    )
    
    parser.add_argument(
        "--keep-verbs",
        dest="keep_verbs",
        default=True,
        action="store_true",
        help="Keep verb tokens during phrase extraction",
    )
    
    parser.add_argument(
        "--no-filter-generic",
        action="store_true",
        help="Keep generic/stopword-heavy phrases",
    )
    
    parser.add_argument(
        "--compute-diversity",
        action="store_true",
        help="Compute pairwise diversity metrics after building",
    )
    
    parser.add_argument(
        "--diversity-sample",
        type=int,
        default=100,
        help="Number of documents to sample for diversity computation (default: 100)",
    )
    
    parser.add_argument(
        "--min-peak-distance",
        type=int,
        default=2,
        help="Minimum distance between semantic hotspots in grid cells (default: 2)",
    )
    
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=0.5,
        help="Gaussian smoothing sigma before peak detection (default: 0.5, set 0 to disable)",
    )

    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument(
        "--morton",
        dest="morton_encoded",
        action="store_true",
        default=True,
        help="Phrase fingerprints are in Morton (Z‑order) encoding (default: True)",
    )
    exclusive_group.add_argument(
        "--no-morton",
        action="store_false",
        dest="morton_encoded",
        help="Use row‑major order for phrase fingerprints",
    )

    args = parser.parse_args()

    # Build fingerprints
    fp_matrix, doc_index_map, stats = build_customtext_fingerprints(
        corpus_path       = args.corpus,
        fingerprints_path = args.fingerprints,
        idf_weights_path  = args.idf_weights,
        grid_size         = args.grid_size,
        top_percent       = args.top_percent,
        normalize         = not args.no_normalize,
        normalize_method  = args.normalize_method,
        use_spacy         = SPACY_AVAILABLE,
        remove_verbs      = not args.keep_verbs,
        filter_generic    = not args.no_filter_generic,
        min_word_length   = args.min_word_length,
        compute_diversity = args.compute_diversity,
        diversity_sample  = args.diversity_sample,
        min_peak_distance = args.min_peak_distance,
        smoothing_sigma   = args.smoothing_sigma,
        morton_override   =  args.morton_encoded
    )
    
    # Write outputs
    write_outputs(
        fingerprints  = fp_matrix,
        doc_index_map = doc_index_map,
        stats         = stats,
        output_dir    = args.output,
        use_morton    = args.morton_encoded,
        grid_size     = args.grid_size 
    )
    
    logger.info("✓ Step 6 complete")


if __name__ == "__main__":
    main()
