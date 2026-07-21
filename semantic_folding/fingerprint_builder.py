#!/usr/bin/env python3
"""
fingerprint_builder.py — Shared fingerprint construction logic for Steps 5 and 6.

Aggregates phrase-level sparse fingerprints (Step 4) into document-level
Sparse Distributed Representations (SDRs) using TF-IDF weighted union,
then sparsifies via topology-preserving peak detection on 2D semantic grids.

Usage (indirect)
----------------
Called from doc_fingerprints.py (Step 5) and customtext_fingerprints.py (Step 6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.ndimage import maximum_filter, gaussian_filter
from tqdm import tqdm

from phrase_extractor import (
    extract_raw_phrases_spacy,
    extract_raw_phrases_fallback,
    SPACY_AVAILABLE,
)

if SPACY_AVAILABLE:
    from phrase_extractor import nlp

from lib import (
    xy_to_morton,
    xy_to_morton_vectorized,
    morton_to_xy,
    compute_fingerprint_diversity,
    expand_phrases,
    normalize_phrase,
    is_valid_phrase_structure,
    load_contexts_dict,
    normalize_fingerprint,
    sparsify_fingerprint,
    normalize_hyphens,
    extract_raw_phrases_ar_fa,
    split_arabic_english,
    normalize_arabic_phrase,
    get_logger,
)

from hazm import Normalizer, word_tokenize as hazm_word_tokenize

normalizer = Normalizer()

import re
_ARABIC_SCRIPT = re.compile(r'[\u0600-\u06FF]')

logger = get_logger("fingerprint_builder")


# NLTK only needed when spaCy is unavailable
if not SPACY_AVAILABLE:
    logger.debug("Importing NLTK fallback tokenizer and POS tagger")
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
    from nltk import pos_tag


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_outputs(
    fingerprints  : np.ndarray,
    doc_index_map : Dict[str, int],
    stats         : dict,
    output_dir    : Path,
    use_morton    : bool,
    grid_size     : int,
    file_prefix   : str = "doc",
    doc_norms     : Optional[np.ndarray] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path   = output_dir / f"{file_prefix}_fingerprints.npz"
    meta_path  = output_dir / f"{file_prefix}_fingerprints_meta.json"
    stats_path = output_dir / f"{file_prefix}_fingerprints_stats.json"

    # Save fingerprints and optional precomputed norms
    save_dict = {"fingerprints": fingerprints}
    if doc_norms is not None:
        save_dict["doc_norms"] = doc_norms
    np.savez_compressed(str(npz_path), **save_dict)
    logger.info(f"Fingerprint matrix written -> {npz_path}  shape={fingerprints.shape}")

    meta_dict = {
        "doc_to_row": doc_index_map,
        "use_morton": use_morton,
        "grid_size": grid_size,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_dict, fh, ensure_ascii=False, indent=2)
    logger.info(
        f"Metadata written -> {meta_path}  ({len(doc_index_map)} docs, "
        f"morton={use_morton}, grid={grid_size})"
    )

    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    logger.info(f"Run statistics written -> {stats_path}")


# ---------------------------------------------------------------------------
# Per-document phrase extractor
# ---------------------------------------------------------------------------

def extract_phrases_from_doc(
    text            : str,
    phrase_vocab    : Set[str],
    use_spacy       : bool = True,
    remove_verbs    : bool = False,
    filter_generic  : bool = True,
    min_word_length : int  = 3,
) -> List[str]:
    arabic_text, english_text = split_arabic_english(text)
    ar_valid = set()
    if arabic_text:
        ar_row = extract_raw_phrases_ar_fa(arabic_text)
        logger.debug(f"for {arabic_text} | {len(ar_row)} Ar raw phrases extracted")
        if not ar_row:
            logger.debug(f"for {arabic_text} | no Ar raw phrases — checking English only")
        for p in ar_row:
            norm = normalize_arabic_phrase(p)
            if norm:
                ar_valid.add(norm)
                logger.debug(f"[AR][KEEP] '{p}' -> '{norm}'")

    en_valid = set()
    candidates: Set[str] = set()
    if english_text:
        english_clean = normalize_hyphens(english_text)
        english_clean_lower = normalize_hyphens(english_text.lower())
        if english_clean != english_text:
            logger.debug(
                f"for {english_clean} | hyphen normalization applied: "
                f"'{english_text[:60]}' -> '{english_clean[:60]}'"
            )
        if use_spacy and SPACY_AVAILABLE:
            doc = nlp(english_clean)
            en_raw = extract_raw_phrases_spacy(doc)
        else:
            en_raw = extract_raw_phrases_fallback(english_clean_lower)
        logger.debug(f"for {english_clean} | {len(en_raw)} En raw phrases extracted")
        if not en_raw:
            logger.debug(f"[CORPUS] Line {english_clean} | no En raw phrases — checking Arabic only")
            en_raw = set()
        en_valid = expand_phrases(
            list(en_raw),
            context_text=english_clean,
            remove_verbs=remove_verbs,
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

    matched: List[str] = [p for p in valid_sub_phrases if p in phrase_vocab]
    oov: List[str] = [p for p in valid_sub_phrases if p not in phrase_vocab]

    if oov:
        logger.debug(f"  [OOV] {len(oov)} phrases not in vocab: {oov}")
    if matched:
        logger.debug(f"  [MATCHED] {matched}")

    logger.info(
        f"Query phrase extraction: {len(ar_valid)} arabic raw + {len(en_valid)} english raw -> "
        f"{len(valid_sub_phrases)} normalization and expanded -> "
        f"{len(matched)} vocab hits"
    )

    if not matched:
        logger.debug(f"No vocabulary matches in text snippet: {text[:80]!r}...")

    return matched


# ---------------------------------------------------------------------------
# Single-document fingerprint builder (2D topology-preserving)
# ---------------------------------------------------------------------------

def build_index_to_xy_table(grid_size: int, use_morton: bool = True) -> np.ndarray:
    total = grid_size * grid_size
    table = np.zeros((total, 2), dtype=np.int32)
    if use_morton:
        for idx in range(total):
            x, y = morton_to_xy(idx, grid_size)
            if 0 <= x < grid_size and 0 <= y < grid_size:
                table[idx] = (y, x)
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
    use_morton          : bool,
    index_to_xy         : np.ndarray,
    use_spacy           : bool = True,
    remove_verbs        : bool = True,
    filter_generic      : bool = True,
    min_word_length     : int  = 3,
) -> Optional[np.ndarray]:
    logger.debug(f"Building 2D fingerprint for document (length={len(doc_text)} chars)")

    grid_2d = np.zeros((grid_size, grid_size), dtype=np.float32)

    matched_phrases = extract_phrases_from_doc(
        text=doc_text,
        phrase_vocab=set(phrase_to_row.keys()),
        use_spacy=use_spacy,
        remove_verbs=remove_verbs,
        filter_generic=filter_generic,
        min_word_length=min_word_length,
    )

    if not matched_phrases:
        logger.debug("  -> No matched phrases, returning None")
        return None

    logger.debug(f"  -> Matched {len(matched_phrases)} phrases (with duplicates)")

    tf: Dict[str, int] = {}
    for phrase in matched_phrases:
        tf[phrase] = tf.get(phrase, 0) + 1

    logger.debug(f"  -> Unique phrases: {len(tf)}")

    hits = 0
    total_weight = 0.0

    for phrase, term_freq in tf.items():
        if phrase not in phrase_to_row:
            logger.debug(f"    Warning: Phrase '{phrase}' not in phrase_to_row (should not happen)")
            continue

        row_index = phrase_to_row[phrase]
        vec_1d = phrase_fingerprints[row_index]
        idf = idf_weights.get(phrase, 1.0)
        weight = term_freq * idf
        total_weight += weight

        # Vectorized scatter: only process non-zero elements
        nz_mask = vec_1d != 0
        if np.any(nz_mask):
            nz_indices = np.where(nz_mask)[0]
            nz_coords = index_to_xy[nz_indices]  # shape: (nnz, 2)
            nz_values = weight * vec_1d[nz_indices]
            np.add.at(grid_2d, (nz_coords[:, 0], nz_coords[:, 1]), nz_values)

        logger.debug(
            f"    + '{phrase[:40]}': TF={term_freq}, IDF={idf:.3f}, "
            f"weight={weight:.3f}, nnz_1d={np.count_nonzero(vec_1d)}"
        )
        hits += 1

    if hits == 0:
        logger.debug("  -> No phrase fingerprints accumulated (hits=0)")
        return None

    nnz_2d = np.count_nonzero(grid_2d)
    logger.debug(
        f"  -> Accumulated {hits} phrases, total_weight={total_weight:.2f}, "
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
    logger.debug("  [Fallback] Using global top-K sparsification")

    flat = grid_2d.flatten()
    top_k = max(1, int(round(top_percent * len(flat))))

    if len(flat) >= top_k:
        threshold = np.partition(flat, -top_k)[-top_k]
    else:
        threshold = 0

    flat[flat < threshold] = 0

    logger.debug(f"    -> top_k={top_k}, threshold={threshold:.4f}, nnz={np.count_nonzero(flat)}")

    return csr_matrix(flat.reshape(1, -1))


# ---------------------------------------------------------------------------
# Topology-preserving SDR sparsifier
# ---------------------------------------------------------------------------

def sparsify_to_sdr_topological(
    grid_2d           : np.ndarray,
    top_percent       : float,
    grid_size         : int,
    min_peak_distance : int   = 2,
    smoothing_sigma   : float = 1.5,
    contrast_boost    : bool  = True,
) -> csr_matrix:
    logger.debug(
        f"Sparsifying 2D grid: size={grid_size}, top_percent={top_percent:.3f}, min_peak_dist={min_peak_distance}, sigma={smoothing_sigma:.2f}"
    )

    if contrast_boost:
        mean_val = grid_2d.mean()
        grid_2d = np.maximum(grid_2d - 0.5 * mean_val, 0.0)

    if smoothing_sigma > 0:
        smoothed = gaussian_filter(grid_2d, sigma=smoothing_sigma)
        logger.debug(f"  -> Applied Gaussian smoothing (sigma={smoothing_sigma:.2f})")
    else:
        smoothed = grid_2d.copy()
        logger.debug("  -> Skipped smoothing (sigma=0)")

    neighborhood_size = 2 * min_peak_distance + 1
    local_max = maximum_filter(smoothed, size=neighborhood_size)
    peaks = (smoothed == local_max) & (smoothed > 0)

    peak_coords = np.argwhere(peaks)
    peak_values = smoothed[peaks]

    logger.debug(
        f"  -> Detected {len(peak_coords)} peaks (neighborhood_size={neighborhood_size})"
    )

    if len(peak_coords) == 0:
        logger.warning("  Warning: No peaks detected, falling back to global top-K")
        return _fallback_global_topk(grid_2d, top_percent, grid_size)

    total_bits = int(round(top_percent * grid_size * grid_size))

    sorted_indices = np.argsort(-peak_values)
    peak_coords = peak_coords[sorted_indices]
    peak_values = peak_values[sorted_indices]

    logger.debug(f"  -> Total bits budget: {total_bits}")
    logger.debug(f"  -> Peak strengths (top 5): {peak_values[:5]}")

    peak_weights = peak_values / peak_values.sum()

    bits_per_peak = np.maximum(1, (peak_weights * total_bits).astype(int))

    logger.debug(f"  -> Initial bit allocation: {bits_per_peak[:5]}")

    adjustment_iterations = 0
    while bits_per_peak.sum() > total_bits:
        max_idx = np.argmax(bits_per_peak)
        bits_per_peak[max_idx] -= 1
        adjustment_iterations += 1

    while bits_per_peak.sum() < total_bits:
        bits_per_peak[0] += 1
        adjustment_iterations += 1

    if adjustment_iterations > 0:
        logger.debug(f"  -> Adjusted bit allocation ({adjustment_iterations} iterations)")

    logger.debug(f"  -> Final bit allocation: {bits_per_peak[:5]} (sum={bits_per_peak.sum()})")

    result_2d = np.zeros_like(grid_2d)

    for peak_idx, ((y, x), n_bits) in enumerate(zip(peak_coords, bits_per_peak)):
        radius = max(2, int(np.sqrt(n_bits)))
        y_min, y_max = max(0, y - radius), min(grid_size, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(grid_size, x + radius + 1)

        window = grid_2d[y_min:y_max, x_min:x_max].copy()

        logger.debug(
            f"    Peak {peak_idx} at ({y},{x}): strength={peak_values[peak_idx]:.3f}, bits={n_bits}, radius={radius}, window_shape={window.shape}"
        )

        flat_window = window.flatten()
        if len(flat_window) > n_bits:
            threshold = np.partition(flat_window, -n_bits)[-n_bits]
            window[window < threshold] = 0
            logger.debug(f"      -> Applied threshold={threshold:.4f}, kept {np.count_nonzero(window)}/{len(flat_window)} cells")
        else:
            logger.debug(f"      -> Window smaller than bit budget, kept all {len(flat_window)} cells")

        result_2d[y_min:y_max, x_min:x_max] += window

    final_nnz_2d = np.count_nonzero(result_2d)
    logger.debug(
        f"  -> Activated {final_nnz_2d} cells in 2D (target was {total_bits}, {100.0 * final_nnz_2d / total_bits if total_bits > 0 else 0:.1f}% match)"
    )

    result_1d = np.zeros(grid_size * grid_size, dtype=np.float32)

    # Vectorized Morton encoding
    nz_y, nz_x = np.nonzero(result_2d)
    if nz_y.size > 0:
        morton_indices = xy_to_morton_vectorized(nz_x, nz_y)
        result_1d[morton_indices] = result_2d[nz_y, nz_x]

    final_nnz_1d = np.count_nonzero(result_1d)
    logger.debug(f"  -> Flattened to 1D (Morton): nnz={final_nnz_1d}")

    return csr_matrix(result_1d.reshape(1, -1))


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------

def build_fingerprints(
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
    morton_override   : bool  = True,
    step_label        : str   = "5",
    file_prefix       : str   = "doc",
) -> Tuple[np.ndarray, Dict[str, int], dict, np.ndarray]:
    step_name = f"Step {step_label}"
    total_bits = grid_size * grid_size
    target_active = int(total_bits * top_percent)

    logger.info("=" * 70)
    logger.info(f"{step_name}: Building Fingerprints (Topology-Preserving)")
    logger.info("=" * 70)
    logger.info(f"Corpus:       {corpus_path}")
    logger.info(f"Fingerprints: {fingerprints_path}")
    logger.info(f"IDF weights:  {idf_weights_path or '(none, using TF only)'}")
    logger.info(f"Grid size:    {grid_size} x {grid_size} = {total_bits} bits")
    logger.info(f"Top percent:  {top_percent*100:.1f}% -> ~{target_active} active bits")
    logger.info(f"Peak distance: {min_peak_distance} cells")
    logger.info(f"Smoothing s:   {smoothing_sigma:.2f}")
    logger.info("-" * 70)

    logger.info(f"Loading corpus from {corpus_path}...")
    corpus = load_contexts_dict(corpus_path)

    n_docs = len(corpus)
    logger.info(f"  -> Loaded {n_docs} documents")

    if n_docs == 0:
        logger.error("Corpus is empty, cannot proceed")
        sys.exit(1)

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
    logger.info(f"  -> Loaded phrase fingerprints: shape={phrase_fingerprints.shape}")

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    if "phrase_to_row" in meta:
        phrase_to_row = meta["phrase_to_row"]
        logger.info(f"  -> Loaded phrase_to_row mapping (nested format): {len(phrase_to_row)} phrases")
    else:
        phrase_to_row = meta
        logger.info(f"  -> Loaded phrase_to_row mapping (flat format): {len(phrase_to_row)} phrases")

    if "use_morton" in meta:
        use_morton = meta["use_morton"]
        logger.info(f"  -> Morton encoding flag from metadata: {use_morton}")
    else:
        use_morton = morton_override
        logger.info(f"  -> Morton encoding flag not in metadata; using override: {use_morton}")
    logger.info(f"  -> Final use_morton: {use_morton}")

    logger.info(f"  -> Phrase fingerprints use Morton encoding: {use_morton}")

    index_to_xy_table = build_index_to_xy_table(grid_size, use_morton)
    logger.debug(f"  -> Built index_to_xy table: shape={index_to_xy_table.shape}")

    idf_weights: Dict[str, float] = {}

    if idf_weights_path and idf_weights_path.exists():
        logger.info(f"Loading IDF weights from {idf_weights_path}...")
        with open(idf_weights_path, "r", encoding="utf-8") as fh:
            idf_weights = json.load(fh)
        logger.info(f"  -> Loaded {len(idf_weights)} IDF weights")
    else:
        logger.warning("IDF weights not provided or not found, using TF-only weighting")

    logger.info("Building fingerprints...")
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

        grid_2d = build_document_fingerprint_2d(
            doc_text=doc_text,
            phrase_fingerprints=phrase_fingerprints,
            phrase_to_row=phrase_to_row,
            idf_weights=idf_weights,
            grid_size=grid_size,
            use_morton=use_morton,
            index_to_xy=index_to_xy_table,
            use_spacy=use_spacy,
            remove_verbs=remove_verbs,
            filter_generic=filter_generic,
            min_word_length=min_word_length,
        )

        if grid_2d is None:
            logger.warning(f"  Skipping document {doc_id}: no phrases matched")
            skipped += 1
            continue

        fp_sparse = sparsify_to_sdr_topological(
            grid_2d=grid_2d,
            top_percent=top_percent,
            grid_size=grid_size,
            min_peak_distance=min_peak_distance,
            smoothing_sigma=smoothing_sigma,
        )

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
        sys.exit(1)

    logger.info("Stacking fingerprints into dense matrix...")
    fp_matrix = np.vstack([fp.toarray() for fp in fp_list]).astype(np.float32)
    logger.info(f"  -> Final shape: {fp_matrix.shape}")

    logger.info("Precomputing document L2 norms...")
    doc_norms = np.sqrt(np.sum(fp_matrix ** 2, axis=1)).astype(np.float32)
    logger.info(f"  -> Computed norms for {len(doc_norms)} documents")

    logger.info("Computing statistics...")

    sparsity_per_doc = [np.count_nonzero(row) / total_bits for row in fp_matrix]
    avg_sparsity = np.mean(sparsity_per_doc)
    std_sparsity = np.std(sparsity_per_doc)

    logger.info(f"  Average sparsity: {avg_sparsity*100:.2f}% +/- {std_sparsity*100:.2f}%")
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
        "use_morton": use_morton,
    }

    if compute_diversity and processed > 1:
        logger.info(f"Computing fingerprint diversity (sample size: {diversity_sample})...")

        sample_size = min(diversity_sample, processed)
        sample_indices = np.random.choice(processed, size=sample_size, replace=False)
        sample_fps = fp_matrix[sample_indices]

        diversity_metrics = compute_fingerprint_diversity(sample_fps)

        logger.info(f"  Avg similarity: {diversity_metrics['avg_similarity']*100:.2f}%")
        logger.info(f"  Diversity score: {diversity_metrics['diversity_score']:.4f}")
        logger.info(f"  Num samples: {diversity_metrics['num_samples']}")

        stats["diversity"] = diversity_metrics

    logger.info("=" * 70)

    return fp_matrix, doc_index_map, stats, doc_norms
