#!/usr/bin/env python3
"""
doc_fingerprints.py — Step 5 of the Semantic Folding Pipeline

Aggregates phrase-level sparse fingerprints (Step 4) into document-level
Sparse Distributed Representations (SDRs) using TF-IDF weighted union,
then sparsifies via topology-preserving peak detection on 2D semantic grids.

Pipeline position
-----------------
Step 1  phrase_extractor.py        -> vocabulary.csv
Step 2  term_context.py            -> term_context_matrix.*, idf_weights.json
Step 3  semantic_space.py          -> context_coordinates.json
Step 4  phrase_fingerprints.py     -> phrase_fingerprints/
Step 5  doc_fingerprints.py        -> doc_fingerprints/          <- THIS FILE
Step 6  customtext_fingerprints.py -> customtext_fingerprints/
Step 7  query_processing.py        -> query results

Usage
-----
    python doc_fingerprints.py \
        --corpus      data/corpus.txt \
        --fingerprints outputs/run/phrase_fingerprints \
        --idf-weights outputs/run/term_context/idf_weights.json \
        --output-dir  outputs/run/doc_fingerprints \
        --grid-size   16 \
        --top-percent 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lib import get_logger
from phrase_extractor import SPACY_AVAILABLE
from fingerprint_builder import build_fingerprints, write_outputs

logger = get_logger("doc_fingerprints")


def build_doc_fingerprints(*args, **kwargs):
    """Wrapper around build_fingerprints with Step 5 defaults."""
    return build_fingerprints(*args, step_label="5", file_prefix="doc", **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Build document-level SDRs from phrase fingerprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--corpus", type=Path, required=True, help="Path to corpus JSON file (doc_id -> text)")
    parser.add_argument("--fingerprints", type=Path, required=True, help="Directory containing phrase_fingerprints.npz and metadata")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for document fingerprints")
    parser.add_argument("--idf-weights", type=Path, default=None, help="Path to idf_weights.json (optional, from Step 2)")
    parser.add_argument("--grid-size", type=int, default=128, help="SDR grid side length")
    parser.add_argument("--top-percent", type=float, default=0.05, help="Fraction of bits to activate")
    parser.add_argument("--normalize-method", type=str, default="l2", choices=["l1", "l2", "max"], help="Normalization method")
    parser.add_argument("--no-normalize", action="store_true", help="Skip fingerprint normalization")
    parser.add_argument("--min-word-length", type=int, default=3, help="Minimum character length for tokens")
    parser.add_argument("--keep-verbs", dest="keep_verbs", default=True, action="store_true", help="Keep verb tokens during phrase extraction")
    parser.add_argument("--no-filter-generic", action="store_true", help="Keep generic/stopword-heavy phrases")
    parser.add_argument("--compute-diversity", action="store_true", help="Compute pairwise diversity metrics after building")
    parser.add_argument("--diversity-sample", type=int, default=100, help="Number of documents to sample for diversity computation")
    parser.add_argument("--min-peak-distance", type=int, default=2, help="Minimum distance between semantic hotspots in grid cells")
    parser.add_argument("--smoothing-sigma", type=float, default=1.5, help="Gaussian smoothing sigma before peak detection")

    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument("--morton", dest="morton_encoded", action="store_true", default=True, help="Phrase fingerprints are in Morton encoding")
    exclusive_group.add_argument("--no-morton", action="store_false", dest="morton_encoded", help="Use row-major order for phrase fingerprints")

    args = parser.parse_args()

    fp_matrix, doc_index_map, stats = build_doc_fingerprints(
        corpus_path=args.corpus,
        fingerprints_path=args.fingerprints,
        idf_weights_path=args.idf_weights,
        grid_size=args.grid_size,
        top_percent=args.top_percent,
        normalize=not args.no_normalize,
        normalize_method=args.normalize_method,
        use_spacy=SPACY_AVAILABLE,
        remove_verbs=not args.keep_verbs,
        filter_generic=not args.no_filter_generic,
        min_word_length=args.min_word_length,
        compute_diversity=args.compute_diversity,
        diversity_sample=args.diversity_sample,
        min_peak_distance=args.min_peak_distance,
        smoothing_sigma=args.smoothing_sigma,
        morton_override=args.morton_encoded,
    )

    write_outputs(
        fingerprints=fp_matrix,
        doc_index_map=doc_index_map,
        stats=stats,
        output_dir=args.output,
        use_morton=args.morton_encoded,
        grid_size=args.grid_size,
        file_prefix="doc",
    )

    logger.info("Check: Step 5 complete")


if __name__ == "__main__":
    main()
