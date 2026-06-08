#!/usr/bin/env python3
r"""
Term-Context Matrix Builder (Architectural Bypass Edition)
==========================================================

Pipeline step: **term-context-matrix**

Constructs a sparse term-context co-occurrence matrix from the pre-validated
vocabulary and context mapping generated in Step 1.

By leveraging the `phrase_to_contexts.json` bipartite graph, this module
bypasses the $\mathcal{O}(C \times V)$ text-matching bottleneck entirely,
operating in pure $\mathcal{O}(N)$ time (where $N$ is the number of mapped
phrase-context pairs).

Output directory layout
-----------------------
    <output_dir>/
    ├── term_context_matrix.npz      ← scipy sparse matrix (Phrases × Contexts)
    ├── term_context_matrix.json     ← metadata / vocab / context IDs / integer maps
    └── idf_weights.json             ← per-phrase IDF floats (if TF-IDF enabled)

Log level
---------
Controlled via the ``LOG_LEVEL`` environment variable (default: ``INFO``).
Set ``LOG_LEVEL=DEBUG`` for per-phrase and per-cell trace output.

    export LOG_LEVEL=DEBUG
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from lib import get_logger
logger = get_logger("term_context")
# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Read log level from environment; fall back to INFO for production runs.
# Valid values: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
_LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

logger.remove()  # drop the default stderr sink
logger.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)
logger.debug(f"[LOGGING] level set to {_LOG_LEVEL!r} (from LOG_LEVEL env var)")

# ---------------------------------------------------------------------------
# Scipy Import
# ---------------------------------------------------------------------------
try:
    import scipy.sparse
    SCIPY_AVAILABLE = True
    logger.debug("[IMPORT] scipy.sparse loaded successfully")
except ImportError:
    logger.error(
        "scipy is required for sparse matrix operations. "
        "Install with: pip install scipy numpy"
    )
    SCIPY_AVAILABLE = False
    sys.exit(1)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_corpus_ids(corpus_path: Path) -> List[str]:
    """
    Scan the corpus CSV to extract the ordered list of context IDs.

    Only the ID column is read — raw text is intentionally ignored here.
    This establishes the column dimension of the matrix without loading
    potentially large text blobs into memory.

    Expected CSV format (no header assumed):
        <context_id>,<text>

    Lines that are blank or contain no comma are silently skipped (e.g. a
    header row that uses a non-comma separator, or trailing newlines).

    Parameters
    ----------
    corpus_path : Path
        Path to the raw corpus file.

    Returns
    -------
    List[str]
        Ordered list of context ID strings. Order determines column indices
        in the final matrix — do not sort or shuffle downstream.

    Notes
    -----
    - The first ``split(',', 1)`` ensures text fields containing commas are
      not mis-parsed as additional columns.
    - IDs are stripped of surrounding whitespace for robustness.
    """
    logger.debug(f"[LOAD CORPUS IDS] reading from {corpus_path}")
    context_ids: List[str] = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            # Skip blank lines and lines without a comma separator
            if not line.strip() or ',' not in line:
                logger.debug(f"  [SKIP LINE {lineno}] blank or no comma: {line.rstrip()!r}")
                continue

            ctx_id, _ = line.split(',', 1)
            ctx_id = ctx_id.strip()
            context_ids.append(ctx_id)
            logger.debug(f"  [CORPUS ID] line={lineno} id={ctx_id!r}")

    logger.info(f"[LOAD CORPUS IDS] {len(context_ids):,} context IDs loaded from {corpus_path.name}")
    return context_ids


def load_vocabulary(vocab_path: Path) -> List[Tuple[str, int]]:
    """
    Load the ordered vocabulary produced by Step 1.

    Expected CSV format (no header):
        <phrase>,<frequency>

    Rows with fewer than two columns are silently skipped (e.g. a trailing
    empty line). Frequency is cast to ``int``; non-integer values will raise
    a ``ValueError`` at parse time.

    Parameters
    ----------
    vocab_path : Path
        Path to ``vocabulary.csv``.

    Returns
    -------
    List[Tuple[str, int]]
        Ordered list of ``(phrase_string, total_frequency)`` tuples.
        Order determines row indices in the final matrix.

    Notes
    -----
    - Phrases are stripped of surrounding whitespace.
    - The list order is preserved from the CSV; do not re-sort downstream
      unless you also rebuild the index maps.
    """
    logger.debug(f"[LOAD VOCAB] reading from {vocab_path}")
    phrases: List[Tuple[str, int]] = []

    with open(vocab_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for rowno, row in enumerate(reader, start=1):
            # Skip malformed / short rows
            if len(row) < 2:
                logger.debug(f"  [SKIP ROW {rowno}] insufficient columns: {row}")
                continue

            phrase = row[0].strip()
            freq   = int(row[1])
            phrases.append((phrase, freq))
            logger.debug(f"  [VOCAB ROW {rowno}] phrase={phrase!r} freq={freq}")

    logger.info(f"[LOAD VOCAB] {len(phrases):,} phrases loaded from {vocab_path.name}")
    return phrases


# ---------------------------------------------------------------------------
# TF-IDF Normalization
# ---------------------------------------------------------------------------

def apply_tf_idf_normalization(
    matrix: "scipy.sparse.csr_matrix",
    num_contexts: int,
) -> Tuple["scipy.sparse.csr_matrix", np.ndarray]:
    r"""
    Apply TF-IDF weighting to a (Phrases × Contexts) sparse matrix.

    Formula
    -------
    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\!\left(\frac{N}{\text{DF}(t) + 1}\right)$$

    Where:
      - $N$       = total number of contexts (``num_contexts``)
      - $DF(t)$   = number of contexts in which term $t$ appears (row-wise nnz)
      - $+1$      = Laplace smoothing to avoid division by zero for unseen terms

    Implementation notes
    --------------------
    - The input matrix is binary (0/1), so TF is always 1 where non-zero.
      The formula therefore reduces to a pure IDF scaling.
    - ``np.diff(matrix.indptr)`` extracts per-row non-zero counts in $O(P)$
      time without materializing the dense matrix.
    - IDF scaling is applied via left-multiplication with a diagonal matrix
      (``scipy.sparse.diags``), which is $O(\text{nnz})$ and avoids any
      dense intermediate.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Binary occurrence matrix of shape ``(num_phrases, num_contexts)``.
    num_contexts : int
        Total number of documents/contexts in the corpus ($N$).

    Returns
    -------
    Tuple[scipy.sparse.csr_matrix, np.ndarray]
        - TF-IDF normalized sparse matrix (same shape, float32 values).
        - 1-D numpy array of IDF weights, one per phrase row.
    """
    logger.debug(
        f"[TF-IDF ENTER] matrix shape={matrix.shape} "
        f"nnz={matrix.nnz:,} num_contexts={num_contexts}"
    )

    # ── compute document frequency per phrase (row-wise nnz) ─────────────────
    # np.diff on the CSR indptr gives the count of non-zero entries per row,
    # which equals DF(t) for a binary matrix.
    df: np.ndarray = np.diff(matrix.indptr)
    logger.debug(f"[TF-IDF] DF range: min={df.min()} max={df.max()} mean={df.mean():.2f}")

    # ── smoothed IDF: log(N / (DF + 1)) ──────────────────────────────────────
    # +1 prevents log(0) for phrases that appear in every context (DF == N).
    idf: np.ndarray = np.log(num_contexts / (df + 1))
    logger.debug(f"[TF-IDF] IDF range: min={idf.min():.4f} max={idf.max():.4f}")

    # ── scale rows by IDF via diagonal left-multiplication ───────────────────
    # diag(idf) @ matrix scales row i by idf[i]; O(nnz), no dense intermediate.
    idf_diag = scipy.sparse.diags(idf, format="csr")
    normalized_matrix = idf_diag @ matrix
    logger.debug(
        f"[TF-IDF] normalized nnz={normalized_matrix.nnz:,} "
        f"(unchanged from binary — only values scaled)"
    )

    logger.info(
        f"[TF-IDF] applied: {matrix.nnz:,} entries scaled | "
        f"IDF range [{idf.min():.4f}, {idf.max():.4f}]"
    )
    return normalized_matrix, idf


# ---------------------------------------------------------------------------
# Core Matrix Builder (The Bypass)
# ---------------------------------------------------------------------------

def build_term_context_matrix(
    phrases: List[Tuple[str, int]],
    context_ids: List[str],
    phrase_mapping: Dict[str, List[str]],
    normalize_tfidf: bool = True,
) -> Tuple["scipy.sparse.csr_matrix", Optional[np.ndarray]]:
    r"""
    Construct the term-context matrix via $\mathcal{O}(1)$ dictionary lookups.

    Architecture
    ------------
    Traditional approaches scan every context for every vocabulary term,
    yielding $\mathcal{O}(C \times V)$ complexity. This function inverts the
    problem: the ``phrase_to_contexts.json`` bipartite graph from Step 1
    already encodes which phrases appear in which contexts. We simply iterate
    over that graph and set the corresponding matrix cells, giving
    $\mathcal{O}(N)$ where $N$ is the total number of phrase-context pairs.

    Processing pipeline
    -------------------
    1. Build reverse-lookup dicts:
         ``ctx_id  → column index``  (from ordered ``context_ids``)
         ``phrase  → row index``     (from ordered ``phrases``)
    2. Allocate a LIL sparse matrix — optimal for targeted cell insertions.
    3. Iterate over ``phrase_mapping``; for each (phrase, context_list) pair:
         a. Skip phrases absent from the vocabulary index (OOV guard).
         b. For each context ID, resolve to a column index and set cell = 1.0.
         c. Skip context IDs absent from the corpus index (stale mapping guard).
    4. Convert LIL → CSR for efficient arithmetic and serialization.
    5. Optionally apply TF-IDF normalization (see ``apply_tf_idf_normalization``).

    Parameters
    ----------
    phrases : List[Tuple[str, int]]
        Ordered vocabulary from ``load_vocabulary``.
    context_ids : List[str]
        Ordered corpus context IDs from ``load_corpus_ids``.
    phrase_mapping : Dict[str, List[str]]
        ``phrase_to_contexts.json`` mapping from Step 1.
    normalize_tfidf : bool, default=True
        Whether to apply TF-IDF weighting to the raw binary matrix.

    Returns
    -------
    Tuple[scipy.sparse.csr_matrix, Optional[np.ndarray]]
        - Populated sparse CSR matrix of shape ``(num_phrases, num_contexts)``.
        - IDF weight array (one float per phrase row), or ``None`` if TF-IDF
          was disabled.

    Notes
    -----
    - Phrases in ``phrase_mapping`` that are absent from ``phrases`` are silently
      skipped; they represent terms filtered out by Step 1's vocabulary pruning.
    - Context IDs in the mapping that are absent from ``context_ids`` are silently
      skipped; this guards against stale mappings after corpus edits.
    - The matrix is binary before TF-IDF: repeated occurrences of a phrase in
      the same context are collapsed to a single 1.0 entry.
    """
    num_phrases  = len(phrases)
    num_contexts = len(context_ids)

    logger.info(
        f"[BUILD MATRIX] dimensions: {num_phrases:,} phrases × {num_contexts:,} contexts"
    )
    logger.debug(
        f"[BUILD MATRIX] phrase_mapping contains {len(phrase_mapping):,} entries | "
        f"normalize_tfidf={normalize_tfidf}"
    )

    # ── step 1: build reverse-lookup dicts ───────────────────────────────────
    # O(C) and O(V) construction; O(1) lookup thereafter.
    ctx_id_to_idx:  Dict[str, int] = {cid: idx for idx, cid in enumerate(context_ids)}
    phrase_to_idx:  Dict[str, int] = {p[0]: idx for idx, p in enumerate(phrases)}
    logger.debug(
        f"[BUILD MATRIX] lookup tables built: "
        f"{len(ctx_id_to_idx):,} context slots, {len(phrase_to_idx):,} phrase slots"
    )

    # ── step 2: allocate LIL matrix ──────────────────────────────────────────
    # LIL (List of Lists) is the most efficient scipy format for incremental
    # cell-by-cell insertion. We convert to CSR after population.
    matrix = scipy.sparse.lil_matrix((num_phrases, num_contexts), dtype=np.float32)
    logger.debug(f"[BUILD MATRIX] LIL matrix allocated ({num_phrases} × {num_contexts})")

    # ── step 3: populate matrix from bipartite graph ─────────────────────────
    total_occurrences = 0
    skipped_oov       = 0   # phrases in mapping but not in vocabulary
    skipped_stale     = 0   # context IDs in mapping but not in corpus

    for phrase, mapped_contexts in phrase_mapping.items():

        # ── 3a: OOV guard — phrase not in vocabulary index ───────────────────
        # Happens when Step 1 pruned a phrase after the mapping was written,
        # or when the vocab and mapping files are from different runs.
        if phrase not in phrase_to_idx:
            logger.debug(f"  [OOV SKIP] '{phrase}' — not in vocabulary index")
            skipped_oov += 1
            continue

        row_idx = phrase_to_idx[phrase]
        logger.debug(
            f"  [PHRASE] '{phrase}' → row {row_idx} | "
            f"{len(mapped_contexts)} mapped context(s)"
        )

        for ctx_id in mapped_contexts:

            # ── 3b: stale mapping guard — context ID not in corpus ────────────
            # Happens when the corpus was edited after Step 1 ran.
            if ctx_id not in ctx_id_to_idx:
                logger.debug(f"    [STALE CTX SKIP] ctx_id={ctx_id!r} — not in corpus index")
                skipped_stale += 1
                continue

            col_idx = ctx_id_to_idx[ctx_id]
            # Binary: set to 1.0 regardless of how many times the phrase
            # appears in this context (TF-IDF handles weighting later).
            matrix[row_idx, col_idx] = 1.0
            total_occurrences += 1
            logger.debug(f"    [CELL SET] row={row_idx} col={col_idx} ctx={ctx_id!r}")

    logger.info(
        f"[BUILD MATRIX] population complete: "
        f"{total_occurrences:,} cells set | "
        f"{skipped_oov} OOV phrases skipped | "
        f"{skipped_stale} stale context IDs skipped"
    )

    # ── step 4: convert LIL → CSR ────────────────────────────────────────────
    # CSR is optimal for row-slicing, arithmetic, and scipy serialization.
    logger.debug("[BUILD MATRIX] converting LIL → CSR")
    matrix_csr = matrix.tocsr()
    logger.debug(
        f"[BUILD MATRIX] CSR shape={matrix_csr.shape} "
        f"nnz={matrix_csr.nnz:,} "
        f"density={matrix_csr.nnz / max(num_phrases * num_contexts, 1):.6f}"
    )

    # ── step 5: optional TF-IDF normalization ─────────────────────────────────
    idf_array: Optional[np.ndarray] = None
    if normalize_tfidf:
        logger.info("[BUILD MATRIX] applying TF-IDF normalization...")
        matrix_csr, idf_array = apply_tf_idf_normalization(matrix_csr, num_contexts)
    else:
        logger.debug("[BUILD MATRIX] TF-IDF disabled — raw binary matrix retained")

    return matrix_csr, idf_array


# ---------------------------------------------------------------------------
# Output Writer
# ---------------------------------------------------------------------------

def save_outputs(
    matrix: "scipy.sparse.csr_matrix",
    context_ids: List[str],
    phrases: List[Tuple[str, int]],
    phrase_mapping: Dict[str, List[str]],
    idf_array: Optional[np.ndarray],
    output_dir: Path,
) -> None:
    """
    Persist the (Phrases × Contexts) matrix and all associated metadata.

    Artifacts written
    -----------------
    1. ``term_context_matrix.npz``
       The scipy sparse matrix in compressed NPZ format. Loadable via
       ``scipy.sparse.load_npz()``.

    2. ``term_context_matrix.json``
       Metadata bundle consumed by downstream steps, notably Step 4
       (``phrase_fingerprints.py``). Contains:
         - ``num_phrases``, ``num_contexts``, ``nnz``, ``density``
         - ``matrix_shape``       : [num_phrases, num_contexts]
         - ``matrix_orientation`` : "phrases x contexts"
         - ``context_ids``        : ordered list of string IDs (column labels)
         - ``phrases``            : ordered list of phrase strings (row labels)
         - ``phrase_frequencies`` : parallel list of integer frequencies
         - ``phrase_contexts``    : Dict[phrase → List[int column indices]]
           *** This integer-index mapping is strictly required by Step 4. ***

    3. ``idf_weights.json``
       Dict[phrase → float IDF weight]. Written only when TF-IDF was applied.
       Useful for inspecting which phrases are most discriminative.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The populated (and optionally TF-IDF weighted) matrix.
    context_ids : List[str]
        Ordered corpus context IDs (column labels).
    phrases : List[Tuple[str, int]]
        Ordered vocabulary (row labels + frequencies).
    phrase_mapping : Dict[str, List[str]]
        Raw Step 1 mapping of ``phrase → List[context_id]`` (string IDs).
        Used here only to build the integer-index version for the JSON.
    idf_array : Optional[np.ndarray]
        IDF weights to persist; ``None`` if TF-IDF was disabled.
    output_dir : Path
        Target directory; created (with parents) if it does not exist.
    """
    logger.debug(f"[SAVE] output_dir={output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = output_dir / "term_context_matrix.npz"
    meta_path = output_dir / "term_context_matrix.json"
    idf_path  = output_dir / "idf_weights.json"

    # ── artifact 1: sparse matrix (.npz) ─────────────────────────────────────
    logger.debug(f"[SAVE] writing sparse matrix → {npz_path}")
    scipy.sparse.save_npz(npz_path, matrix)
    logger.success(
        f"Matrix written      → {npz_path}  "
        f"(shape={matrix.shape}, nnz={matrix.nnz:,})"
    )

    # ── artifact 2: metadata JSON ─────────────────────────────────────────────
    # Remap string context IDs → integer column indices for Step 4.
    # Step 4 works with numpy index arrays, not string IDs, so this conversion
    # must happen here rather than being deferred to the consumer.
    logger.debug("[SAVE] remapping phrase_mapping string IDs → integer column indices")
    ctx_id_to_idx: Dict[str, int] = {cid: idx for idx, cid in enumerate(context_ids)}
    numeric_phrase_contexts: Dict[str, List[int]] = {}

    for phrase_tuple in phrases:
        phrase = phrase_tuple[0]
        if phrase in phrase_mapping:
            int_indices = [
                ctx_id_to_idx[cid]
                for cid in phrase_mapping[phrase]
                if cid in ctx_id_to_idx
            ]
            numeric_phrase_contexts[phrase] = int_indices
            logger.debug(
                f"  [REMAP] '{phrase}' → {len(int_indices)} integer context indices"
            )
        else:
            # Phrase is in vocabulary but absent from mapping — write empty list
            # so Step 4 can still index the phrase without a KeyError.
            logger.debug(f"  [REMAP MISS] '{phrase}' — not in phrase_mapping, writing []")
            numeric_phrase_contexts[phrase] = []

    density = float(matrix.nnz / max(len(phrases) * len(context_ids), 1))
    metadata = {
        "num_phrases"        : len(phrases),
        "num_contexts"       : len(context_ids),
        "nnz"                : int(matrix.nnz),
        "density"            : density,
        "matrix_shape"       : list(matrix.shape),
        "matrix_orientation" : "phrases x contexts",
        "context_ids"        : context_ids,
        "phrases"            : [p[0] for p in phrases],
        "phrase_frequencies" : [p[1] for p in phrases],
        # Integer-index mapping — strictly required by Step 4
        "phrase_contexts"    : numeric_phrase_contexts,
    }
    logger.debug(
        f"[SAVE] metadata: {len(phrases)} phrases, {len(context_ids)} contexts, "
        f"density={density:.6f}"
    )

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.success(f"Metadata written    → {meta_path}")

    # ── artifact 3: IDF weights (conditional) ────────────────────────────────
    if idf_array is not None and len(idf_array) > 0:
        idf_dict = {
            phrase[0]: float(idf_val)
            for phrase, idf_val in zip(phrases, idf_array)
        }
        logger.debug(f"[SAVE] writing {len(idf_dict):,} IDF weights → {idf_path}")
        with open(idf_path, "w", encoding="utf-8") as fh:
            json.dump(idf_dict, fh, indent=2, ensure_ascii=False)
        logger.success(f"IDF weights written → {idf_path}")
    else:
        logger.debug("[SAVE] idf_array is None or empty — idf_weights.json not written")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            r"Build term-context matrix using the \mathcal{O}(1) Architectural Bypass."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vocab", required=True, type=Path,
        help="Path to vocabulary.csv generated by Step 1",
    )
    parser.add_argument(
        "--mapping", required=True, type=Path,
        help="Path to phrase_to_contexts.json generated by Step 1",
    )
    parser.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to corpus file (to establish Context ID order/columns)",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output DIRECTORY — all artefacts are written here",
    )
    parser.add_argument(
        "--no-tfidf", action="store_true",
        help="Disable TF-IDF normalization (raw binary occurrences are saved)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Term-Context Matrix Builder (Architectural Bypass)")
    logger.info(f"LOG_LEVEL={_LOG_LEVEL}")
    logger.info("=" * 60)

    # ── step 1: load artifacts ────────────────────────────────────────────────
    logger.info(f"Loading vocabulary    → {args.vocab}")
    phrases = load_vocabulary(args.vocab)
    logger.debug(f"[MAIN] {len(phrases):,} phrases loaded")

    logger.info(f"Loading corpus IDs    → {args.corpus}")
    context_ids = load_corpus_ids(args.corpus)
    logger.debug(f"[MAIN] {len(context_ids):,} context IDs loaded")

    logger.info(f"Loading context map   → {args.mapping}")
    with open(args.mapping, 'r', encoding='utf-8') as f:
        phrase_mapping = json.load(f)
    logger.debug(f"[MAIN] {len(phrase_mapping):,} phrase→context entries in mapping")

    # ── step 2: build matrix ──────────────────────────────────────────────────
    logger.info(f"Building matrix (TF-IDF={'disabled' if args.no_tfidf else 'enabled'})...")
    matrix, idf_array = build_term_context_matrix(
        phrases=phrases,
        context_ids=context_ids,
        phrase_mapping=phrase_mapping,
        normalize_tfidf=not args.no_tfidf,
    )
    logger.debug(
        f"[MAIN] matrix built: shape={matrix.shape} "
        f"nnz={matrix.nnz:,} idf={'yes' if idf_array is not None else 'no'}"
    )

    # ── step 3: save outputs ──────────────────────────────────────────────────
    logger.info(f"Saving outputs        → {args.output_dir}")
    save_outputs(
        matrix=matrix,
        context_ids=context_ids,
        phrases=phrases,
        phrase_mapping=phrase_mapping,
        idf_array=idf_array,
        output_dir=args.output_dir,
    )

    logger.success("=" * 60)
    logger.success("Matrix construction completed successfully.")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
