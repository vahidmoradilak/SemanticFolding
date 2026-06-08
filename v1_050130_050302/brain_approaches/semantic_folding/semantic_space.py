#!/usr/bin/env python3
"""
semantic_space.py
=================

Pipeline step: **semantic-space**

Embeds corpus *contexts* into a 2-D integer grid so that semantically similar
contexts occupy neighbouring cells.  The resulting grid coordinates are the
primary input consumed by the ``phrase-fingerprints`` step to construct
fixed-length binary fingerprint vectors for every phrase.

Overview
--------
The script operates on the term-context matrix produced by an upstream
extraction step (e.g. ``term_context_matrix.npz`` + ``term_context_matrix.json``).
Each *column* of that matrix is a high-dimensional sparse vector representing
one context in phrase-space.  The pipeline is:

1.  **Load** the sparse matrix and metadata.
2.  **Transpose** so contexts become rows ``(num_contexts x num_phrases)``.
3.  **Normalise** context vectors to unit L2 length (optional).
4.  **Reduce** to 2-D continuous coordinates using t-SNE, UMAP, or PCA.
5.  **Scale** continuous coordinates onto a discrete ``grid_size x grid_size``
    integer grid.
6.  **Resolve collisions** — contexts that map to the same cell are displaced
    outward by spiral search.
7.  **Write outputs** — continuous CSV, discrete CSV, primary JSON lookup map,
    summary statistics, and optional visualisation PNG files.

Inputs
------
``term_context_matrix.npz``
    Compressed sparse matrix in CSR format with shape
    ``(num_phrases, num_contexts)``.  Must contain the arrays ``data``,
    ``indices``, ``indptr``, and ``shape``.

``term_context_matrix.json``
    Metadata file produced alongside the matrix.  Must contain at minimum:

    * ``"num_contexts"`` — integer count of context columns.
    * ``"num_phrases"``  — integer count of phrase rows.
    * ``"context_ids"``  — ordered list of context identifier strings whose
      position matches the column order of the matrix.

Outputs
-------
``context_coordinates_continuous.csv``
    Human-readable CSV of the raw floating-point 2-D coordinates produced by
    the dimensionality reduction step, before any grid quantisation.  Format::

        context_id,x,y
        context_0,0.382941,-1.204718
        context_1,2.019384, 0.774022

``context_coordinates.csv``
    Human-readable CSV of the final integer grid positions after quantisation
    and collision resolution.  For inspection only — **not** consumed by
    downstream automation::

        context_id,x,y
        context_0,5,12
        context_1,14,3

``context_coordinates.json``  *(primary machine-readable output)*
    JSON object mapping every ``context_id`` to its finalised integer grid
    position.  This file is the **only** coordinates file read by
    ``phrase_fingerprints.py`` because it enables ``O(1)`` dictionary
    lookup::

        {
            "context_0": {"x": 5,  "y": 12},
            "context_1": {"x": 14, "y": 3}
        }

``coordinate_statistics.json``
    JSON object containing run-level summary statistics.  All grid metrics
    (collision rate, unique positions) are computed **after** collision
    resolution is complete so the reported numbers reflect the true final
    state of the grid.

``semantic_space_{method}_continuous.png``  *(optional)*
    Scatter plot of the continuous 2-D embedding.  Generated only when
    ``--visualize`` is passed.

``semantic_space_{method}_grid.png``  *(optional)*
    Scatter plot of the discrete grid positions.  Generated only when
    ``--visualize`` is passed and ``--no-grid`` is not set.

Design Decisions
----------------
**JSON over CSV for downstream consumption**
    ``context_coordinates.json`` is the single authoritative coordinate source
    for all downstream automation.  The CSV files exist solely for human
    inspection and debugging.  This separation was introduced to provide
    ``O(1)`` context lookup instead of ``O(N)`` CSV scanning.

**Morton codes are NOT used here**
    Morton (Z-order) encoding was evaluated for grid construction but was
    superseded by spiral-search collision resolution, which preserves the
    spatial layout produced by the dimensionality reducer more faithfully.
    Morton encoding *is* used in ``phrase_fingerprints.py`` for a different
    purpose — linearising 2-D grid coordinates into a 1-D fingerprint index.

**Statistics after finalisation**
    All statistics (collision rate, unique positions) are computed after the
    spiral-search pass is complete so they accurately describe the outputs
    that downstream steps will consume.

**Stable output filenames**
    Output filenames are fixed strings (e.g. ``context_coordinates.json``)
    and never embed the method name or any runtime parameter.  Method-labelled
    names are used only for the optional visualisation PNGs.

Usage
-----
::

    # t-SNE (default) on a 64x64 grid
    python semantic_space.py \\
        --matrix   runs/run_001/term_context_matrix.npz \\
        --metadata runs/run_001/term_context_matrix.json \\
        --output   runs/run_001/ \\
        --method   tsne \\
        --grid-size 64

    # UMAP, skip collision resolution, produce visualisations
    python semantic_space.py \\
        --matrix   runs/run_001/term_context_matrix.npz \\
        --metadata runs/run_001/term_context_matrix.json \\
        --output   runs/run_001/ \\
        --method   umap \\
        --grid-size 128 \\
        --no-collision-resolution \\
        --visualize \\
        --show-density

    # PCA, continuous coordinates only (no grid)
    python semantic_space.py \\
        --matrix   runs/run_001/term_context_matrix.npz \\
        --metadata runs/run_001/term_context_matrix.json \\
        --output   runs/run_001/ \\
        --method   pca \\
        --no-grid

Exit Codes
----------
0   Success.
1   Input validation error (file not found, shape mismatch, import failure).
"""

import argparse
import json
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib import get_logger
logger = get_logger("semantic_space")


try:
    import numpy as np
except ImportError:
    logger.error("numpy is required.  Install with: pip install numpy")
    exit(1)

try:
    from scipy.sparse import csr_matrix, issparse
except ImportError:
    logger.error("scipy is required.  Install with: pip install scipy")
    exit(1)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load the term-context matrix metadata JSON produced by an upstream step.

    The metadata file is expected to be a flat JSON object.  The following
    keys are required by this script:

    * ``"num_contexts"`` — total number of context columns in the matrix.
    * ``"num_phrases"``  — total number of phrase rows in the matrix.
    * ``"context_ids"``  — ordered list of context identifier strings.  The
      position of each string in the list must correspond to the column index
      of that context in the sparse matrix.

    Additional keys (e.g. ``"created_at"``, ``"source_corpus"``) are allowed
    and are ignored.

    Parameters
    ----------
    metadata_path:
        Absolute or relative path to ``term_context_matrix.json``.

    Returns
    -------
    Dict[str, Any]
        The full parsed metadata dictionary.

    Raises
    ------
    FileNotFoundError
        If ``metadata_path`` does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError
        If any of the three required keys is absent.
    """
    logger.info(f"Loading metadata from: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    logger.success(
        f"Loaded metadata: {metadata['num_contexts']} contexts, "
        f"{metadata['num_phrases']} phrases"
    )
    return metadata


def load_sparse_matrix(matrix_path: Path) -> csr_matrix:
    """
    Load the sparse term-context matrix from a compressed NumPy archive.

    The ``.npz`` file must have been saved in CSR (Compressed Sparse Row)
    format and must contain the four arrays that define a CSR matrix:

    * ``"data"``    — non-zero values.
    * ``"indices"`` — column indices for each value in ``data``.
    * ``"indptr"``  — row pointer array.
    * ``"shape"``   — tuple ``(num_phrases, num_contexts)``.

    The matrix is oriented as **phrases × contexts** (rows are phrases,
    columns are contexts).  The transposition to **contexts × phrases** is
    performed in :func:`prepare_context_vectors`.

    Parameters
    ----------
    matrix_path:
        Absolute or relative path to ``term_context_matrix.npz``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape ``(num_phrases, num_contexts)``.

    Raises
    ------
    FileNotFoundError
        If ``matrix_path`` does not exist.
    KeyError
        If any of the four required arrays is missing from the archive.
    """
    logger.info(f"Loading matrix from: {matrix_path}")

    npz_data = np.load(matrix_path)
    matrix = csr_matrix(
        (npz_data["data"], npz_data["indices"], npz_data["indptr"]),
        shape=tuple(npz_data["shape"]),
    )

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100
    logger.success(
        f"Matrix shape: {matrix.shape} (phrases x contexts), "
        f"density: {density:.4f}%, nnz: {matrix.nnz}"
    )
    return matrix


# ---------------------------------------------------------------------------
# Vector Preparation
# ---------------------------------------------------------------------------

def prepare_context_vectors(
    matrix: csr_matrix,
    normalize: bool = True,
    keep_sparse: bool = False,
):
    """
    Prepare context feature vectors from the phrase-context matrix.

    Transposes the input matrix from ``(num_phrases, num_contexts)`` to
    ``(num_contexts, num_phrases)`` so that each *row* becomes the feature
    vector of one context expressed in phrase-space.  Optionally applies L2
    normalisation and optionally converts to a dense NumPy array.

    Parameters
    ----------
    matrix:
        Sparse matrix of shape ``(num_phrases, num_contexts)`` as returned by
        :func:`load_sparse_matrix`.
    normalize:
        If ``True`` (default), each context vector is scaled to unit L2 norm
        using ``sklearn.preprocessing.normalize``.  This ensures cosine
        similarity is equivalent to dot-product similarity, which is
        appropriate for UMAP with ``metric='cosine'`` and for t-SNE.
    keep_sparse:
        If ``True``, the matrix is kept in sparse format after normalisation.
        Useful when ``--use-sparse`` is passed and the method supports sparse
        input (UMAP, TruncatedSVD).  If ``False`` (default) the matrix is
        converted to a dense ``numpy.ndarray``.  A warning is emitted if the
        estimated dense size exceeds 1 GB.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix
        Context vectors of shape ``(num_contexts, num_phrases)``.
        Type is ``numpy.ndarray`` when ``keep_sparse=False``, or
        ``csr_matrix`` when ``keep_sparse=True``.

    Notes
    -----
    The dense memory estimate uses 8 bytes per element (float64).  If the
    actual dtype is float32 the true cost is half that, but the warning is
    intentionally conservative.
    """
    logger.info("Transposing matrix to get context vectors (contexts x phrases)...")
    context_matrix = matrix.T.tocsr()

    if normalize:
        from sklearn.preprocessing import normalize as sk_normalize

        logger.info("Normalizing context vectors (L2)...")
        context_matrix = sk_normalize(context_matrix, norm="l2", axis=1)

    if not keep_sparse:
        dense_gb = (
            context_matrix.shape[0] * context_matrix.shape[1] * 8
        ) / (1024 ** 3)
        if dense_gb > 1.0:
            logger.warning(
                f"Dense matrix would be ~{dense_gb:.1f} GB. "
                f"Consider passing --use-sparse for UMAP or PCA."
            )
        context_matrix = context_matrix.toarray()

    logger.success(
        f"Context vectors ready: "
        f"{context_matrix.shape[0]} contexts x {context_matrix.shape[1]} phrases"
    )
    return context_matrix


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------

def reduce_dimensions_tsne(
    vectors: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    n_jobs: int = 1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce context vectors to 2-D continuous coordinates using t-SNE.

    t-SNE (t-distributed Stochastic Neighbour Embedding) is the default
    reduction method.  It excels at revealing local cluster structure but
    does not preserve global distances.  For small corpora (< 500 contexts)
    t-SNE typically produces the most visually interpretable layouts.

    The ``perplexity`` parameter is automatically clamped to
    ``min(perplexity, max(5, n_samples // 3))`` to prevent sklearn from
    raising an error when the number of samples is small.

    Parameters
    ----------
    vectors:
        Dense float array of shape ``(num_contexts, num_phrases)``.
    perplexity:
        t-SNE perplexity, loosely interpretable as the number of effective
        nearest neighbours.  Typical range: 5–50.  Clamped automatically.
    n_iter:
        Maximum number of optimisation iterations.
    n_jobs:
        Number of parallel threads for nearest-neighbour search.
        ``-1`` uses all available cores.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(num_contexts, 2)`` containing the 2-D
        continuous coordinates.

    Notes
    -----
    ``sklearn.manifold.TSNE`` is used internally.  The KL divergence of the
    final embedding is logged at ``SUCCESS`` level as a quality indicator —
    lower values indicate a better-fitting embedding.
    """
    from sklearn.manifold import TSNE


    n_samples = vectors.shape[0]
    perplexity = min(perplexity, max(2, n_samples - 1))  # must be < n_samples
    logger.info(
        f"Running t-SNE: n_samples={n_samples}, "
        f"perplexity={perplexity}, n_iter={n_iter}"
    )

    if vectors.shape[1] > 100:
        logger.info("High dimensionality detected. Applying TruncatedSVD pre-reduction...")
        svd = TruncatedSVD(n_components=100, random_state=random_state)
        vectors = svd.fit_transform(vectors)
    elif issparse(vectors):
        vectors = vectors.toarray()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,       
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
    )
    coordinates = tsne.fit_transform(vectors)
    logger.success(f"t-SNE done.  KL divergence: {tsne.kl_divergence_:.4f}")
    return coordinates


def reduce_dimensions_umap(
    vectors: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    n_jobs: int = 1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce context vectors to 2-D continuous coordinates using UMAP.

    UMAP (Uniform Manifold Approximation and Projection) is faster than t-SNE
    for large corpora and better preserves global structure.  It is
    recommended when the number of contexts exceeds ~2 000.

    The ``n_neighbors`` parameter is automatically clamped to
    ``min(n_neighbors, max(2, n_samples // 2))`` to avoid errors when the
    dataset is small.

    Parameters
    ----------
    vectors:
        Dense or sparse float array of shape ``(num_contexts, num_phrases)``.
        Sparse input is supported when ``metric='cosine'`` and the ``umap``
        package version supports it.
    n_neighbors:
        Number of neighbours considered for manifold approximation.  Higher
        values emphasise global structure; lower values emphasise local
        clusters.  Clamped automatically.
    min_dist:
        Minimum distance between embedded points.  Smaller values allow
        tighter clusters; larger values produce a more uniform spread.
    metric:
        Distance metric for the high-dimensional neighbour graph.  Defaults
        to ``'cosine'``, which pairs well with L2-normalised vectors.
    n_jobs:
        Number of parallel threads.  ``-1`` uses all available cores.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(num_contexts, 2)`` containing the 2-D
        continuous coordinates.

    Raises
    ------
    ImportError
        If the ``umap-learn`` package is not installed.
    """
    import umap

    n_samples = vectors.shape[0]
    n_neighbors = min(n_neighbors, max(2, n_samples // 2))
    logger.info(
        f"Running UMAP: n_samples={n_samples}, n_neighbors={n_neighbors}, "
        f"min_dist={min_dist}, metric={metric}"
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=True,
    )
    coordinates = reducer.fit_transform(vectors)
    logger.success("UMAP completed.")
    return coordinates


def reduce_dimensions_pca(vectors: np.ndarray) -> np.ndarray:
    """
    Reduce context vectors to 2-D continuous coordinates using PCA.

    PCA (Principal Component Analysis) is the fastest of the three supported
    methods and is fully deterministic, making it suitable for debugging and
    for corpora where interpretable global variance is more important than
    local cluster separation.

    When ``vectors`` is a sparse matrix, ``sklearn.decomposition.TruncatedSVD``
    is used instead of full PCA because it avoids materialising the dense
    centred matrix.  The result is mathematically equivalent to PCA on
    non-centred data.

    Parameters
    ----------
    vectors:
        Dense or sparse float array of shape ``(num_contexts, num_phrases)``.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(num_contexts, 2)`` containing the 2-D
        continuous coordinates along the top two principal components.

    Notes
    -----
    The fraction of variance explained by each principal component is logged
    at ``SUCCESS`` level.  Very low values (e.g. < 5 %) indicate that the
    two-dimensional projection captures little of the true variance and that
    t-SNE or UMAP may produce a more meaningful layout.
    """
    from sklearn.decomposition import PCA, TruncatedSVD

    logger.info(f"Running PCA: n_samples={vectors.shape[0]}")

    if issparse(vectors):
        pca = TruncatedSVD(n_components=2, random_state=42)
    else:
        pca = PCA(n_components=2, random_state=42)

    coordinates = pca.fit_transform(vectors)
    explained = pca.explained_variance_ratio_
    logger.success(
        f"PCA done.  Variance explained: "
        f"PC1={explained[0]:.2%}, PC2={explained[1]:.2%}"
    )
    return coordinates


# ---------------------------------------------------------------------------
# Grid Mapping
# ---------------------------------------------------------------------------

def resolve_collisions(
    grid_coords: np.ndarray,
    grid_size: int,
    max_radius: int = 10,
) -> np.ndarray:
    """
    Displace contexts that share a grid cell by outward spiral search.

    After quantisation, multiple contexts may map to the same integer cell.
    This function iterates through all contexts in their original order and,
    for each context whose target cell is already occupied, searches outward
    in expanding square shells until it finds the nearest free cell within
    ``max_radius`` steps.

    Algorithm
    ~~~~~~~~~
    For each context ``idx`` at quantised position ``(x, y)``:

    * If ``(x, y)`` is unoccupied, claim it and continue.
    * Otherwise, iterate shells ``radius = 1, 2, …, max_radius``.  For each
      shell, visit every cell ``(x + dx, y + dy)`` where
      ``max(|dx|, |dy|) == radius`` (the Chebyshev-distance boundary).
    * Claim the first free in-bounds cell found and break.
    * If no free cell exists within ``max_radius``, emit a ``WARNING`` and
      leave the context at its original (shared) position.

    The traversal order within each shell is deterministic (row-major over
    ``dx`` then ``dy``) so results are reproducible for identical inputs.

    Parameters
    ----------
    grid_coords:
        Integer array of shape ``(num_contexts, 2)`` containing the
        quantised ``(x, y)`` positions produced by :func:`scale_to_grid`
        before collision handling.
    grid_size:
        Side length of the square grid.  Candidate cells are rejected if
        either coordinate falls outside ``[0, grid_size - 1]``.
    max_radius:
        Maximum Chebyshev radius of the spiral search.  Contexts that cannot
        be placed within this radius share a cell with another context.
        Increase ``--grid-size`` or ``--collision-radius`` to reduce this.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(num_contexts, 2)`` with collision-resolved
        positions.  The input array is not modified in place; a copy is
        returned.

    Notes
    -----
    This function is intentionally **not** named ``resolve_collisions`` in the
    parameter list of :func:`scale_to_grid` — the parameter there is called
    ``fix_collisions`` to avoid shadowing this function's name within the same
    module scope.
    """
    logger.info("Resolving collisions on the grid...")

    num_contexts = grid_coords.shape[0]
    occupied = set()
    resolved = grid_coords.copy()

    for idx in range(num_contexts):
        x, y = resolved[idx]
        original = (int(x), int(y))

        # Because we used np.clip in scale_to_grid, x and y are guaranteed 
        # to be within [0, grid_size - 1], but we check anyway.
        if original not in occupied and 0 <= x < grid_size and 0 <= y < grid_size:
            occupied.add(original)
            continue

        # Search outward in Chebyshev shells
        placed = False
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check the boundary of the current shell
                    if max(abs(dx), abs(dy)) != radius:
                        continue

                    nx, ny = x + dx, y + dy
                    candidate = (int(nx), int(ny))

                    if (
                        0 <= nx < grid_size
                        and 0 <= ny < grid_size
                        and candidate not in occupied
                    ):
                        resolved[idx] = [nx, ny]
                        occupied.add(candidate)
                        placed = True
                        break

                if placed:
                    break

            if placed:
                break

        if not placed:
            logger.warning(
                f"Context {idx} could not be uniquely placed within radius "
                f"{max_radius}. Leaving at original position {original}."
            )
            # We still add it to occupied. Future elements hitting this spot 
            # will spiral around the exact same origin.
            occupied.add(original)

    logger.success("Collision resolution completed.")
    return resolved


def scale_to_grid(
    coordinates: np.ndarray,
    grid_size: int,
    padding: int = 2,
    collision_radius: int = 10,
    resolve: bool = True,
) -> np.ndarray:
    """
    Scale continuous 2-D coordinates to an integer grid with optional
    collision resolution. Uses robust scaling to prevent outliers from
    collapsing the coordinate space.
    """
    # 1. Zero-Division Bug Fix
    if grid_size <= 2 * padding:
        raise ValueError(
            f"grid_size ({grid_size}) must be strictly greater than "
            f"2 * padding ({2 * padding}) to calculate cell expansion."
        )

    logger.info(
        f"Scaling continuous coordinates to a {grid_size}x{grid_size} grid "
        f"with padding={padding} using robust percentile bounds."
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # 2. Outlier Issue Fix: Use 1st and 99th percentiles instead of absolute min/max
    x_min, x_max = np.percentile(x, 1), np.percentile(x, 99)
    y_min, y_max = np.percentile(y, 1), np.percentile(y, 99)

    # Fallback in case of highly degenerate data where percentiles match
    if x_max == x_min:
        x_min, x_max = x.min(), x.max()
    if y_max == y_min:
        y_min, y_max = y.min(), y.max()

    logger.debug(f"Robust x range (1st-99th percentile): [{x_min:.4f}, {x_max:.4f}]")
    logger.debug(f"Robust y range (1st-99th percentile): [{y_min:.4f}, {y_max:.4f}]")

    # Expand ranges symmetrically by padding cells on each side
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Denominator is guaranteed to be > 0 due to the validation check above
    x_min -= padding * x_range / (grid_size - 2 * padding)
    x_max += padding * x_range / (grid_size - 2 * padding)
    y_min -= padding * y_range / (grid_size - 2 * padding)
    y_max += padding * y_range / (grid_size - 2 * padding)

    # Avoid division by zero if all points are virtually identical
    if x_max == x_min:
        x_max = x_min + 1e-6
    if y_max == y_min:
        y_max = y_min + 1e-6

    # Scale to [0, grid_size - 1]
    x_scaled = (x - x_min) / (x_max - x_min) * (grid_size - 1)
    y_scaled = (y - y_min) / (y_max - y_min) * (grid_size - 1)

    # 3. Clip the outliers so they stay within the grid boundaries
    x_scaled = np.clip(x_scaled, 0, grid_size - 1)
    y_scaled = np.clip(y_scaled, 0, grid_size - 1)

    grid_coords = np.stack([np.round(x_scaled), np.round(y_scaled)], axis=1).astype(int)

    logger.info("Initial quantisation to grid completed.")

    if resolve:
        grid_coords = resolve_collisions(
            grid_coords=grid_coords,
            grid_size=grid_size,
            max_radius=collision_radius,
        )

    logger.success("Grid mapping completed.")
    return grid_coords

# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------

def save_coordinates_csv(
    coordinates: np.ndarray,
    context_ids: List[str],
    output_path: Path,
    continuous: bool = False,
) -> None:
    """
    Persist coordinates to a human-readable CSV file.

    Writes a three-column CSV with a header row:

    * Column 1 — ``context_id``
    * Column 2 — ``x``  (float with 6 decimal places if ``continuous=True``,
      otherwise integer)
    * Column 3 — ``y``  (same format as ``x``)

    This file is intended for debugging and manual inspection **only**.
    Downstream pipeline steps must read ``context_coordinates.json`` instead.

    Parameters
    ----------
    coordinates:
        Array of shape ``(num_contexts, 2)``.  May be float (continuous) or
        int (grid).
    context_ids:
        Ordered list of context identifier strings matching the row order of
        ``coordinates``.
    output_path:
        Destination path including filename.  Parent directories are created
        if they do not exist.
    continuous:
        If ``True``, format each coordinate value as a float with 6 decimal
        places.  If ``False`` (default), format as an integer.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("context_id,x,y\n")
        for cid, (x, y) in zip(context_ids, coordinates):
            if continuous:
                fh.write(f"{cid},{x:.6f},{y:.6f}\n")
            else:
                fh.write(f"{cid},{x},{y}\n")

    logger.success(
        f"Saved CSV coordinates ({len(coordinates)} rows): {output_path}"
    )


def save_coordinates_json(
    grid_coords: np.ndarray,
    context_ids: List[str],
    output_path: Path,
) -> None:
    """
    Persist finalised grid coordinates to the primary JSON lookup map.

    This is the **authoritative** output consumed by ``phrase_fingerprints.py``
    and any other downstream automation.  It maps every ``context_id`` string
    to a dict with integer ``"x"`` and ``"y"`` keys so that callers can
    retrieve a context's position in ``O(1)`` without scanning a CSV file.

    Output format::

        {
            "context_0": {"x": 5,  "y": 12},
            "context_1": {"x": 14, "y": 3},
            ...
        }

    The coordinates stored here reflect the **final** positions after
    spiral-search collision resolution (when enabled), not the raw quantised
    positions.

    Parameters
    ----------
    grid_coords:
        Integer array of shape ``(num_contexts, 2)`` with values in
        ``[0, grid_size - 1]``, as returned by :func:`scale_to_grid`.
    context_ids:
        Ordered list of context identifier strings matching the row order of
        ``grid_coords``.
    output_path:
        Destination path.  Conventionally ``<output_dir>/context_coordinates.json``.
        Parent directories are created if they do not exist.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    coord_map = {
        cid: {"x": int(xy[0]), "y": int(xy[1])}
        for cid, xy in zip(context_ids, grid_coords)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(coord_map, fh, indent=2)

    logger.success(
        f"Saved JSON coordinate map ({len(coord_map)} contexts): {output_path}"
    )


def save_statistics(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Persist run-level summary statistics to a JSON file.

    The statistics dict written here is constructed in :func:`main` **after**
    all grid processing (including collision resolution) is complete.  This
    guarantees that reported values such as ``collision_rate`` and
    ``unique_positions`` describe the actual outputs rather than any
    intermediate state.

    Parameters
    ----------
    stats:
        Arbitrary JSON-serialisable dict.  Expected top-level keys:

        ``"num_contexts"``
            Total number of contexts processed.
        ``"method"``
            Dimensionality reduction method used (``"tsne"``, ``"umap"``,
            or ``"pca"``).
        ``"continuous"``
            Sub-dict with ``"x_range"`` and ``"y_range"`` from the
            continuous embedding.
        ``"grid"`` *(optional)*
            Sub-dict with ``"size"``, ``"total_contexts"``,
            ``"unique_positions"``, and ``"collision_rate"`` — present only
            when ``--no-grid`` is not set.

    output_path:
        Destination path.  Conventionally
        ``<output_dir>/coordinate_statistics.json``.  Parent directories are
        created if they do not exist.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    logger.success(f"Saved statistics: {output_path}")


# ---------------------------------------------------------------------------
# Optional Visualisation
# ---------------------------------------------------------------------------

def create_visualization(
    coordinates: np.ndarray,
    output_path: Path,
    title: str,
    show_density: bool = False,
) -> None:
    """
    Produce a 2-D scatter plot of the semantic space and save it as a PNG.

    The function is a best-effort visualisation helper.  If ``matplotlib`` is
    not installed, or if any plotting operation raises an exception, a
    ``WARNING`` is logged and the function returns silently without aborting
    the pipeline.

    When ``show_density=True`` the scatter points are coloured by a kernel
    density estimate so dense clusters are visually distinguishable from
    sparse regions.  If ``scipy.stats.gaussian_kde`` fails (e.g. singular
    covariance matrix for very small datasets), the function falls back to
    the plain coloured-by-index scatter.

    Parameters
    ----------
    coordinates:
        Float array of shape ``(num_contexts, 2)``.  May be continuous
        embedding coordinates or float-cast integer grid positions.
    output_path:
        Destination path for the PNG file.  Parent directories are created
        if they do not exist.
    title:
        Title string rendered at the top of the figure.
    show_density:
        If ``True``, colour points by Gaussian KDE density.  If ``False``
        (default for grid plots), colour points by their index in the array
        (effectively by the order contexts were processed).

    Notes
    -----
    The figure is always closed after saving (``plt.close()``) to prevent
    memory accumulation when the function is called multiple times in a
    single run.
    """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))
        
        if show_density:
            try:
                from scipy.stats import gaussian_kde

                xy = coordinates.T
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                scatter = ax.scatter(
                    coordinates[idx, 0],
                    coordinates[idx, 1],
                    c=z[idx],
                    s=20,
                    alpha=0.6,
                    cmap="viridis",
                )
                plt.colorbar(scatter, label="Density")
            except Exception:
                show_density = False

        if not show_density:
            ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                alpha=0.6,
                s=20,
                c=range(len(coordinates)),
                cmap="viridis",
            )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, alpha=0.3)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.success(f"Saved visualisation: {output_path}")

    except ImportError:
        logger.warning("matplotlib not available — skipping visualisation.")
    except Exception as exc:
        logger.warning(f"Visualisation failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Entry point for the ``semantic-space`` pipeline step.

    Parses command-line arguments, orchestrates the full embedding and grid-
    placement workflow, and returns an integer exit code.

    Workflow
    ~~~~~~~~
    1.  Parse CLI arguments via ``argparse``.
    2.  Load the sparse term-context matrix and metadata JSON.
    3.  Validate that the number of context IDs in the metadata matches the
        number of columns in the matrix.
    4.  Prepare context feature vectors (transpose + optional L2 normalise).
    5.  Reduce to 2-D continuous coordinates (t-SNE / UMAP / PCA).
    6.  Save continuous coordinates to ``context_coordinates_continuous.csv``.
    7.  Unless ``--no-grid`` is set:

        a.  Call :func:`scale_to_grid` to quantise continuous coords.
        b.  Optionally resolve collisions via spiral search.
        c.  Save discrete grid coordinates to ``context_coordinates.csv``
            (human inspection).
        d.  Save discrete grid coordinates to ``context_coordinates.json``
            (machine consumption — primary output for ``phrase_fingerprints.py``).

    8.  Compute and save summary statistics to ``coordinate_statistics.json``.
        **Statistics are computed after all grid processing is complete** so
        the reported collision rate and unique-position count accurately
        describe the final outputs.
    9.  Optionally generate visualisation PNG files.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a recoverable input error (logged before
        returning).

    Notes
    -----
    Output filenames are fixed strings and never embed the reduction method
    name or any runtime parameter.  Method-labelled names are used only for
    the optional visualisation PNGs so that successive runs with different
    methods do not clobber each other's diagnostic images.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Semantic Space Generation: embed contexts in a 2-D grid so that "
            "semantically adjacent contexts occupy neighbouring cells."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--matrix",
        required=True,
        help="Path to term_context_matrix.npz (phrases x contexts, CSR format).",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to term_context_matrix.json (must contain 'context_ids').",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory.  Created if it does not exist.",
    )

    # Reduction method
    parser.add_argument(
        "--method",
        choices=["tsne", "umap", "pca"],
        default="tsne",
        help="Dimensionality reduction method.",
    )

    # Grid parameters
    parser.add_argument(
        "--grid-size",
        type=int,
        default=16,
        dest="grid_size",
        help="Side length N of the N x N output grid.",
    )
    parser.add_argument(
        "--grid-padding",
        type=int,
        default=0,
        dest="grid_padding",
        help="Border margin in cells reserved on each side of the grid.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip grid quantisation — save continuous coordinates only.",
    )
    parser.add_argument(
        "--no-collision-resolution",
        action="store_true",
        help="Skip the spiral-search collision resolution step.",
    )
    parser.add_argument(
        "--collision-radius",
        type=int,
        default=10,
        dest="collision_radius",
        help="Maximum spiral search radius for collision resolution.",
    )

    # t-SNE parameters
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity (effective nearest-neighbour count).",
    )
    parser.add_argument(
        "--tsne-iter",
        type=int,
        default=1000,
        dest="tsne_iter",
        help="Maximum t-SNE optimisation iterations.",
    )

    # UMAP parameters
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        dest="n_neighbors",
        help="UMAP number of neighbours for manifold approximation.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.25,
        dest="min_dist",
        help="UMAP minimum distance between embedded points.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric for UMAP neighbour graph.",
    )

    # General parameters
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip L2 normalisation of context vectors before reduction.",
    )
    parser.add_argument(
        "--use-sparse",
        action="store_true",
        dest="use_sparse",
        help=(
            "Keep the context matrix sparse when passing it to UMAP or PCA. "
            "Reduces peak RAM usage for large corpora."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        dest="n_jobs",
        help="Parallel threads for t-SNE / UMAP.  -1 = all cores.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        dest="random_seed",
        help="Random seed for reproducibility.",
    )

    # Visualisation
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Produce PNG scatter plots of the continuous and grid embeddings.",
    )
    parser.add_argument(
        "--show-density",
        action="store_true",
        dest="show_density",
        help="Colour scatter points by KDE density in the continuous plot.",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Semantic Space Generation")
    logger.info(f"Method: {args.method.upper()} | Grid: {args.grid_size}x{args.grid_size}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    metadata    = load_metadata(Path(args.metadata))
    matrix      = load_sparse_matrix(Path(args.matrix))
    context_ids: List[str] = metadata["context_ids"]

    if len(context_ids) != matrix.shape[1]:
        logger.error(
            f"Mismatch: metadata has {len(context_ids)} context IDs "
            f"but matrix has {matrix.shape[1]} columns."
        )
        return 1

    # ------------------------------------------------------------------
    # 2. Prepare context vectors
    # ------------------------------------------------------------------
    keep_sparse = args.use_sparse and args.method in ["umap", "pca"]
    vectors = prepare_context_vectors(
        matrix,
        normalize=not args.no_normalize,
        keep_sparse=keep_sparse,
    )

    # ------------------------------------------------------------------
    # 3. Dimensionality reduction → continuous 2-D coordinates
    # ------------------------------------------------------------------
    if args.method == "tsne":
        coords = reduce_dimensions_tsne(
            vectors,
            args.perplexity,
            args.tsne_iter,
            args.n_jobs,
            args.random_seed,
        )
    elif args.method == "umap":
        coords = reduce_dimensions_umap(
            vectors,
            args.n_neighbors,
            args.min_dist,
            args.metric,
            args.n_jobs,
            args.random_seed,
        )
    else:
        coords = reduce_dimensions_pca(vectors)

    # ------------------------------------------------------------------
    # 4. Save continuous coordinates (fixed filename, no method suffix)
    # ------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_coordinates_csv(
        coords,
        context_ids,
        output_dir / "context_coordinates_continuous.csv",
        continuous=True,
    )

    # ------------------------------------------------------------------
    # 5. Grid quantisation + collision resolution
    # ------------------------------------------------------------------
    grid_coords: Optional[np.ndarray] = None

    if not args.no_grid:
        grid_coords = scale_to_grid(
            coordinates=coords,  # Using keywords for clarity
            grid_size=args.grid_size,
            padding=args.grid_padding,
            resolve=not args.no_collision_resolution, # Use 'resolve' instead of 'fix_collisions'
            collision_radius=args.collision_radius,
        )

        # CSV — human inspection only
        save_coordinates_csv(
            grid_coords,
            context_ids,
            output_dir / "context_coordinates.csv",
        )

        # JSON — primary machine-readable output for phrase_fingerprints.py
        save_coordinates_json(
            grid_coords,
            context_ids,
            output_dir / "context_coordinates.json",
        )

    # ------------------------------------------------------------------
    # 6. Statistics — computed AFTER all grid processing is finalised
    # ------------------------------------------------------------------
    stats: Dict[str, Any] = {
        "num_contexts": len(context_ids),
        "method":       args.method,
        "continuous": {
            "x_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
            "y_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
        },
    }

    if grid_coords is not None:
        unique_final = len(set(map(tuple, grid_coords.tolist())))
        stats["grid"] = {
            "size":             args.grid_size,
            "total_contexts":   len(context_ids),
            "unique_positions": unique_final,
            "collision_rate":   float(
                1.0 - unique_final / len(context_ids)
            ),
        }

    save_statistics(stats, output_dir / "coordinate_statistics.json")

    # ------------------------------------------------------------------
    # 7. Optional visualisations
    # ------------------------------------------------------------------
    if args.visualize:
        create_visualization(
            coords,
            output_dir / f"semantic_space_{args.method}_continuous.png",
            f"Semantic Space — {args.method.upper()} (continuous)",
            args.show_density,
        )
        if grid_coords is not None:
            create_visualization(
                grid_coords.astype(float),
                output_dir / f"semantic_space_{args.method}_grid.png",
                f"Semantic Space — {args.grid_size}x{args.grid_size} Grid",
                show_density=False,
            )

    logger.info("=" * 60)
    logger.success(f"Done.  Outputs written to: {output_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
