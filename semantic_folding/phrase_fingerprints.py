
from __future__ import annotations

from scipy.ndimage import gaussian_filter
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
from scipy.ndimage import gaussian_filter1d

from lib import get_logger
logger = get_logger("phrase_fingerprints")

"""
phrase_fingerprints.py
======================

Pipeline step: **phrase-fingerprints**

Converts embedded, grid-placed semantic contexts into fixed-length binary
fingerprint vectors that downstream classifiers and similarity search can
consume directly.

Overview
--------
This step sits between ``semantic_space.py`` (which embeds contexts onto a 2-D
integer grid) and any model training or retrieval step that needs a compact,
deterministic numeric representation of each phrase's semantic neighbourhood.

For every phrase that appears in the phrase-metadata JSON the script:

1.  Looks up the phrase's parent context in ``context_coordinates.json`` to
    obtain its ``(x, y)`` position on the semantic grid.
2.  Linearises the 2-D coordinate into a 1-D index using either Morton
    (Z-order) encoding or simple row-major order.
3.  Sets that index as a "hot" bit in a zero-initialised binary vector whose
    length equals ``grid_size * grid_size``.
4.  Optionally smooths the fingerprint with a configurable Gaussian kernel so
    that nearby contexts contribute fractional signal rather than hard zeros.
5.  Writes all fingerprint vectors to a compressed ``.npz`` archive and a
    companion ``phrase_fingerprints_meta.json`` that maps every phrase token to
    its row index in the matrix.

Inputs
------
``context_coordinates.json``
    Primary output of ``semantic_space.py``.  Maps every ``context_id`` to its
    finalised integer grid position after spiral-search collision resolution::

        {
            "context_0": {"x": 5,  "y": 12},
            "context_1": {"x": 14, "y": 3}
        }

``phrase_metadata.json``
    Produced by an earlier pipeline step (e.g. ``phrase_extraction.py``).
    Each entry associates a phrase token with the context it was drawn from::

        {
            "phrase_id": "ph_0042",
            "token":     "neural plasticity",
            "context_id": "context_7"
        }

Outputs
-------
``phrase_fingerprints.npz``
    Compressed NumPy archive.  Contains a single 2-D ``float32`` array named
    ``"fingerprints"`` with shape ``(n_phrases, grid_size * grid_size)``.

``phrase_fingerprints_meta.json``
    JSON map from phrase token strings to their row index in the fingerprint
    matrix::

        {
            "neural plasticity": 42,
            "synaptic pruning":  43
        }

Morton Indexing
---------------
The ``--use-morton`` flag (default: enabled) encodes 2-D coordinates as
Morton (Z-order curve) codes before placing the hot bit.  This preserves 2-D
spatial locality in the 1-D fingerprint index so that bitwise Hamming distance
between two fingerprints roughly correlates with semantic distance on the grid.

When Morton encoding is disabled (``--no-morton``) simple row-major order is
used instead::

    index = y * grid_size + x

Gaussian Smoothing
------------------
When ``--smooth`` is set, the raw binary fingerprint is convolved with a 1-D
Gaussian kernel (sigma configurable via ``--sigma``).  The kernel is applied
**after** all hot bits have been placed so the smoothed result integrates
signal from every context associated with a phrase before normalisation.

Usage
-----
::

    python phrase_fingerprints.py \\
        --coordinates   runs/run_001/context_coordinates.json \\
        --metadata      runs/run_001/phrase_metadata.json \\
        --output-dir    runs/run_001/ \\
        --grid-size     64 \\
        --use-morton \\
        --smooth \\
        --sigma         1.0

    python phrase_fingerprints.py \\
        --coordinates   runs/run_001/context_coordinates.json \\
        --metadata      runs/run_001/phrase_metadata.json \\
        --output-dir    runs/run_001/ \\
        --grid-size     64 \\
        --no-morton \\
        --no-smooth

Exit Codes
----------
0   Success — all phrases fingerprinted and written.
1   Input file not found or unreadable.
2   Malformed JSON in coordinates or metadata file.
3   Grid size mismatch — a coordinate value exceeds ``grid_size - 1``.
4   Unexpected runtime error.
"""


# ---------------------------------------------------------------------------
# Morton (Z-order) encoding helpers
# ---------------------------------------------------------------------------

def _spread_bits(value: int) -> int:
    """
    Spread the bits of a non-negative integer by inserting a zero between
    every pair of adjacent bits.

    This is the core primitive for 2-D Morton code computation.  Given an
    integer whose binary representation is ``...b3 b2 b1 b0``, the result is
    ``...0 b3 0 b2 0 b1 0 b0``.

    Parameters
    ----------
    value:
        Non-negative integer to spread.  Behaviour is undefined for negative
        values or values that require more than 16 bits.

    Returns
    -------
    int
        The bit-spread integer.

    Examples
    --------
    >>> _spread_bits(0b0011)   # 3
    0b00000101               # 5
    >>> _spread_bits(0b1010)   # 10
    0b01000100               # 68
    """
    value &= 0x0000FFFF
    value = (value | (value << 8))  & 0x00FF00FF
    value = (value | (value << 4))  & 0x0F0F0F0F
    value = (value | (value << 2))  & 0x33333333
    value = (value | (value << 1))  & 0x55555555
    return value


def xy_to_morton(x: int, y: int, grid_size: int) -> int:
    """
    Encode a 2-D integer coordinate as a Morton (Z-order curve) code.

    The Morton code interleaves the bits of ``x`` and ``y`` so that
    coordinates that are close in 2-D space map to indices that are close in
    1-D space.  This locality property makes the resulting fingerprint index
    meaningful under Hamming distance.

    The code is **not** the raw Morton number but is remapped to the range
    ``[0, grid_size * grid_size - 1]`` so it can directly index a fingerprint
    vector of that length.

    Parameters
    ----------
    x:
        Column coordinate on the semantic grid, in ``[0, grid_size - 1]``.
    y:
        Row coordinate on the semantic grid, in ``[0, grid_size - 1]``.
    grid_size:
        Side length of the square grid.  Must be a power of two for the
        Morton mapping to be bijective.

    Returns
    -------
    int
        Morton index in ``[0, grid_size * grid_size - 1]``.

    Raises
    ------
    ValueError
        If ``x`` or ``y`` is negative or ``>= grid_size``.

    Examples
    --------
    >>> xy_to_morton(0, 0, 64)
    0
    >>> xy_to_morton(1, 0, 64)
    1
    >>> xy_to_morton(0, 1, 64)
    2
    >>> xy_to_morton(1, 1, 64)
    3
    """
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        raise ValueError(
            f"Coordinates ({x}, {y}) out of range for grid_size={grid_size}."
        )
    return _spread_bits(x) | (_spread_bits(y) << 1)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_context_coordinates(coordinates_path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Load finalised context coordinates from the JSON map produced by
    ``semantic_space.py``.

    The function reads ``context_coordinates.json`` — the primary machine-
    readable output of the semantic-space step — and returns a plain Python
    dict for ``O(1)`` lookup by ``context_id``.

    Expected file format::

        {
            "CTX-AI-01": {"x": 5,  "y": 12},
            "CTX-AI-02": {"x": 14, "y": 3},
            ...
        }

    Keys are kept as-is from the JSON file (strings such as ``"CTX-AI-01"``).
    They are **not** converted to integers.  The integer column indices stored
    in ``phrase_contexts`` must first be resolved to string IDs via the
    ``context_ids`` lookup table before querying this dict — see
    :func:`main` for the bridging logic.

    Entries that are malformed (missing keys, non-integer coordinate values,
    etc.) are skipped with a ``WARNING`` log message so a single bad row does
    not abort the entire run.

    Parameters
    ----------
    coordinates_path:
        Absolute or relative path to ``context_coordinates.json``.

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Mapping from string ``context_id`` (e.g. ``"CTX-AI-01"``) to
        ``(x, y)`` integer tuples.

    Raises
    ------
    FileNotFoundError
        If ``coordinates_path`` does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(coordinates_path, "r", encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    coordinates: Dict[str, Tuple[int, int]] = {}
    for context_id, xy in raw.items():
        try:
            coordinates[context_id] = (int(xy["x"]), int(xy["y"]))  # key stays str
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(f"Skipping malformed coordinate entry '{context_id}': {exc}")

    return coordinates
 


def load_phrase_metadata(path: str):
    """
    Load phrase list and frequencies from the term_context_matrix.json
    produced by term_context.py.

    Args:
        path: Path to term_context_matrix.json

    Returns:
        phrases    : List[str]  — phrase strings in matrix row order
        frequencies: List[int]  — parallel per-phrase frequency counts
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required keys are missing or types are wrong.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: '{path}'")

    logger.info(f"Loading phrase metadata from: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in '{path}', got {type(data).__name__}."
        )

    for key in ("phrases", "phrase_frequencies"):
        if key not in data:
            raise ValueError(
                f"Required key '{key}' not found in '{path}'. "
                f"Found keys: {list(data.keys())}"
            )

    phrases     = data["phrases"]
    frequencies = data["phrase_frequencies"]

    if len(phrases) != len(frequencies):
        raise ValueError(
            f"Length mismatch in '{path}': "
            f"phrases={len(phrases)}, phrase_frequencies={len(frequencies)}."
        )

    logger.success(
        f"Loaded {len(phrases)} phrases "
        f"(num_contexts={data.get('num_contexts', '?')}, "
        f"matrix_shape={data.get('matrix_shape', '?')})."
    )
    return phrases, frequencies

# ---------------------------------------------------------------------------
# Fingerprint construction
# ---------------------------------------------------------------------------



def build_and_smooth_fingerprint(
    phrase_text: str,
    context_coords_list: List[Tuple[int, int]],
    grid_size: int,
    use_morton: bool,
    apply_smooth: bool,
    sigma: float
) -> np.ndarray:
    """
    Constructs a topological Semantic Fingerprint for a given phrase.

    This function builds a sparse, multi-hot 2D representation of a phrase based 
    on the semantic contexts it appears in. It avoids the "centroid fallacy" by 
    activating all corresponding context coordinates on a 2D grid rather than 
    averaging them. It optionally applies true 2D spatial smoothing to capture 
    semantic overlap before flattening the grid into a 1D vector (using either 
    linear or Morton Z-order encoding). Finally, the vector is min-max 
    normalized to the [0.0, 1.0] range.

    Args:
        phrase_text (str): 
            The textual representation of the phrase (used primarily for error 
            logging/tracking).
        context_coords_list (List[Tuple[int, int]]): 
            A list of (x, y) integer tuples representing the 2D grid coordinates 
            of the contexts in which this phrase appears.
        grid_size (int): 
            The width and height of the square semantic grid (e.g., 64 implies 
            a 64x64 grid).
        use_morton (bool): 
            If True, flattens the 2D grid into a 1D array using Morton (Z-order) 
            encoding, which better preserves 2D spatial locality in 1D space. 
            If False, uses standard row-major (y * grid_size + x) flattening.
        apply_smooth (bool): 
            If True, applies a 2D Gaussian blur to the grid to create overlapping 
            semantic topologies.
        sigma (float): 
            The standard deviation for the Gaussian filter. Higher values create 
            a wider "blur" or semantic generalization. Ignored if apply_smooth 
            is False.

    Returns:
        np.ndarray: 
            A 1D numpy array of shape (grid_size * grid_size,) with dtype 
            float32, containing the normalized semantic fingerprint.

    Raises:
        ValueError: 
            If `context_coords_list` is empty, indicating the phrase has no 
            valid contexts to map.
    """
    if not context_coords_list:
        raise ValueError(f"Phrase '{phrase_text}': no coordinates provided.")

    # 1. Create the 2D grid
    grid_2d = np.zeros((grid_size, grid_size), dtype=np.float32)

    # 2. Activate ALL contexts (multi-hot representation)
    for x, y in context_coords_list:
        # Protect against out-of-bounds just in case
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid_2d[y, x] += 1.0  # Accumulate overlapping points

    # 3. Apply True 2D Semantic Smoothing
    if apply_smooth and sigma > 0.0:
        grid_2d = gaussian_filter(grid_2d, sigma=sigma)

    # Normalize to [0, 1]
    max_val = grid_2d.max()
    if max_val > 0.0:
        grid_2d /= max_val

    # 4. Linearise to 1D
    vector_size = grid_size * grid_size
    fp = np.zeros(vector_size, dtype=np.float32)

    for y in range(grid_size):
        for x in range(grid_size):
            val = grid_2d[y, x]
            if val > 0:
                idx = xy_to_morton(x, y, grid_size) if use_morton else (y * grid_size + x)
                fp[idx] = val

    return fp


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(
    fingerprints: np.ndarray,
    skipped: int,
    total: int,
) -> dict:
    """
    Compute summary statistics over the finalised fingerprint matrix.

    Statistics are computed **after** all fingerprints have been built and
    smoothed so that the reported numbers reflect the true final state of the
    matrix rather than an intermediate accumulation.

    Parameters
    ----------
    fingerprints:
        2-D float32 array of shape ``(n_phrases, vector_size)`` containing all
        finalised fingerprint vectors.
    skipped:
        Number of phrases that were skipped because their ``context_id`` was
        absent from ``context_coordinates.json``.
    total:
        Total number of phrase entries in ``phrase_metadata.json`` before any
        filtering.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``total_phrases``
            Total phrases in the metadata file.
        ``fingerprinted_phrases``
            Phrases that received a fingerprint (``total - skipped``).
        ``skipped_phrases``
            Phrases whose context was not in the coordinate map.
        ``skip_rate_pct``
            ``skipped / total * 100`` rounded to two decimal places.
        ``vector_size``
            Length of each fingerprint vector.
        ``sparsity_pct``
            Percentage of elements in the matrix that are exactly ``0.0``.
        ``mean_max_activation``
            Average of the per-row maximum values, indicating how strongly each
            fingerprint peaks after smoothing.
    """
    n_phrases, vector_size = fingerprints.shape
    total_elements = n_phrases * vector_size

    zero_elements = int(np.sum(fingerprints == 0.0))
    sparsity = (zero_elements / total_elements * 100) if total_elements > 0 else 0.0
    mean_max = float(np.mean(np.max(fingerprints, axis=1))) if n_phrases > 0 else 0.0

    return {
        "total_phrases":       total,
        "fingerprinted_phrases": total - skipped,
        "skipped_phrases":     skipped,
        "skip_rate_pct":       round(skipped / total * 100, 2) if total > 0 else 0.0,
        "vector_size":         vector_size,
        "sparsity_pct":        round(sparsity, 2),
        "mean_max_activation": round(mean_max, 6),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    fingerprints    : np.ndarray,
    token_index_map : Dict[str, int],
    stats           : dict,
    output_dir      : Path,
    use_morton      : bool,          # NEW: encode whether Morton encoding was used
    grid_size       : int,           # NEW: store grid size for validation
) -> None:
    """
    Persist fingerprint matrix, token-index map, and run statistics to disk.

    Three files are written to ``output_dir``:

    ``phrase_fingerprints.npz``
        Compressed NumPy archive containing a single array named
        ``"fingerprints"`` with shape ``(n_phrases, vector_size)`` and dtype
        ``float32``.

    ``phrase_fingerprints_meta.json``
        JSON object containing:
        - ``"phrase_to_row"`` : mapping from phrase string to row index
          (the same as the original token‑index map).
        - ``"use_morton"``    : boolean indicating whether fingerprints are
          in Morton (Z‑order) encoding (vs row‑major).
        - ``"grid_size"``     : side length of the square semantic grid.
        Downstream steps (e.g., Step 5) can use this additional information
        to correctly back‑project fingerprints onto the 2D grid.

    ``phrase_fingerprints_stats.json``
        JSON object containing the summary statistics returned by
        :func:`compute_stats`.  Useful for run logging and quality gates.

    Parameters
    ----------
    fingerprints : np.ndarray, shape (n_phrases, vector_size), dtype float32
        The dense fingerprint matrix.
    token_index_map : Dict[str, int]
        Mapping from phrase token string to its row in the matrix.
    stats : dict
        Statistics produced by :func:`compute_stats`.
    output_dir : Path
        Directory in which to write the three output files.
    use_morton : bool
        Whether the 1D vectors were linearised via Morton (Z‑order)
        encoding (True) or row‑major (False).
    grid_size : int
        Side length of the square semantic grid; used to calculate
        vector length = grid_size * grid_size.

    Raises
    ------
    OSError
        If any file cannot be written (permissions, disk full, etc.).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path   = output_dir / "phrase_fingerprints.npz"
    meta_path  = output_dir / "phrase_fingerprints_meta.json"
    stats_path = output_dir / "phrase_fingerprints_stats.json"

    # --- fingerprint matrix ---
    np.savez_compressed(str(npz_path), fingerprints=fingerprints)
    logger.success(f"Fingerprint matrix written → {npz_path}  shape={fingerprints.shape}")

    # --- structured metadata (now includes encoding info) ---
    meta_dict = {
        "phrase_to_row": token_index_map,
        "use_morton": use_morton,
        "grid_size": grid_size,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_dict, fh, ensure_ascii=False, indent=2)
    logger.success(
        f"Metadata written → {meta_path}  ({len(token_index_map):,} phrases, "
        f"morton={use_morton}, grid={grid_size})"
    )

    # --- run statistics ---
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    logger.success(f"Run statistics written → {stats_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace with the following attributes:

        ``coordinates`` (Path)
            Path to ``context_coordinates.json``.
        ``metadata`` (Path)
            Path to ``phrase_metadata.json``.
        ``output_dir`` (Path)
            Directory for output files.
        ``grid_size`` (int)
            Side length of the semantic grid.
        ``use_morton`` (bool)
            ``True``  → Morton linearisation.
            ``False`` → row-major linearisation.
        ``smooth`` (bool)
            Whether to apply Gaussian smoothing.
        ``sigma`` (float)
            Gaussian sigma for smoothing.
    """
    parser = argparse.ArgumentParser(
        prog="phrase_fingerprints.py",
        description=(
            "Convert semantic-space context coordinates into fixed-length "
            "binary fingerprint vectors for every phrase in the corpus."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--coordinates",
        required=True,
        type=Path,
        help=(
            "Path to context_coordinates.json produced by semantic_space.py. "
            "Do NOT pass the .csv variant — it is for human inspection only."
        ),
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to phrase_metadata.json produced by an upstream extraction step.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        dest="output",
        help="Directory into which fingerprint outputs are written.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        dest="grid_size",
        help=(
            "Side length of the square semantic grid.  Must match the value "
            "used in semantic_space.py.  Fingerprint vectors will have length "
            "grid_size * grid_size."
        ),
    )

    morton_group = parser.add_mutually_exclusive_group()
    morton_group.add_argument(
        "--morton",
        action="store_true",
        default=True,
        dest="use_morton",
        help=(
            "Linearise 2-D grid coordinates with Morton (Z-order) encoding. "
            "Preserves spatial locality in the fingerprint index (default)."
        ),
    )
    morton_group.add_argument(
        "--no-morton",
        action="store_false",
        dest="use_morton",
        help="Use row-major linearisation instead of Morton encoding.",
    )

    parser.add_argument(
        "--no-smooth",
        action="store_true",
        default=False,
        dest="no_smooth",
        help="Emit raw binary fingerprint vectors without smoothing.",
    )

    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=1.0,
        dest="sigma",
        help=(
            "Standard deviation of the Gaussian smoothing kernel in index "
            "units.  Ignored when --no-smooth is set."
        ),
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_inputs(args: argparse.Namespace) -> None:
    """
    Validate input paths and parameter ranges before any file I/O begins.

    Checks performed
    ~~~~~~~~~~~~~~~~
    * ``--coordinates`` must exist and have a ``.json`` extension.
    * ``--metadata`` must exist.
    * ``--grid-size`` must be a positive integer.
    * ``--sigma`` must be non-negative.
    * ``--grid-size`` should ideally be a power of two when Morton encoding is
      enabled (emits a WARNING if not, does not abort).

    Parameters
    ----------
    args:
        Parsed argument namespace from :func:`parse_args`.

    Raises
    ------
    SystemExit
        With exit code ``1`` if any hard validation fails (file not found,
        wrong extension, non-positive grid size, negative sigma).
    """
    errors: List[str] = []

    if not args.coordinates.exists():
        errors.append(f"Coordinates file not found: {args.coordinates}")

    if args.coordinates.suffix != ".json":
        errors.append(
            f"--coordinates must point to a .json file "
            f"(got '{args.coordinates.suffix}'). "
            f"Pass context_coordinates.json, not the .csv variant."
        )

    if not args.metadata.exists():
        errors.append(f"Metadata file not found: {args.metadata}")

    if args.grid_size <= 0:
        errors.append(f"--grid-size must be a positive integer (got {args.grid_size}).")

    if args.sigma < 0.0:
        errors.append(f"--sigma must be non-negative (got {args.sigma}).")

    if errors:
        for msg in errors:
            logger.error(msg)
        sys.exit(1)

    # Non-fatal: warn if grid_size is not a power of two with Morton encoding.
    if args.use_morton and (args.grid_size & (args.grid_size - 1)) != 0:
        logger.warning(
            f"--grid-size {args.grid_size} is not a power of two.  Morton "
            f"encoding requires a power-of-two grid for a bijective index "
            f"mapping.  Consider 32, 64, 128, or 256."
        )


def validate_grid_bounds(
    coordinates: Dict[str, Tuple[int, int]],
    grid_size: int,
) -> None:
    """
    Verify that every loaded coordinate fits within the declared grid.

    If any context has an ``x`` or ``y`` value ``>= grid_size`` the script
    cannot safely build fingerprints because the computed index would exceed
    the vector length.  This situation indicates a mismatch between the
    ``--grid-size`` argument and the grid size used in ``semantic_space.py``.

    Parameters
    ----------
    coordinates:
        Mapping from ``context_id`` to ``(x, y)`` as returned by
        :func:`load_context_coordinates`.
    grid_size:
        Declared grid side length from ``--grid-size``.

    Raises
    ------
    SystemExit
        With exit code ``3`` if any coordinate is out of bounds.
    """
    out_of_bounds = [
        (cid, x, y)
        for cid, (x, y) in coordinates.items()
        if x >= grid_size or y >= grid_size or x < 0 or y < 0
    ]

    if out_of_bounds:
        logger.error(
            f"{len(out_of_bounds):,} context(s) have coordinates outside "
            f"[0, {grid_size - 1}].  The --grid-size argument does not match "
            f"the grid used in semantic_space.py.  First offenders:"
        )
        for cid, x, y in out_of_bounds[:5]:
            logger.error(f"  {cid}: x={x}, y={y}")
        sys.exit(3)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the ``phrase-fingerprints`` pipeline step.

    Orchestrates the full fingerprinting workflow:

    1.  Parse and validate CLI arguments.
    2.  Load ``context_coordinates.json`` — string-keyed ``Dict[str, (x, y)]``.
    3.  Load ``term_context_matrix.json`` (phrases + frequencies).
    4.  Load ``phrase_contexts`` and ``context_ids`` from metadata.
        ``phrase_contexts`` maps each phrase to a list of **integer column
        indices**; ``context_ids`` is the ordered list of string context IDs
        (e.g. ``["CTX-AI-01", "CTX-AI-02", …]``) that bridges those indices
        to the string keys used in ``coordinates``.
    5.  Validate grid bounds.
    6.  For each phrase, resolve its integer context indices → string IDs →
        ``(x, y)`` coordinates, build a 2-D multi-hot fingerprint, optionally
        apply 2-D Gaussian smoothing, and linearise using Morton or row-major
        encoding.
    7.  Compute summary statistics over the **finalised** matrix.
    8.  Write ``.npz``, ``_meta.json``, and ``_stats.json`` outputs.

    Parameters
    ----------
    argv:
        Argument list forwarded to :func:`parse_args`.  Pass ``None`` to read
        from ``sys.argv``.

    Returns
    -------
    None
        The function calls ``sys.exit`` on any unrecoverable error.

    Notes
    -----
    The integer-to-string ID bridge (step 4) is the critical alignment point
    between Step 2 and Step 4 of the pipeline:

    * ``phrase_contexts["representation"] = [0, 2, 3]``
      — integer column indices produced by Step 2.
    * ``context_ids[0] = "CTX-AI-01"``
      — the lookup table that maps each index to its string context ID.
    * ``coordinates["CTX-AI-01"] = (5, 12)``
      — the string-keyed coordinate map produced by Step 3.

    Without loading ``context_ids`` and using it as the bridge, the lookup
    ``ctx_idx in coordinates`` would always miss because integer keys never
    exist in a string-keyed dictionary.

    Statistics (skip rate, sparsity, mean max activation) are computed in
    step 7 — **after** fingerprint construction and smoothing are complete —
    so they accurately represent the final outputs rather than any
    intermediate state.
    """
    args = parse_args(argv)
    validate_inputs(args)

    logger.info("=" * 60)
    logger.info("phrase_fingerprints.py  —  starting")
    logger.info(f"  coordinates : {args.coordinates}")
    logger.info(f"  metadata    : {args.metadata}")
    logger.info(f"  output_dir  : {args.output}")
    logger.info(f"  grid_size   : {args.grid_size}")
    logger.info(f"  use_morton  : {args.use_morton}")
    logger.info(f"  smooth      : {not args.no_smooth}  (sigma={args.sigma})")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load context coordinates
    #    Keys are strings (e.g. "CTX-AI-01") as written by Step 3.
    #    load_context_coordinates must NOT convert them to int.
    # ------------------------------------------------------------------
    try:
        coordinates = load_context_coordinates(args.coordinates)
    except json.JSONDecodeError as exc:
        logger.error(f"Malformed JSON in coordinates file: {exc}")
        sys.exit(2)

    # ------------------------------------------------------------------
    # 2. Load phrases and frequencies from metadata
    # ------------------------------------------------------------------
    try:
        phrases, frequencies = load_phrase_metadata(args.metadata)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error(f"Malformed or unexpected JSON in metadata file: {exc}")
        sys.exit(2)

    # ------------------------------------------------------------------
    # 3. Load phrase_contexts and context_ids from metadata
    #
    #    phrase_contexts : Dict[str, List[int]]
    #        Maps each phrase string to a list of integer column indices
    #        (e.g. {"representation": [0, 2, 3], …}).  These indices are
    #        positions into context_ids, NOT keys into coordinates.
    #
    #    context_ids : List[str]
    #        Ordered list of string context IDs produced by Step 2
    #        (e.g. ["CTX-AI-01", "CTX-AI-02", …]).  Acts as the lookup
    #        table that converts an integer index → string ID so we can
    #        then retrieve (x, y) from the coordinates dict.
    # ------------------------------------------------------------------
    try:
        with open(args.metadata, encoding="utf-8") as fh:
            metadata = json.load(fh)

        phrase_contexts: Dict[str, List[int]] = metadata.get("phrase_contexts", {})
        context_ids: List[str]                = metadata.get("context_ids", [])

        if not phrase_contexts:
            logger.error(
                "Metadata missing 'phrase_contexts' field. "
                "Please re-run Step 2 (term_context.py) to generate this field."
            )
            sys.exit(2)

        if not context_ids:
            logger.error(
                "Metadata missing 'context_ids' field. "
                "This list is required to map integer context indices to string "
                "coordinate keys. Please re-run Step 2 (term_context.py)."
            )
            sys.exit(2)

        logger.info(f"Loaded phrase-context mappings for {len(phrase_contexts):,} phrases")
        logger.info(f"Loaded context_ids lookup table  ({len(context_ids):,} entries)")

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error(f"Failed to load phrase_contexts / context_ids from metadata: {exc}")
        sys.exit(2)

    # ------------------------------------------------------------------
    # 4. Validate grid bounds
    # ------------------------------------------------------------------
    validate_grid_bounds(coordinates, args.grid_size)

    # ------------------------------------------------------------------
    # 5. Build fingerprints
    # ------------------------------------------------------------------
    vector_size     = args.grid_size * args.grid_size
    n_phrases       = len(phrases)
    fingerprints    = np.zeros((n_phrases, vector_size), dtype=np.float32)
    token_index_map: Dict[str, int] = {}
    skipped         = 0

    logger.info(f"Building fingerprints for {n_phrases:,} phrases …")

    for row_idx, phrase_text in tqdm(enumerate(phrases)):
        phrase_id = str(row_idx)

        # Integer column indices where this phrase appears across contexts.
        # These are positions into context_ids, not keys into coordinates.
        context_indices: List[int] = phrase_contexts.get(phrase_text, [])

        if not context_indices:
            logger.warning(
                f"Phrase '{phrase_text}' (id={phrase_id}): "
                f"no contexts found — skipping."
            )
            skipped += 1
            continue

        # Bridge: int index → string context ID → (x, y) coordinate
        #
        #   ctx_idx          : int   — column index from phrase_contexts
        #   context_ids[idx] : str   — e.g. "CTX-AI-01"
        #   coordinates[sid] : tuple — (x, y) grid position
        #
        # We guard against out-of-range indices (malformed metadata) and
        # missing coordinate entries (context exists in matrix but was
        # excluded from the coordinate map for any reason).
        context_coords_list = []
        for ctx_idx in context_indices:
            if ctx_idx >= len(context_ids):
                logger.warning(
                    f"Phrase '{phrase_text}': ctx_idx {ctx_idx} is out of range "
                    f"for context_ids (len={len(context_ids)}) — skipping entry."
                )
                continue

            str_id = context_ids[ctx_idx]          # int  →  "CTX-AI-01"

            if str_id not in coordinates:
                logger.warning(
                    f"Phrase '{phrase_text}': context '{str_id}' not found in "
                    f"coordinates map — skipping entry."
                )
                continue

            context_coords_list.append(coordinates[str_id])   # (x, y)

        if not context_coords_list:
            logger.warning(
                f"Phrase '{phrase_text}' (id={phrase_id}): "
                f"none of its contexts resolved to coordinates — skipping."
            )
            skipped += 1
            continue

        try:
            fp = build_and_smooth_fingerprint(
                phrase_text=phrase_text,
                context_coords_list=context_coords_list,
                grid_size=args.grid_size,
                use_morton=args.use_morton,
                apply_smooth=not args.no_smooth,
                sigma=args.sigma,
            )
        except ValueError as exc:
            logger.warning(
                f"Phrase '{phrase_text}' (id={phrase_id}): "
                f"fingerprint error — {exc} — skipping."
            )
            skipped += 1
            continue

        fingerprints[row_idx]        = fp
        token_index_map[phrase_text] = row_idx

    logger.info(
        f"Fingerprinting complete: "
        f"{n_phrases - skipped:,} built, {skipped:,} skipped."
    )

    # ------------------------------------------------------------------
    # 6. Compute statistics AFTER finalisation
    # ------------------------------------------------------------------
    stats = compute_stats(fingerprints, skipped, n_phrases)

    logger.info("Run statistics:")
    for key, val in stats.items():
        logger.info(f"  {key:<28} {val}")

    # ------------------------------------------------------------------
    # 7. Write outputs
    # ------------------------------------------------------------------
    try:
        write_outputs(
            fingerprints    = fingerprints,
            token_index_map = token_index_map,
            stats           = stats,
            output_dir      = args.output,
            use_morton      = args.use_morton,   # from CLI
            grid_size       = args.grid_size,    # from CLI
        )
    except OSError as exc:
        logger.error(f"Failed to write outputs: {exc}")
        sys.exit(4)

    logger.success("phrase_fingerprints.py  —  done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
