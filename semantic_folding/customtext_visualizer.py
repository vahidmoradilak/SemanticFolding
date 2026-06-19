"""
Customtext Fingerprint Visualizer
PhD Thesis: Semantic Folding for Closed-Domain QA
Step 6: Customtext Fingerprint Analysis Dashboard

Visualizes Customtext-level semantic fingerprints with spatial analysis.
Handles sparse TF-IDF weighted fingerprints with proper normalization.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import argparse
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from doc_fingerprints import morton_to_xy
# Import from your lib module
from lib import load_document_fingerprints, load_phrase_fingerprints_sparse, get_logger

# Initialize logger
logger = get_logger("customtext_visualizer")


def inverse_flatten(flat_vector: np.ndarray, grid_size: int, use_morton: bool = False) -> np.ndarray:
    """
    Reconstruct 2D grid from flattened fingerprint vector.
    
    Args:
        flat_vector: 1D array of length grid_size².
        grid_size: Side length of the square grid.
        use_morton: If True, the vector is Morton-encoded; else row-major.
        
    Returns:
        2D numpy array (grid_size, grid_size).
    """
    if not use_morton:
        return flat_vector.reshape(grid_size, grid_size)
    
    # Morton-encoded: map each index back to (x,y)
    grid_2d = np.zeros((grid_size, grid_size), dtype=flat_vector.dtype)
    for idx, val in enumerate(flat_vector):
        if val != 0:
            x, y = morton_to_xy(idx, grid_size)
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid_2d[y, x] = val
    return grid_2d

def normalize_for_display(fp_2d: np.ndarray) -> np.ndarray:
    """
    Normalize only active cells to [0, 1] range for display.
    
    TF-IDF weighted fingerprints have very small values that map to near-zero
    in standard colormaps, appearing as black. This function normalizes only
    the active (non-zero) cells to the full [0, 1] range for visibility.
    
    Args:
        fp_2d: 2D fingerprint array with TF-IDF weights
        
    Returns:
        Normalized 2D array where active cells span [0, 1]
    """
    result = np.zeros_like(fp_2d)
    active_mask = fp_2d > 0
    
    if active_mask.any():
        active_vals = fp_2d[active_mask]
        min_val = active_vals.min()
        max_val = active_vals.max()
        
        logger.debug(f"Normalizing {active_mask.sum()} active cells: "
                    f"range [{min_val:.6f}, {max_val:.6f}] → [0, 1]")
        
        # Normalize to [0, 1] with epsilon to avoid division by zero
        result[active_mask] = (active_vals - min_val) / (max_val - min_val + 1e-10)
    else:
        logger.warning("No active cells found in fingerprint")
    
    return result


def get_top_active_cells(fingerprint_2d: np.ndarray, top_n: int = 10) -> List[Dict]:
    """
    Extract top-N most activated cells with their coordinates and values.
    
    Args:
        fingerprint_2d: 2D fingerprint array
        top_n: Number of top cells to extract
        
    Returns:
        List of dicts with keys: rank, row, col, value
    """
    # Get indices of top-N values in flattened array
    flat_indices = np.argsort(fingerprint_2d.ravel())[::-1][:top_n]
    rows, cols = np.unravel_index(flat_indices, fingerprint_2d.shape)
    
    cells = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        cells.append({
            'rank': i + 1,
            'col': int(c),
            'row': int(r),
            'value': float(fingerprint_2d[r, c])
        })
    
    logger.debug(f"Extracted top {len(cells)} active cells, "
                f"highest value: {cells[0]['value']:.6f}" if cells else "No active cells")
    
    return cells


def get_top_overlapped_cells(grid1: np.ndarray, grid2: np.ndarray, top_n: int = 20) -> List[Dict]:
    """Extract top-N overlapped cells (minimum of both grids)."""
    overlap = np.minimum(grid1, grid2)
    nonzero_coords = np.argwhere(overlap > 0)

    if len(nonzero_coords) == 0:
        return []

    values = overlap[nonzero_coords[:, 0], nonzero_coords[:, 1]]
    sorted_indices = np.argsort(values)[::-1][:top_n]

    top_cells = []
    for idx in sorted_indices:
        y, x = nonzero_coords[idx]
        overlap_val = values[idx]
        top_cells.append({
            "x": int(x),
            "y": int(y),
            "overlap_activation": float(overlap_val),
            "doc1_activation": float(grid1[y, x]),
            "doc2_activation": float(grid2[y, x])
        })

    return top_cells


def create_document_visualizer(
    fingerprint: np.ndarray,
    doc_id: str,
    doc_text: str,
    metadata: Dict,
    grid_size: int,
    grid_borders: bool = True,
    border_color: str = "lightgray",
    border_width: float = 1.0,
    max_shapes: int = 5000,
    use_morton: bool = True,    
) -> go.Figure:
    """
    Create comprehensive Customtext fingerprint visualization.
    
    Generates a 2×3 dashboard with:
    - Row 1: Customtext matrix (with 4×4 borders) | Spatial heatmap | Metrics
    - Row 2: Activation histogram | Spatial density | Top active cells
    
    Args:
        fingerprint: 1D flattened fingerprint vector (TF-IDF weighted)
        doc_id: Customtext identifier
        doc_text: Customtext
        metadata: Dict with grid_size, num_docs, etc.
        grid_size: Dimension of the square grid
        grid_borders: If True, draw 4×4 block borders on matrix view
        border_color: Color for 4×4 block borders
        border_width: Width of 4×4 block borders
        max_shapes: Maximum number of shapes to draw (safety limit)
        
    Returns:
        Plotly Figure object
    """
    logger.info(f"Creating visualization for Customtext: {doc_id}")
    
    # Reconstruct 2D grid from flattened vector
    fp_2d = inverse_flatten(fingerprint, grid_size, use_morton)
    # Calculate statistics
    active_bits = np.sum(fingerprint > 0)
    total_bits = len(fingerprint)
    sparsity = 1 - (active_bits / total_bits)
    max_activation = np.max(fingerprint)
    mean_activation = np.mean(fingerprint[fingerprint > 0]) if active_bits > 0 else 0
    
    # Log data characteristics for debugging
    logger.info(f"Active cells: {active_bits} / {total_bits} ({active_bits/total_bits:.2%})")
    if active_bits > 0:
        logger.info(f"Value range: {fingerprint[fingerprint > 0].min():.6f} → {max_activation:.6f}")
    logger.debug(f"Sparsity: {sparsity:.2%}, Mean activation: {mean_activation:.6f}")
    
    # Get top cells
    top_cells = get_top_active_cells(fp_2d, top_n=10)
    
    # Normalize for display (fixes black square issue)
    fp_display = normalize_for_display(fp_2d)
    
    # Mask zeros so they appear as background (white), not black
    # np.nan values in Plotly heatmaps render as transparent
    fp_masked = np.where(fp_2d > 0, fp_display, np.nan)
    logger.debug(f"Masked {np.isnan(fp_masked).sum()} inactive cells for display")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Customtext Fingerprint Matrix (4×4 Grid)',
            'Spatial Activation Heatmap',
            'Customtext Metrics',
            'Activation Distribution',
            'Spatial Density Map',
            'Top 10 Active Cells'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'table'}],
            [{'type': 'histogram'}, {'type': 'heatmap'}, {'type': 'table'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # ------------------------------------------------------------------------
    # Row 1, Col 1: Customtext Matrix View with 4×4 block borders
    # ------------------------------------------------------------------------
    # Uses discrete colorscale and cell gaps for matrix effect
    fig.add_trace(
        go.Heatmap(
            z=fp_masked,
            colorscale=[
                [0, 'white'],        # No activation
                [0.001, 'lightblue'], # Minimal activation
                [0.2, 'blue'],       # Low activation
                [0.5, 'purple'],     # Medium activation
                [0.8, 'red'],        # High activation
                [1, 'darkred']       # Maximum activation
            ],
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title="Activation",
                x=0.30,
                len=0.4,
                y=0.75
            ),
            hovertemplate='Cell: (%{x}, %{y})<br>Activation: %{z:.4f}<extra></extra>',
            xgap=1,  # Add 1px gap between cells for grid effect
            ygap=1
        ),
        row=1, col=1
    )
    
    # Draw 4×4 block borders on matrix view
    shape_count = 0
    
    if grid_borders:
        block_size = 4
        num_blocks = grid_size // block_size
        
        logger.debug(f"Drawing 4×4 block borders ({num_blocks}×{num_blocks} blocks)")
        
        # Draw vertical block borders
        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                logger.warning(f"Reached max_shapes limit ({max_shapes}), skipping remaining borders")
                break
            
            x_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=x_pos, y0=-0.5,
                x1=x_pos, y1=grid_size - 0.5,
                line=dict(color=border_color, width=border_width),
                layer="above",
                xref="x", yref="y"  # Reference subplot 1 (row=1, col=1)
            )
            shape_count += 1
        
        # Draw horizontal block borders
        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            
            y_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=-0.5, y0=y_pos,
                x1=grid_size - 0.5, y1=y_pos,
                line=dict(color=border_color, width=border_width),
                layer="above",
                xref="x", yref="y"  # Reference subplot 1 (row=1, col=1)
            )
            shape_count += 1
        
        logger.debug(f"Drew {shape_count} block border lines")
    
    # ------------------------------------------------------------------------
    # Row 1, Col 2: Spatial Heatmap (smoothed with Gaussian filter)
    # ------------------------------------------------------------------------
    # Apply gaussian to normalized values, not raw tiny floats
    smoothed = gaussian_filter(fp_display, sigma=1.5)
    smoothed_masked = np.where(smoothed > 1e-6, smoothed, np.nan)
    
    fig.add_trace(
        go.Heatmap(
            z=smoothed_masked,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(
                title="Density",
                x=0.63,
                len=0.4,
                y=0.75
            ),
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Density: %{z:.4f}<extra></extra>',
            xgap=0,  # No gaps for smooth heatmap
            ygap=0
        ),
        row=1, col=2
    )
    
    # ------------------------------------------------------------------------
    # Row 1, Col 3: Metrics Panel
    # ------------------------------------------------------------------------
    metrics_data = [
        ['Customtext ID', doc_id],
        ['Grid Size', f"{grid_size}×{grid_size}"],
        ['Vector Size', str(total_bits)],
        ['Active Bits', str(active_bits)],
        ['Sparsity', f"{sparsity:.2%}"],
        ['Max Activation', f"{max_activation:.4f}"],
        ['Mean Activation', f"{mean_activation:.4f}"],
        ['Total Docs', str(metadata.get('num_docs', 'N/A'))]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=list(zip(*metrics_data)),
                fill_color='white',
                align='left',
                font=dict(size=11),
                height=25
            )
        ),
        row=1, col=3
    )
    
    # ------------------------------------------------------------------------
    # Row 2, Col 1: Activation Histogram (only non-zero values)
    # ------------------------------------------------------------------------
    active_values = fingerprint[fingerprint > 0]
    
    if len(active_values) > 0:
        fig.add_trace(
            go.Histogram(
                x=active_values,
                nbinsx=50,
                marker_color='steelblue',
                hovertemplate='Value Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        logger.debug(f"Histogram created with {len(active_values)} active values")
    else:
        logger.warning("No active values to plot in histogram")
    
    # ------------------------------------------------------------------------
    # Row 2, Col 2: Spatial Density (binary activation map)
    # ------------------------------------------------------------------------
    # Shows which cells are active (1) vs inactive (0)
    binary_map = (fp_2d > 0).astype(float)
    fig.add_trace(
        go.Heatmap(
            z=binary_map,
            colorscale=[[0, 'white'], [1, 'darkblue']],
            showscale=False,
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Active: %{z}<extra></extra>',
            xgap=0,
            ygap=0
        ),
        row=2, col=2
    )
    
    # ------------------------------------------------------------------------
    # Row 2, Col 3: Top Active Cells Table
    # ------------------------------------------------------------------------
    if top_cells:
        top_cells_data = [
            [str(cell['rank']) for cell in top_cells],
            [f"({cell['col']}, {cell['row']})" for cell in top_cells],
            [f"{cell['value']:.4f}" for cell in top_cells]
        ]
    else:
        # Handle case with no active cells
        top_cells_data = [['N/A'], ['N/A'], ['N/A']]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Rank</b>', '<b>Position</b>', '<b>Value</b>'],
                fill_color='lightgreen',
                align='left',
                font=dict(size=11)
            ),
            cells=dict(
                values=top_cells_data,
                fill_color='white',
                align='left',
                font=dict(size=10),
                height=23
            )
        ),
        row=2, col=3
    )
    
    # ------------------------------------------------------------------------
    # Update layout
    # ------------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=f'Customtext Fingerprint Analysis {doc_id}: {doc_text[:128]}',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=900,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes for heatmaps
    # Matrix view (row=1, col=1) needs square aspect ratio
    fig.update_xaxes(
        title_text="X Coordinate",
        showticklabels=False,
        constrain="domain",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Y Coordinate",
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
        row=1, col=1
    )
    
    # Spatial heatmap (row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    
    # Histogram (row=2, col=1)
    fig.update_xaxes(title_text='Activation Value', row=2, col=1)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    
    # Binary density map (row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    
    logger.info("Visualization created successfully")
    return fig

# ============================================================================
# Single-Customtext Visualization (Plotly)
# ============================================================================

def visualize_single_customtext(
    doc_id: str,
    fingerprints_dir: Path,
    output_dir: Path,
    grid_size: int,
    use_morton: bool,
    threshold: float = 0.01,
    grid_borders: bool = True,
    border_color: str = "lightgray",
    border_width: float = 1.0,
    max_shapes: int = 5000,
    figure_width: int = 2000,
    figure_height: int = 600,
    colorscale: str = "Viridis",
    generate_html: bool = True,
    generate_png: bool = True,
    save_metadata: bool = True,
) -> None:
    """
    Generate interactive single-customtext visualization with Plotly including matrix view.
    
    Creates a comprehensive three-panel visualization for a single customtext fingerprint:
    1. Spatial Activation Heatmap: Standard continuous heatmap showing activation patterns
    2. Matrix View: Discrete cell-based view with 4×4 block borders for structure
    3. Activation Distribution: Histogram of activation values with statistics
    
    The function loads the customtext fingerprint from NPZ format, reconstructs the 2D grid
    (with optional Morton encoding), and generates interactive HTML and static PNG outputs.
    Additionally, it exports metadata and a JSON file listing all activated cells.
    
    Args:
        doc_id: The target customtext id to visualize (must exist in data/customtexts.txt)
        fingerprints_dir: Directory containing doc_fingerprints.npz and 
                         doc_fingerprints_meta.json files
        output_dir: Directory where visualization outputs will be saved
        grid_size: Size of the square grid (e.g., 128 for 128x128 grid)
        use_morton: If True, use Morton (Z-order) encoding for spatial reconstruction;
                   if False, use row-major ordering
        threshold: Activation threshold for determining "activated" cells in metadata.
                  Cells with activation > threshold will be counted and exported.
                  Default is 0.01.
        grid_borders: If True, draw 4×4 block borders on matrix view. Default is True.
        border_color: Color for 4×4 block borders. Default is "lightgray".
        border_width: Width of 4×4 block borders. Default is 1.0.
        max_shapes: Maximum number of shapes to draw (safety limit). Default is 5000.
        figure_width: Width of the output figure in pixels. Default is 2000.
        figure_height: Height of the output figure in pixels. Default is 600.
        colorscale: Colorscale for the spatial heatmap. Default is "Viridis".
        generate_html: If True, save HTML output. Default is True.
        generate_png: If True, save PNG output. Default is True.
        save_metadata: If True, save metadata JSON. Default is True.
    
    Returns:
        None. Outputs are saved to disk:
        - single_{doc_id}.html: Interactive Plotly visualization
        - single_{doc_id}.png: Static image (requires kaleido)
        - single_{doc_id}_meta.json: Visualization metadata and statistics
        - activated_cells_{doc_id}.json: List of activated cell coordinates and values
    
    Raises:
        FileNotFoundError: If fingerprints or metadata files are missing
        ValueError: If the doc_id is not found in the customtext id
    
    Example:
        >>> visualize_single_customtext(
        ...     doc_id="1",
        ...     fingerprints_dir=Path("data/fingerprints"),
        ...     output_dir=Path("outputs/customtext_viz"),
        ...     grid_size=128,
        ...     use_morton=True,
        ...     threshold=0.01
        ... )
    """
    logger.info(f"Loading fingerprint for doc_id: '{doc_id}'")

    # Define paths to fingerprint data and metadata
    npz_path = fingerprints_dir / "doc_fingerprints.npz"
    meta_path = fingerprints_dir / "doc_fingerprints_meta.json"
    ###

def get_document_by_id(file_path, target_id):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)

            if len(parts) != 2:
                continue

            doc_id = parts[0].strip()
            if doc_id == target_id:
                return parts[1].strip()
    return None


def save_visualization_metadata(
    output_path: Path,
    mode: str,
    doc_ids: Dict[str, str],
    doc_texts: Dict[str, str],
    grid_stats: Dict,
    config: Dict,
    top_cells: Dict,
) -> None:
    """Save visualization metadata and statistics to JSON."""
    metadata = {
        "mode": mode,
        "doc_ids": doc_ids,
        "doc_texts": doc_texts,
        "grid_stats": grid_stats,
        "config": config,
        "top_cells": top_cells,
        "generated_at": __import__('datetime').datetime.now().isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info(f"Saved metadata: {output_path}")


# ============================================================================
# Comparative Two-Customtext Visualization (Plotly)
# ============================================================================

def visualize_document_pair(
    doc_id1: str,
    doc_id2: str,
    doc_text1: str,
    doc_text2: str,
    doc_fingerprints: Dict,
    output_dir: Path,
    grid_size: int,
    use_morton: bool,
    threshold: float = 0.0,
    grid_borders: bool = True,
    border_color: str = "lightgray",
    border_width: float = 1.0,
    max_shapes: int = 5000,
    figure_width: int = 1800,
    figure_height: int = 1900,
    colorscale: str = "Blues",
    generate_html: bool = True,
    generate_png: bool = True,
    save_metadata: bool = True,
) -> None:
    """
    Generate interactive 9-panel comparative dashboard with Plotly.

    Creates a comprehensive three-row visualization comparing two document fingerprints:

    Row 1 (Matrix Views):
    - Panel 1: Matrix view of document 1 with 4x4 block borders
    - Panel 2: Matrix view of document 2 with 4x4 block borders
    - Panel 3: Matrix view of semantic overlap with 4x4 block borders

    Row 2 (Spatial Heatmaps):
    - Panel 4: Continuous heatmap of document 1
    - Panel 5: Continuous heatmap of document 2
    - Panel 6: Semantic overlap heatmap

    Row 3 (Analysis):
    - Panel 7: Difference map (doc1 - doc2)
    - Panel 8: Similarity metrics (cosine similarity, distance, overlap)
    - Panel 9: Activation distribution histograms
    """
    logger.info(f"Comparing fingerprints: '{doc_id1}' vs '{doc_id2}'")

    # Verify both documents exist
    missing = [d for d in [doc_id1, doc_id2] if d not in doc_fingerprints]
    if missing:
        logger.error(f"Document IDs not found: {missing}")
        available = list(doc_fingerprints.keys())[:10]
        logger.info(f"Available docs (first 10): {available}")
        raise ValueError(f"Document IDs not found in fingerprints: {missing}")

    # Extract both fingerprints (sparse to dense)
    fp1 = doc_fingerprints[doc_id1].toarray().flatten().astype(np.float32)
    fp2 = doc_fingerprints[doc_id2].toarray().flatten().astype(np.float32)

    logger.debug(f"Doc 1 '{doc_id1}': nnz={np.count_nonzero(fp1)}, "
                 f"max={fp1.max():.4f}")
    logger.debug(f"Doc 2 '{doc_id2}': nnz={np.count_nonzero(fp2)}, "
                 f"max={fp2.max():.4f}")

    # Reconstruct 2D grids from flattened fingerprints
    logger.info(f"Reconstructing 2D grids (size={grid_size}, morton={use_morton})")
    grid1 = inverse_flatten(fp1, grid_size, use_morton)
    grid2 = inverse_flatten(fp2, grid_size, use_morton)

    # Extract top active cells
    logger.debug("Extracting top active cells...")
    top_cells_1 = get_top_active_cells(grid1, top_n=20)
    top_cells_2 = get_top_active_cells(grid2, top_n=20)
    top_overlapped = get_top_overlapped_cells(grid1, grid2, top_n=20)

    # Identify activated cells for matrix views
    logger.debug(f"Identifying activated cells with threshold={threshold}...")
    activated_coords_1 = np.argwhere(grid1 > threshold)
    activated_coords_2 = np.argwhere(grid2 > threshold)

    # Compute overlap and difference grids
    logger.debug("Computing overlap and difference grids...")
    overlap = np.minimum(grid1, grid2)
    activated_coords_overlap = np.argwhere(overlap > threshold)
    diff = grid1 - grid2

    logger.info(f"Doc 1: {len(activated_coords_1)} activated cells (threshold={threshold})")
    logger.info(f"Doc 2: {len(activated_coords_2)} activated cells (threshold={threshold})")
    logger.info(f"Overlap: {len(activated_coords_overlap)} activated cells (threshold={threshold})")

    # Compute comprehensive statistics
    logger.debug("Computing statistics...")
    active1 = fp1[fp1 > 0]
    active2 = fp2[fp2 > 0]

    # Similarity metrics
    norm1, norm2 = np.linalg.norm(fp1), np.linalg.norm(fp2)
    cos_sim = np.dot(fp1, fp2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    grid_stats = {
        "doc1": {
            "doc_id": doc_id1,
            "active_cells": int(len(active1)),
            "activated_cells_threshold": int(len(activated_coords_1)),
            "max_activation": float(fp1.max()),
            "mean_activation": float(active1.mean()) if len(active1) > 0 else 0.0,
            "grid_nonzero": int(np.count_nonzero(grid1))
        },
        "doc2": {
            "doc_id": doc_id2,
            "active_cells": int(len(active2)),
            "activated_cells_threshold": int(len(activated_coords_2)),
            "max_activation": float(fp2.max()),
            "mean_activation": float(active2.mean()) if len(active2) > 0 else 0.0,
            "grid_nonzero": int(np.count_nonzero(grid2))
        },
        "comparison": {
            "cosine_similarity": float(cos_sim),
            "cosine_similarity_pct": float(cos_sim * 100),
            "overlap_cells": int(np.count_nonzero(overlap)),
            "overlap_activated_threshold": int(len(activated_coords_overlap)),
            "overlap_max": float(overlap.max()),
            "difference_range": [float(diff.min()), float(diff.max())],
            "euclidean_distance": float(np.linalg.norm(fp1 - fp2)),
            "threshold": float(threshold)
        }
    }

    logger.debug(f"Comparison statistics: {grid_stats['comparison']}")
    logger.info(f"Cosine similarity: {cos_sim:.4f} ({cos_sim * 100:.2f}%)")

    # ========================================================================
    # Create 9-panel interactive figure (3 rows x 3 columns)
    # ========================================================================
    logger.debug("Creating subplot structure...")
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            f'Matrix: "{doc_id1}"',
            f'Matrix: "{doc_id2}"',
            'Matrix: Overlap',
            f'Spatial: "{doc_id1}"',
            f'Spatial: "{doc_id2}"',
            'Spatial: Overlap',
            'Difference Map',
            'Similarity Metrics',
            'Activation Distribution',
            f'Top 10 Cells: "{doc_id1}"',
            f'Top 10 Cells: "{doc_id2}"',
            'Top 10 Overlapped Cells'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "table"}, {"type": "table"}, {"type": "table"}]
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
        row_heights=[0.27, 0.27, 0.23, 0.23],
        column_widths=[0.33, 0.33, 0.34]
    )

    shape_count = 0

    # ========================================================================
    # ROW 1: Matrix Views with 4x4 Block Borders
    # ========================================================================

    # Panel 1 (Row 1, Col 1): Matrix view of doc 1
    logger.debug("Adding Panel 1: Matrix view of doc 1...")
    fig.add_trace(
        go.Heatmap(
            z=grid1,
            colorscale=[
                [0, 'white'],
                [0.001, 'lightblue'],
                [0.2, 'blue'],
                [0.5, 'darkblue'],
                [0.8, 'navy'],
                [1, 'midnightblue']
            ],
            zmin=0, zmax=1,
            colorbar=dict(title="Activation", x=0.29, len=0.20, y=0.89),
            hovertemplate='Cell: (%{x}, %{y})<br>Activation: %{z:.4f}<extra></extra>',
            xgap=1, ygap=1
        ),
        row=1, col=1
    )

    # Draw 4x4 block borders for doc 1
    if grid_borders:
        logger.debug("Drawing block borders for doc 1...")
        block_size = 4
        num_blocks = grid_size // block_size

        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            x_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=x_pos, y0=-0.5, x1=x_pos, y1=grid_size - 0.5,
                line=dict(color=border_color, width=border_width),
                layer="above", xref="x", yref="y"
            )
            shape_count += 1

        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            y_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=-0.5, y0=y_pos, x1=grid_size - 0.5, y1=y_pos,
                line=dict(color=border_color, width=border_width),
                layer="above", xref="x", yref="y"
            )
            shape_count += 1

    # Panel 2 (Row 1, Col 2): Matrix view of doc 2
    logger.debug("Adding Panel 2: Matrix view of doc 2...")
    fig.add_trace(
        go.Heatmap(
            z=grid2,
            colorscale=[
                [0, 'white'],
                [0.001, 'lightyellow'],
                [0.2, 'orange'],
                [0.5, 'darkorange'],
                [0.8, 'orangered'],
                [1, 'darkred']
            ],
            zmin=0, zmax=1,
            colorbar=dict(title="Activation", x=0.63, len=0.20, y=0.89),
            hovertemplate='Cell: (%{x}, %{y})<br>Activation: %{z:.4f}<extra></extra>',
            xgap=1, ygap=1
        ),
        row=1, col=2
    )

    # Draw 4x4 block borders for doc 2
    if grid_borders and shape_count < max_shapes:
        block_size = 4
        num_blocks = grid_size // block_size

        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            x_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=x_pos, y0=-0.5, x1=x_pos, y1=grid_size - 0.5,
                line=dict(color=border_color, width=border_width),
                layer="above", xref="x2", yref="y2"
            )
            shape_count += 1

        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            y_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=-0.5, y0=y_pos, x1=grid_size - 0.5, y1=y_pos,
                line=dict(color=border_color, width=border_width),
                layer="above", xref="x2", yref="y2"
            )
            shape_count += 1

    # Panel 3 (Row 1, Col 3): Matrix view of overlap
    logger.debug("Adding Panel 3: Matrix view of overlap...")
    fig.add_trace(
        go.Heatmap(
            z=overlap,
            colorscale=[
                [0, 'white'],
                [0.001, 'lavender'],
                [0.2, 'mediumpurple'],
                [0.5, 'purple'],
                [0.8, 'indigo'],
                [1, 'darkviolet']
            ],
            zmin=0, zmax=1,
            colorbar=dict(title="Overlap", x=0.97, len=0.20, y=0.89),
            hovertemplate='Cell: (%{x}, %{y})<br>Overlap: %{z:.4f}<extra></extra>',
            xgap=1, ygap=1
        ),
        row=1, col=3
    )

    # Draw 4x4 block borders for overlap
    if grid_borders and shape_count < max_shapes:
        block_size = 4
        num_blocks = grid_size // block_size

        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            x_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=x_pos, y0=-0.5, x1=x_pos, y1=grid_size - 0.5,
                line=dict(color=border_color, width=border_width),
                layer="above", xref="x3", yref="y3"
            )
            shape_count += 1

        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            y_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=-0.5, y0=y_pos, x1=grid_size - 0.5, y1=y_pos,
                line=dict(color=border_color, width=border_width),
                layer="above", xref="x3", yref="y3"
            )
            shape_count += 1

    logger.debug(f"Total shapes drawn: {shape_count}")

    # ========================================================================
    # ROW 2: Continuous Spatial Heatmaps
    # ========================================================================

    # Panel 4 (Row 2, Col 1): Continuous heatmap of doc 1
    logger.debug("Adding Panel 4: Continuous heatmap of doc 1...")
    fig.add_trace(
        go.Heatmap(
            z=grid1, colorscale='Blues',
            zmin=0, zmax=1,
            colorbar=dict(title="Activation", x=0.29, len=0.20, y=0.61),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Activation: %{z:.4f}<extra></extra>',
            xgap=0, ygap=0
        ),
        row=2, col=1
    )

    # Panel 5 (Row 2, Col 2): Continuous heatmap of doc 2
    logger.debug("Adding Panel 5: Continuous heatmap of doc 2...")
    fig.add_trace(
        go.Heatmap(
            z=grid2, colorscale='Oranges',
            zmin=0, zmax=1,
            colorbar=dict(title="Activation", x=0.63, len=0.20, y=0.61),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Activation: %{z:.4f}<extra></extra>',
            xgap=0, ygap=0
        ),
        row=2, col=2
    )

    # Panel 6 (Row 2, Col 3): Continuous heatmap of overlap
    logger.debug("Adding Panel 6: Continuous heatmap of overlap...")
    fig.add_trace(
        go.Heatmap(
            z=overlap, colorscale='Purples',
            zmin=0, zmax=1,
            colorbar=dict(title="Overlap", x=0.97, len=0.20, y=0.61),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Overlap: %{z:.4f}<extra></extra>',
            xgap=0, ygap=0
        ),
        row=2, col=3
    )

    # ========================================================================
    # ROW 3: Analysis Panels
    # ========================================================================

    # Panel 7 (Row 3, Col 1): Difference map
    logger.debug("Adding Panel 7: Difference map...")
    # Auto-scale z range around 0 so small differences are visible
    max_abs_diff = max(abs(diff.min()), abs(diff.max()))
    if max_abs_diff == 0:
        z_min, z_max = -1, 1
    else:
        z_min, z_max = -max_abs_diff, max_abs_diff
    fig.add_trace(
        go.Heatmap(
            z=diff, colorscale='RdBu',
            # z=diff, colorscale='RdBu_r',
            zmid=0, zmin=z_min, zmax=z_max,
            colorbar=dict(title="Difference", x=0.29, len=0.20, y=0.34),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Difference: %{z:.4f}<extra></extra>',
            xgap=0, ygap=0
        ),
        row=3, col=1
    )

    # Panel 8 (Row 3, Col 2): Similarity metrics
    logger.debug("Adding Panel 8: Similarity metrics...")
    metrics_text = (
        f"<b>Cosine Similarity</b><br><br>"
        f"<span style='font-size:32px'>{cos_sim * 100:.2f}%</span><br><br>"
        f"<b>Euclidean Distance</b><br>"
        f"{grid_stats['comparison']['euclidean_distance']:.4f}<br><br>"
        f"<b>Overlap Cells (>{threshold})</b><br>"
        f"{grid_stats['comparison']['overlap_activated_threshold']}<br><br>"
        f"<b>Total Overlap Cells</b><br>"
        f"{grid_stats['comparison']['overlap_cells']}"
    )

    fig.add_annotation(
        text=metrics_text,
        xref="x8", yref="y8",
        x=0.5, y=0.5,
        xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=14),
        align='center'
    )

    # Panel 9 (Row 3, Col 3): Histograms
    logger.debug("Adding Panel 9: Activation histograms...")
    if len(active1) > 0:
        fig.add_trace(
            go.Histogram(
                x=active1, nbinsx=30, name=f"Doc {doc_id1}",
                marker=dict(color='blue', opacity=0.6, line=dict(color='black', width=1)),
                hovertemplate='Activation: %{x:.4f}<br>Count: %{y}<extra></extra>',
                showlegend=True
            ),
            row=3, col=3
        )

    if len(active2) > 0:
        fig.add_trace(
            go.Histogram(
                x=active2, nbinsx=30, name=f"Doc {doc_id2}",
                marker=dict(color='darkorange', opacity=0.6, line=dict(color='black', width=1)),
                hovertemplate='Activation: %{x:.4f}<br>Count: %{y}<extra></extra>',
                showlegend=True
            ),
            row=3, col=3
        )

    # ========================================================================
    # ROW 4: Top Active Cells Tables
    # ========================================================================

    # Panel 10 (Row 4, Col 1): Top 10 active cells for doc 1
    logger.debug("Adding Panel 10: Top 10 cells for doc 1...")
    top10_1 = top_cells_1[:10]
    if top10_1:
        table_data_1 = [
            [str(c['rank']) for c in top10_1],
            [f"({c['col']}, {c['row']})" for c in top10_1],
            [f"{c['value']:.4f}" for c in top10_1]
        ]
    else:
        table_data_1 = [['N/A'], ['N/A'], ['N/A']]
    fig.add_trace(
        go.Table(
            header=dict(values=['<b>Rank</b>', '<b>Position</b>', '<b>Value</b>'],
                        fill_color='lightblue', align='left', font=dict(size=10)),
            cells=dict(values=table_data_1, fill_color='white', align='left',
                       font=dict(size=9), height=22)
        ),
        row=4, col=1
    )

    # Panel 11 (Row 4, Col 2): Top 10 active cells for doc 2
    logger.debug("Adding Panel 11: Top 10 cells for doc 2...")
    top10_2 = top_cells_2[:10]
    if top10_2:
        table_data_2 = [
            [str(c['rank']) for c in top10_2],
            [f"({c['col']}, {c['row']})" for c in top10_2],
            [f"{c['value']:.4f}" for c in top10_2]
        ]
    else:
        table_data_2 = [['N/A'], ['N/A'], ['N/A']]
    fig.add_trace(
        go.Table(
            header=dict(values=['<b>Rank</b>', '<b>Position</b>', '<b>Value</b>'],
                        fill_color='lightsalmon', align='left', font=dict(size=10)),
            cells=dict(values=table_data_2, fill_color='white', align='left',
                       font=dict(size=9), height=22)
        ),
        row=4, col=2
    )

    # Panel 12 (Row 4, Col 3): Top 10 overlapped cells
    logger.debug("Adding Panel 12: Top 10 overlapped cells...")
    top10_overlap = top_overlapped[:10]
    if top10_overlap:
        table_data_overlap = [
            [str(i + 1) for i in range(len(top10_overlap))],
            [f"({c['x']}, {c['y']})" for c in top10_overlap],
            [f"{c['overlap_activation']:.4f}" for c in top10_overlap],
            [f"{c['doc1_activation']:.4f}" for c in top10_overlap],
            [f"{c['doc2_activation']:.4f}" for c in top10_overlap]
        ]
        overlap_headers = ['<b>Rank</b>', '<b>Position</b>', '<b>Overlap</b>',
                           f'<b>Doc {doc_id1}</b>', f'<b>Doc {doc_id2}</b>']
    else:
        table_data_overlap = [['N/A'], ['N/A'], ['N/A'], ['N/A'], ['N/A']]
        overlap_headers = ['<b>Rank</b>', '<b>Position</b>', '<b>Overlap</b>',
                           f'<b>Doc1</b>', f'<b>Doc2</b>']
    fig.add_trace(
        go.Table(
            header=dict(values=overlap_headers,
                        fill_color='thistle', align='left', font=dict(size=10)),
            cells=dict(values=table_data_overlap, fill_color='white', align='left',
                       font=dict(size=9), height=22)
        ),
        row=4, col=3
    )

    # ========================================================================
    # Update axes
    # ========================================================================
    logger.debug("Updating axes...")

    for row_num in [1, 2]:
        for col_num in [1, 2, 3]:
            subplot_num = (row_num - 1) * 3 + col_num
            xref = f"x{subplot_num}" if subplot_num > 1 else "x"
            yref = f"y{subplot_num}" if subplot_num > 1 else "y"
            fig.update_xaxes(title_text="X", row=row_num, col=col_num,
                             constrain="domain", showgrid=False)
            fig.update_yaxes(title_text="Y", row=row_num, col=col_num,
                             scaleanchor=xref, scaleratio=1,
                             constrain="domain", showgrid=False)

    fig.update_xaxes(title_text="X", row=3, col=1, constrain="domain", showgrid=False)
    fig.update_yaxes(title_text="Y", row=3, col=1, scaleanchor="x7", scaleratio=1,
                     constrain="domain", showgrid=False)

    fig.update_xaxes(visible=False, row=3, col=2)
    fig.update_yaxes(visible=False, row=3, col=2)

    fig.update_xaxes(title_text="Activation", row=3, col=3, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Count", row=3, col=3, showgrid=True, gridcolor='lightgray')

    # Row 4 tables: hide axes
    for col_num in [1, 2, 3]:
        fig.update_xaxes(visible=False, row=4, col=col_num)
        fig.update_yaxes(visible=False, row=4, col=col_num)

    # ========================================================================
    # Layout
    # ========================================================================
    logger.debug("Updating layout...")
    fig.update_layout(
        title=dict(
            text=f'<b>Comparative Analysis: Doc "{doc_id1}" vs Doc "{doc_id2}"</b>',
            x=0.5, xanchor='center', font=dict(size=18)
        ),
        height=figure_height,
        width=figure_width,
        showlegend=True,
        legend=dict(x=0.85, y=0.34, bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='lightgray', borderwidth=1),
        template='plotly_white',
        autosize=False,
        margin=dict(l=60, r=60, t=100, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # ========================================================================
    # Save outputs
    # ========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name1 = doc_id1.replace(' ', '_').replace('/', '_')
    safe_name2 = doc_id2.replace(' ', '_').replace('/', '_')
    output_path = output_dir / f"compare_{safe_name1}_vs_{safe_name2}"

    if generate_html:
        html_path = output_path.with_suffix('.html')
        logger.debug(f"Saving HTML to {html_path}...")
        fig.write_html(
            str(html_path),
            config={'displayModeBar': True, 'responsive': False, 'displaylogo': False}
        )
        logger.info(f"Saved interactive HTML: {html_path}")

    if generate_png:
        png_path = output_path.with_suffix('.png')
        logger.debug(f"Saving PNG to {png_path}...")
        try:
            fig.write_image(str(png_path), width=figure_width, height=figure_height)
            logger.info(f"Saved static PNG: {png_path}")
        except Exception as e:
            logger.warning(f"Could not save PNG (kaleido required): {e}")

    if save_metadata:
        meta_out_path = output_dir / f"{output_path.stem}_meta.json"
        save_visualization_metadata(
            output_path=meta_out_path,
            mode="comparative",
            doc_ids={"doc1": doc_id1, "doc2": doc_id2},
            doc_texts={"doc1": doc_text1, "doc2": doc_text2},
            grid_stats=grid_stats,
            config={
                "grid_size": grid_size,
                "use_morton": use_morton,
                "threshold": threshold,
                "grid_borders": grid_borders,
                "border_color": border_color,
                "border_width": border_width,
                "max_shapes": max_shapes,
                "figure_width": figure_width,
                "figure_height": figure_height,
                "colorscale": colorscale
            },
            top_cells={"doc1": top_cells_1, "doc2": top_cells_2}
        )

    logger.info(f"Visualization complete for '{doc_id1}' vs '{doc_id2}'")


def main():
    """
    Main entry point for customtext fingerprint visualization.
    
    Usage:
        python customtext_visualizer.py --run-dir outputs/20260423_023143/ \
                                 --doc-id doc_001 \
                                 --output visualizations/
    """
    parser = argparse.ArgumentParser(
        description='Visualize customtext semantic fingerprints'
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        required=True,
        help='Path to run output directory (e.g., outputs/20260423_023143/)'
    )
    # parser.add_argument(
    #     '--doc-id',
    #     type=str,
    #     required=True,
    #     help='Customtext ID to visualize'
    # )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output folder path for visualizations'
    )
    # Border styling
    parser.add_argument(
        '--no-grid-borders',
        action='store_true',
        help='Disable 4×4 block borders on matrix view'
    )
    parser.add_argument(
        '--border-color',
        type=str,
        default='lightgray',
        help='Color for 4×4 block borders (default: lightgray)'
    )
    parser.add_argument(
        '--border-width',
        type=float,
        default=1.0,
        help='Width of 4×4 block borders (default: 1.0)'
    )

    ###
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--doc-id', type=str, help='Single customtext to visualize')
    group.add_argument('--doc-id1', type=str, help='First customtext for comparison')
    parser.add_argument('--doc-id2', type=str, help='Second customtext for comparison')

    parser.add_argument('--grid-size', type=int, default=128, help='Grid dimension (default: 128)')

    # Activation threshold
    parser.add_argument('--threshold', type=float, default=0.0, 
                        help='Activation threshold (default: 0.0)')
    
    # Performance
    parser.add_argument('--max-shapes', type=int, default=300,
                        help='Maximum shapes to render, prevents hanging (default: 300)')
    
    # Figure dimensions
    parser.add_argument('--width', type=int, default=1800,
                        help='Figure width in pixels (default: 1800)')
    parser.add_argument('--height', type=int, default=1500,
                        help='Figure height in pixels (default: 1500)')
    
    # Color scheme
    parser.add_argument('--colorscale', type=str, default='Blues',
                        help='Plotly colorscale name (default: Blues)')
    
    # Output formats
    parser.add_argument('--no-html', action='store_true',
                        help='Skip HTML output generation')
    parser.add_argument('--no-png', action='store_true',
                        help='Skip PNG output generation')
    parser.add_argument('--no-metadata', action='store_true',
                        help='Skip metadata JSON generation')
    ####
    
    args = parser.parse_args()

    if args.doc_id1 and not args.doc_id2:
        parser.error("--doc-id2 is required when using --doc-id1")
    
    logger.info("=" * 60)
    logger.info("Customtext Fingerprint Visualization Tool (Plotly)")
    logger.info("=" * 60)
    logger.debug(f"Run directory: {args.run_dir}")
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {args.output}")
    # Construct paths from run directory
    doc_fp_dir = args.run_dir / 'customtext_fingerprints'
    
    # Validate directories exist
    if not doc_fp_dir.exists():
        logger.error(f"Customtext fingerprints directory not found: {doc_fp_dir}")
        print(f"Error: Customtext fingerprints directory not found: {doc_fp_dir}")
        return
    
    # Load customtext fingerprints using lib function
    logger.info(f"Loading customtext fingerprints from {doc_fp_dir}...")
    print(f"Loading customtext fingerprints from {doc_fp_dir}...")
    logger.debug(f"Output directory: {args.output}")
    logger.info(f"Run directory: {args.run_dir}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Grid size: {args.grid_size}")
    # logger.info(f"Encoding: {'Row-major' if not args.morton else 'Morton (Z-order)'}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Grid borders: {'Disabled' if args.no_grid_borders else f'Enabled ({args.border_color}, width={args.border_width})'}")
    logger.info(f"Figure size: {args.width}×{args.height}")
    logger.info(f"Colorscale: {args.colorscale}")
    logger.info(f"Max shapes: {args.max_shapes}")
    logger.info("=" * 60)
    
    
    
    try:
        doc_fingerprints, metadata = load_document_fingerprints(doc_fp_dir)
        use_morton = metadata.get('use_morton', False)   # fallback False (row-major) for old runs
        grid_size = metadata['grid_size']
        logger.info(f"Loaded {len(doc_fingerprints)} customtext fingerprints")
        logger.debug(f"Metadata: {metadata}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load customtext fingerprints: {e}")
        print(f"Error loading customtext fingerprints: {e}")
        return
    
    try:
        if args.doc_id:
            logger.info("Mode: Single-customtext visualization")
            logger.info(f"Starting customtext visualization for {args.doc_id}")
            # visualize_single_customtext(
            #     doc_id=args.doc_id,
            #     fingerprints_dir=doc_fp_dir,
            #     output_dir=args.output,
            #     grid_size=args.grid_size,
            #     use_morton=args.morton,
            #     threshold=args.threshold,
            #     grid_borders=not args.no_grid_borders,
            #     border_color=args.border_color,
            #     border_width=args.border_width,
            #     max_shapes=args.max_shapes,
            #     figure_width=args.width,
            #     figure_height=args.height,
            #     colorscale=args.colorscale,
            #     generate_html=not args.no_html,
            #     generate_png=not args.no_png,
            #     save_metadata=not args.no_metadata,
            # )
            
            # Check if customtext exists
            if args.doc_id not in doc_fingerprints:
                available_docs = list(doc_fingerprints.keys())[:10]
                logger.error(f"Customtext ID '{args.doc_id}' not found in fingerprints")
                logger.debug(f"Available customtexts (first 10): {available_docs}")
                print(f"Error: Customtext ID '{args.doc_id}' not found.")
                print(f"Available customtexts: {available_docs}...")
                return
            
            # Get fingerprint (convert from sparse to dense)
            doc_fp_sparse = doc_fingerprints[args.doc_id]
            fingerprint = doc_fp_sparse.toarray().flatten()
            
            logger.debug(f"Converted sparse fingerprint to dense array: shape {fingerprint.shape}")
            
            grid_size = metadata['grid_size']

            text_dir = "data\\customtexts.txt"
            doc_text = get_document_by_id(text_dir, args.doc_id)
            
            print(f"Visualizing customtext {args.doc_id}: {doc_text}")
            print(f"Grid size: {grid_size}×{grid_size}")
            print(f"Total Customtexts: {metadata['num_docs']}")
            
            # Create visualization
            fig = create_document_visualizer(
                fingerprint,
                args.doc_id,
                doc_text,
                metadata,
                grid_size,
                grid_borders=not args.no_grid_borders,
                border_color=args.border_color,
                border_width=args.border_width,
                use_morton= use_morton
            )
            
            # Save HTML
            html_path = args.output / f"{args.doc_id}_visualization.html"
            fig.write_html(str(html_path))
            logger.info(f"HTML visualization saved to {html_path}")
            print(f"HTML saved to {html_path}")
            
            # Save PNG
            try:
                png_path = args.output / f"{args.doc_id}_visualization.png"
                fig.write_image(str(png_path), width=1800, height=900)
                logger.info(f"PNG visualization saved to {png_path}")
                print(f"PNG saved to {png_path}")
            except Exception as e:
                logger.warning(f"Failed to save PNG (kaleido may not be installed): {e}")
                print(f"Warning: Could not save PNG. Install kaleido for image export.")
            
            print(f"\nVisualization complete! Files saved in {args.output}")
            logger.info("=" * 60)
            logger.info("Visualization completed successfully.")
            logger.info("=" * 60)


        else:
            logger.info("Mode: Comparative two-customtext visualization")
            logger.info(f"Starting customtext visualization for {args.doc_id1} and {args.doc_id2}")

            if args.doc_id1 not in doc_fingerprints:
                available_docs = list(doc_fingerprints.keys())[:10]
                logger.error(f"Customtext ID '{args.doc_id1}' not found in fingerprints")
                print(f"Error: Customtext ID '{args.doc_id1}' not found.")
                print(f"Available customtexts: {available_docs}...")
                return

            if args.doc_id2 not in doc_fingerprints:
                available_docs = list(doc_fingerprints.keys())[:10]
                logger.error(f"Customtext ID '{args.doc_id2}' not found in fingerprints")
                print(f"Error: Customtext ID '{args.doc_id2}' not found.")
                print(f"Available customtexts: {available_docs}...")
                return

            grid_size = metadata['grid_size']
            use_morton = metadata.get('use_morton', False)

            text_dir = "data\\customtexts.txt"
            doc_text1 = get_document_by_id(text_dir, args.doc_id1)
            doc_text2 = get_document_by_id(text_dir, args.doc_id2)

            print(f"Comparing customtext {args.doc_id1}: {doc_text1}")
            print(f"                  vs {args.doc_id2}: {doc_text2}")
            print(f"Grid size: {grid_size}×{grid_size}")
            print(f"Total Customtexts: {metadata['num_docs']}")

            visualize_document_pair(
                doc_id1=args.doc_id1,
                doc_id2=args.doc_id2,
                doc_text1=doc_text1,
                doc_text2=doc_text2,
                doc_fingerprints=doc_fingerprints,
                output_dir=args.output,
                grid_size=grid_size,
                use_morton=use_morton,
                threshold=args.threshold,
                grid_borders=not args.no_grid_borders,
                border_color=args.border_color,
                border_width=args.border_width,
                max_shapes=args.max_shapes,
                figure_width=args.width,
                figure_height=args.height,
                colorscale=args.colorscale,
                generate_html=not args.no_html,
                generate_png=not args.no_png,
                save_metadata=not args.no_metadata,
            )

            print(f"\nComparison complete! Files saved in {args.output}")
            logger.info("=" * 60)
            logger.info("Comparison visualization completed successfully.")
            logger.info("=" * 60)
    
    except Exception as e:
        logger.exception(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
