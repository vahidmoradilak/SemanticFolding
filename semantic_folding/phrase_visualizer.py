#!/usr/bin/env python3
"""
Phrase Fingerprint Visualization Tool (Enhanced with Plotly)
=============================================================
Visualizes semantic folding fingerprints with interactive, high-quality graphics.

Features:
- Interactive zoom/pan with Plotly
- Crisp pixel-perfect rendering
- Hover tooltips showing exact coordinates and values
- Professional color schemes
- Export to HTML and PNG
- Top-N overlapped cells in metadata for debugging

Author: Mojtaba Banaie
Date: 2026-04-21 (1405/02/01 Jalali)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from lib import get_logger
logger = get_logger("phrase_visualizer")


# ============================================================================
# Morton Encoding Reversal (Z-order curve decoding)
# ============================================================================

def _compact_bits(value: int) -> int:
    """Inverse of _spread_bits: extracts every other bit to un-zip Morton code."""
    value &= 0x55555555
    value = (value ^ (value >> 1)) & 0x33333333
    value = (value ^ (value >> 2)) & 0x0F0F0F0F
    value = (value ^ (value >> 4)) & 0x00FF00FF
    value = (value ^ (value >> 8)) & 0x0000FFFF
    return value


def morton_to_xy(morton_code: int) -> Tuple[int, int]:
    """Convert 1D Morton index back to 2D (x, y) coordinates."""
    x = _compact_bits(morton_code)
    y = _compact_bits(morton_code >> 1)
    logger.debug(f"Morton {morton_code} -> (x={x}, y={y})")
    return x, y


def inverse_flatten(fp: np.ndarray, grid_size: int, use_morton: bool) -> np.ndarray:
    """Reconstruct 2D spatial grid from 1D fingerprint vector."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    active_count = 0
    out_of_bounds = []
    
    for idx, val in enumerate(fp):
        if val > 0:
            if use_morton:
                x, y = morton_to_xy(idx)
            else:
                y, x = divmod(idx, grid_size)
            
            if x >= grid_size or y >= grid_size:
                out_of_bounds.append((idx, x, y, val))
                logger.warning(f"Out of bounds: idx={idx} -> ({x}, {y}), val={val:.4f}")
                continue
            
            grid[y, x] = val
            active_count += 1
            
            if active_count <= 5:
                logger.debug(f"Active cell {active_count}: idx={idx} -> ({x}, {y}), val={val:.4f}")
    
    if out_of_bounds:
        logger.error(f"Found {len(out_of_bounds)} out-of-bounds activations!")
        logger.debug(f"Out-of-bounds samples: {out_of_bounds[:10]}")
    
    logger.debug(f"Reconstructed grid: {active_count} active cells, "
                 f"{len(out_of_bounds)} out-of-bounds")
    
    return grid


def get_top_active_cells(grid: np.ndarray, top_n: int = 20) -> List[Dict]:
    """Extract top-N active cells with coordinates and values."""
    nonzero_coords = np.argwhere(grid > 0)
    if len(nonzero_coords) == 0:
        return []
    
    values = grid[nonzero_coords[:, 0], nonzero_coords[:, 1]]
    sorted_indices = np.argsort(values)[::-1][:top_n]
    
    top_cells = []
    for idx in sorted_indices:
        y, x = nonzero_coords[idx]
        val = values[idx]
        top_cells.append({
            "x": int(x),
            "y": int(y),
            "activation": float(val)
        })
    
    return top_cells


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
            "phrase1_activation": float(grid1[y, x]),
            "phrase2_activation": float(grid2[y, x])
        })
    
    return top_cells

# Add this new function after the get_top_overlapped_cells function

def visualize_matrix_view(
    phrase: str,
    fingerprints_dir: Path,
    output_dir: Path,
    grid_size: int,
    use_morton: bool,
    threshold: float = 0.01,
) -> None:
    """Generate matrix view with activated cells highlighted."""
    logger.info(f"Creating matrix view for phrase: '{phrase}'")
    
    npz_path = fingerprints_dir / "phrase_fingerprints.npz"
    meta_path = fingerprints_dir / "phrase_fingerprints_meta.json"
    
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing fingerprint files in {fingerprints_dir}")
    
    phrase_to_row = load_phrase_metadata(meta_path)
    
    if phrase not in phrase_to_row:
        raise ValueError(f"Phrase '{phrase}' not found in vocabulary.")
    
    fingerprints = np.load(npz_path)['fingerprints']
    idx = phrase_to_row[phrase]
    fp = fingerprints[idx]
    
    grid = inverse_flatten(fp, grid_size, use_morton)
    
    # Get activated cells above threshold
    activated_coords = np.argwhere(grid > threshold)
    activated_values = grid[activated_coords[:, 0], activated_coords[:, 1]]
    
    logger.info(f"Found {len(activated_coords)} activated cells (threshold={threshold})")
    
    # Create figure with matrix view
    fig = go.Figure()
    
    # Add heatmap with discrete colors for better cell visibility
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale=[
            [0, 'white'],
            [0.001, 'lightblue'],
            [0.2, 'blue'],
            [0.5, 'purple'],
            [0.8, 'red'],
            [1, 'darkred']
        ],
        zmin=0,
        zmax=1,
        colorbar=dict(title="Activation"),
        hovertemplate='Cell: (%{x}, %{y})<br>Activation: %{z:.4f}<extra></extra>',
        xgap=1,  # Add gaps between cells for matrix effect
        ygap=1
    ))
    
    # Add cell borders for activated cells
    if len(activated_coords) > 0:
        for coord, val in zip(activated_coords, activated_values):
            y, x = coord
            # Add rectangle border around activated cells
            fig.add_shape(
                type="rect",
                x0=x-0.5, y0=y-0.5,
                x1=x+0.5, y1=y+0.5,
                line=dict(color="black", width=1),
                layer="above"
            )
    
    # Add grid lines
    for i in range(grid_size + 1):
        # Vertical lines
        fig.add_shape(
            type="line",
            x0=i-0.5, y0=-0.5,
            x1=i-0.5, y1=grid_size-0.5,
            line=dict(color="lightgray", width=0.5),
            layer="below"
        )
        # Horizontal lines
        fig.add_shape(
            type="line",
            x0=-0.5, y0=i-0.5,
            x1=grid_size-0.5, y1=i-0.5,
            line=dict(color="lightgray", width=0.5),
            layer="below"
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Matrix View: "{phrase}"</b><br><sub>{len(activated_coords)} activated cells (threshold={threshold})</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title="X Coordinate",
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            showgrid=False
        ),
        yaxis=dict(
            title="Y Coordinate",
            showgrid=False
        ),
        height=800,
        width=800,
        template='plotly_white'
    )
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = phrase.replace(' ', '_').replace('/', '_')
    output_path = output_dir / f"matrix_{safe_filename}"
    
    fig.write_html(output_path.with_suffix('.html'))
    logger.info(f"Saved matrix HTML: {output_path.with_suffix('.html')}")
    
    try:
        fig.write_image(output_path.with_suffix('.png'), width=800, height=800, scale=2)
        logger.info(f"Saved matrix PNG: {output_path.with_suffix('.png')}")
    except Exception as e:
        logger.warning(f"PNG export failed: {e}")
    
    # Save activated cells list
    cells_data = {
        "phrase": phrase,
        "threshold": threshold,
        "total_activated": len(activated_coords),
        "activated_cells": [
            {"x": int(x), "y": int(y), "activation": float(val)}
            for (y, x), val in zip(activated_coords, activated_values)
        ]
    }
    
    cells_path = output_path.with_suffix('.json')
    with open(cells_path, 'w', encoding='utf-8') as f:
        json.dump(cells_data, f, indent=2)
    
    logger.info(f"Saved activated cells data: {cells_path}")

# ============================================================================
# Metadata Loader
# ============================================================================

def load_phrase_metadata(meta_path: Path) -> Dict[str, int]:
    """Load phrase-to-row mapping from Step 4 metadata file."""
    logger.debug(f"Loading metadata from: {meta_path}")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    phrase_to_row = meta_data.get("phrase_to_row")
    
    if phrase_to_row is None:
        if (isinstance(meta_data, dict) 
            and len(meta_data) > 0 
            and isinstance(next(iter(meta_data.values())), int)):
            phrase_to_row = meta_data
            logger.debug(f"Loaded flat-format metadata ({len(phrase_to_row)} phrases)")
        else:
            raise ValueError("Invalid metadata format.")
    else:
        logger.debug(f"Loaded nested-format metadata ({len(phrase_to_row)} phrases)")
    
    if not phrase_to_row:
        raise ValueError("Metadata is empty.")
    
    sample_phrases = list(phrase_to_row.items())[:5]
    logger.debug(f"Sample phrases: {sample_phrases}")
    
    return phrase_to_row


# ============================================================================
# Metadata Output Helper
# ============================================================================

def save_visualization_metadata(
    output_path: Path,
    mode: str,
    phrases: Dict[str, any],
    grid_stats: Dict[str, any],
    config: Dict[str, any],
    top_cells: Optional[Dict[str, List[Dict]]] = None,
) -> None:
    """Save visualization metadata for debugging and reproducibility."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "phrases": phrases,
        "grid_statistics": grid_stats,
        "configuration": config,
        "visualization_files": {
            "html": str(output_path.with_suffix('.html')),
            "png": str(output_path.with_suffix('.png'))
        }
    }
    
    if top_cells:
        metadata["top_active_cells"] = top_cells
    
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.debug(f"Saved visualization metadata to: {meta_path}")


# ============================================================================
# Single-Phrase Visualization (Plotly)
# ============================================================================

def visualize_single_phrase(
    phrase: str,
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
    Generate interactive single-phrase visualization with Plotly including matrix view.
    
    Creates a comprehensive three-panel visualization for a single phrase fingerprint:
    1. Spatial Activation Heatmap: Standard continuous heatmap showing activation patterns
    2. Matrix View: Discrete cell-based view with 4×4 block borders for structure
    3. Activation Distribution: Histogram of activation values with statistics
    
    The function loads the phrase fingerprint from NPZ format, reconstructs the 2D grid
    (with optional Morton encoding), and generates interactive HTML and static PNG outputs.
    Additionally, it exports metadata and a JSON file listing all activated cells.
    
    Args:
        phrase: The target phrase to visualize (must exist in metadata)
        fingerprints_dir: Directory containing phrase_fingerprints.npz and 
                         phrase_fingerprints_meta.json files
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
        - single_{phrase}.html: Interactive Plotly visualization
        - single_{phrase}.png: Static image (requires kaleido)
        - single_{phrase}_meta.json: Visualization metadata and statistics
        - activated_cells_{phrase}.json: List of activated cell coordinates and values
    
    Raises:
        FileNotFoundError: If fingerprints or metadata files are missing
        ValueError: If the phrase is not found in the vocabulary
    
    Example:
        >>> visualize_single_phrase(
        ...     phrase="machine learning",
        ...     fingerprints_dir=Path("data/fingerprints"),
        ...     output_dir=Path("outputs/viz"),
        ...     grid_size=128,
        ...     use_morton=True,
        ...     threshold=0.01
        ... )
    """
    logger.info(f"Loading fingerprint for phrase: '{phrase}'")
    
    # Define paths to fingerprint data and metadata
    npz_path = fingerprints_dir / "phrase_fingerprints.npz"
    meta_path = fingerprints_dir / "phrase_fingerprints_meta.json"
    
    # Validate that required files exist
    if not npz_path.exists():
        logger.error(f"Fingerprints file not found: {npz_path}")
        raise FileNotFoundError(f"Missing {npz_path}")
    
    if not meta_path.exists():
        logger.error(f"Metadata file not found: {meta_path}")
        raise FileNotFoundError(f"Missing {meta_path}")
    
    # Load phrase-to-row mapping from metadata
    phrase_to_row = load_phrase_metadata(meta_path)
    
    # Verify that the requested phrase exists in the vocabulary
    if phrase not in phrase_to_row:
        logger.error(f"Phrase '{phrase}' not found in metadata.")
        available = list(phrase_to_row.keys())[:10]
        logger.info(f"Available phrases (first 10): {available}")
        raise ValueError(f"Phrase '{phrase}' not found in vocabulary.")
    
    # Load fingerprint array and extract the specific phrase's fingerprint
    fingerprints = np.load(npz_path)['fingerprints']
    idx = phrase_to_row[phrase]
    fp = fingerprints[idx]  # 1D flattened fingerprint vector
    
    # Log fingerprint characteristics for debugging
    logger.debug(f"Fingerprint shape: {fp.shape}, dtype: {fp.dtype}")
    logger.debug(f"Fingerprint stats: min={fp.min():.4f}, max={fp.max():.4f}, "
                 f"mean={fp.mean():.4f}, nnz={np.count_nonzero(fp)}")
    
    # Reconstruct 2D grid from flattened fingerprint
    logger.info(f"Reconstructing 2D grid (size={grid_size}, morton={use_morton})")
    grid = inverse_flatten(fp, grid_size, use_morton)
    
    # Extract top 20 most active cells for metadata export
    top_cells = get_top_active_cells(grid, top_n=20)
    logger.debug(f"Top 20 active cells: {top_cells[:5]}...")
    
    # Identify all cells exceeding the activation threshold for metadata export
    activated_coords = np.argwhere(grid > threshold)
    activated_values = grid[activated_coords[:, 0], activated_coords[:, 1]]
    logger.info(f"Found {len(activated_coords)} activated cells (threshold={threshold})")
    
    # Compute comprehensive grid statistics
    active_vals = fp[fp > 0]  # Only non-zero activations
    grid_stats = {
        "total_cells": int(grid_size ** 2),  # Total grid capacity
        "active_cells": int(len(active_vals)),  # Cells with any activation > 0
        "activated_cells_threshold": int(len(activated_coords)),  # Cells > threshold
        "threshold": float(threshold),  # Threshold used for activation detection
        "sparsity_percent": float(100 * (1 - len(active_vals) / (grid_size ** 2))),
        "max_activation": float(fp.max()),
        "min_activation": float(fp.min()),
        "mean_activation_all": float(fp.mean()),  # Mean across all cells
        "mean_activation_active": float(active_vals.mean()) if len(active_vals) > 0 else 0.0,
        "std_activation_active": float(active_vals.std()) if len(active_vals) > 0 else 0.0,
        "grid_min": float(grid.min()),
        "grid_max": float(grid.max()),
        "grid_nonzero": int(np.count_nonzero(grid))
    }
    
    logger.debug(f"Grid statistics: {grid_stats}")
    
    # ========================================================================
    # Create interactive Plotly figure with 3 subplots
    # ========================================================================
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f'Spatial Activation: "{phrase}"',
            'Matrix View (4×4 Block Grid)',
            'Activation Distribution'
        ),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "histogram"}]],
        column_widths=[0.35, 0.35, 0.3],  # Distribute space across three panels
        horizontal_spacing=0.08
    )
    
    # ------------------------------------------------------------------------
    # Subplot 1: Standard spatial activation heatmap
    # ------------------------------------------------------------------------
    # Uses continuous colorscale for smooth gradient visualization
    # xgap=0, ygap=0 ensures crisp rendering without gaps between cells
    heatmap = go.Heatmap(
        z=grid,
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Activation",
            x=0.32,  # Position colorbar between subplot 1 and 2
            len=0.9
        ),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Activation: %{z:.4f}<extra></extra>',
        xgap=0,  # No gap between cells for crisp rendering
        ygap=0
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # ------------------------------------------------------------------------
    # Subplot 2: Matrix view with discrete color scaling and 4×4 block borders
    # ------------------------------------------------------------------------
    # Uses discrete colorscale from white (inactive) to darkred (highly active)
    # xgap=1, ygap=1 creates visible separation between cells
    matrix_heatmap = go.Heatmap(
        z=grid,
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
        colorbar=dict(
            title="Activation",
            x=0.65,  # Position colorbar between subplot 2 and 3
            len=0.9
        ),
        hovertemplate='Cell: (%{x}, %{y})<br>Activation: %{z:.4f}<extra></extra>',
        xgap=1,  # Add 1px gap between cells for grid effect
        ygap=1
    )
    fig.add_trace(matrix_heatmap, row=1, col=2)
    
    # Draw 4×4 block borders FIRST (if enabled)
    # This ensures the structural grid is always visible and respects max_shapes limit
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
                xref="x2", yref="y2"
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
                xref="x2", yref="y2"
            )
            shape_count += 1
        
        logger.debug(f"Drew {shape_count} block border lines")
    
    # ------------------------------------------------------------------------
    # Subplot 3: Histogram of activation values with statistics annotation
    # ------------------------------------------------------------------------
    if len(active_vals) > 0:
        # Create histogram of non-zero activation values
        hist = go.Histogram(
            x=active_vals,
            nbinsx=30,  # Use 30 bins for distribution
            marker=dict(color='teal', line=dict(color='black', width=1)),
            hovertemplate='Activation: %{x:.4f}<br>Count: %{y}<extra></extra>'
        )
        fig.add_trace(hist, row=1, col=3)
        
        # Add text annotation with comprehensive statistics
        # Positioned in top-right corner of histogram subplot
        stats_text = (
            f"<b>Statistics</b><br>"
            f"Total cells: {grid_stats['total_cells']}<br>"
            f"Active cells: {grid_stats['active_cells']}<br>"
            f"Activated (>{threshold}): {grid_stats['activated_cells_threshold']}<br>"
            f"Sparsity: {grid_stats['sparsity_percent']:.2f}%<br>"
            f"Max: {grid_stats['max_activation']:.4f}<br>"
            f"Mean (active): {grid_stats['mean_activation_active']:.4f}<br>"
            f"Std (active): {grid_stats['std_activation_active']:.4f}"
        )
        
        fig.add_annotation(
            text=stats_text,
            xref="x3", yref="y3",  # Reference subplot 3 axes
            x=0.95, y=0.95,  # Position in data coordinates (normalized)
            xanchor='right', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255, 255, 224, 0.8)",  # Light yellow background
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
            align='left'
        )
    
    # ========================================================================
    # Update layout and axis properties
    # ========================================================================
    
    # Subplot 1 axes: Standard heatmap with square aspect ratio
    fig.update_xaxes(title_text="X Coordinate", row=1, col=1, constrain="domain")
    fig.update_yaxes(title_text="Y Coordinate", row=1, col=1, scaleanchor="x", scaleratio=1)
    
    # Subplot 2 axes: Matrix view with square aspect ratio
    fig.update_xaxes(title_text="X Coordinate", row=1, col=2, constrain="domain")
    fig.update_yaxes(title_text="Y Coordinate", row=1, col=2, scaleanchor="x2", scaleratio=1)
    
    # Subplot 3 axes: Histogram
    fig.update_xaxes(title_text="Activation Strength", row=1, col=3)
    fig.update_yaxes(title_text="Cell Count", row=1, col=3)
    
    # Global layout settings
    fig.update_layout(
        title=dict(
            text=f'<b>Phrase Fingerprint Analysis: "{phrase}"</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        height=figure_height,
        width=figure_width,
        showlegend=False,
        template='plotly_white'
    )
    
    # ========================================================================
    # Save outputs to disk
    # ========================================================================
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize phrase for use in filename (replace spaces and slashes)
    safe_filename = phrase.replace(' ', '_').replace('/', '_')
    output_path = output_dir / f"single_{safe_filename}"
    
    # Save interactive HTML visualization
    if generate_html:
        fig.write_html(output_path.with_suffix('.html'))
        logger.info(f"Saved HTML: {output_path.with_suffix('.html')}")
    
    # Save static PNG image (requires kaleido package)
    if generate_png:
        try:
            fig.write_image(output_path.with_suffix('.png'), width=figure_width, height=figure_height, scale=2)
            logger.info(f"Saved PNG: {output_path.with_suffix('.png')}")
        except Exception as e:
            logger.warning(f"PNG export failed (install kaleido): {e}")
    
    # Save activated cells data as JSON
    # This provides a machine-readable list of all cells exceeding the threshold
    cells_data = {
        "phrase": phrase,
        "threshold": threshold,
        "total_activated": len(activated_coords),
        "activated_cells": [
            {"x": int(x), "y": int(y), "activation": float(val)}
            for (y, x), val in zip(activated_coords, activated_values)
        ]
    }
    
    cells_path = output_path.parent / f"activated_cells_{safe_filename}.json"
    with open(cells_path, 'w', encoding='utf-8') as f:
        json.dump(cells_data, f, indent=2)
    logger.info(f"Saved activated cells data: {cells_path}")
    
    # Save comprehensive metadata including statistics and configuration
    if save_metadata:
        save_visualization_metadata(
            output_path=output_path,
            mode="single",
            phrases={"phrase": phrase, "index": int(idx)},
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
                "colorscale": colorscale,
                "fingerprints_dir": str(fingerprints_dir)
            },
            top_cells={"phrase": top_cells}
        )


# ============================================================================
# Comparative Two-Phrase Visualization (Plotly 6-panel)
# ============================================================================

def visualize_phrase_pair(
    phrase1: str,
    phrase2: str,
    fingerprints_dir: Path,
    output_dir: Path,
    grid_size: int,
    use_morton: bool,
    threshold: float = 0.01,
    grid_borders: bool = True,
    border_color: str = "lightgray",
    border_width: float = 1.0,
    max_shapes: int = 5000,
    figure_width: int = 1800,
    figure_height: int = 1500,
    colorscale: str = "Viridis",
    generate_html: bool = True,
    generate_png: bool = True,
    save_metadata: bool = True,
) -> None:
    """
    Generate interactive 9-panel comparative dashboard with Plotly.
    
    Creates a comprehensive three-row visualization comparing two phrase fingerprints:
    
    Row 1 (Matrix Views):
    - Panel 1: Matrix view of phrase 1 with 4×4 block borders
    - Panel 2: Matrix view of phrase 2 with 4×4 block borders
    - Panel 3: Matrix view of semantic overlap with 4×4 block borders
    
    Row 2 (Spatial Heatmaps):
    - Panel 4: Continuous heatmap of phrase 1
    - Panel 5: Continuous heatmap of phrase 2
    - Panel 6: Semantic overlap heatmap
    
    Row 3 (Analysis):
    - Panel 7: Difference map (phrase1 - phrase2)
    - Panel 8: Similarity metrics (cosine similarity, distance, overlap)
    - Panel 9: Activation distribution histograms
    
    Args:
        phrase1: First phrase to compare (must exist in metadata)
        phrase2: Second phrase to compare (must exist in metadata)
        fingerprints_dir: Directory containing phrase_fingerprints.npz and 
                         phrase_fingerprints_meta.json files
        output_dir: Directory where visualization outputs will be saved
        grid_size: Size of the square grid (e.g., 128 for 128x128 grid)
        use_morton: If True, use Morton (Z-order) encoding for spatial reconstruction;
                   if False, use row-major ordering
        threshold: Activation threshold for determining "activated" cells in matrix views.
                  Cells with activation > threshold will have black borders drawn.
                  Default is 0.01.
        grid_borders: If True, draw 4×4 block borders on matrix views. Default is True.
        border_color: Color for 4×4 block borders. Default is "lightgray".
        border_width: Width of 4×4 block borders. Default is 1.0.
        max_shapes: Maximum number of shapes to draw (safety limit). Default is 5000.
        figure_width: Width of the output figure in pixels. Default is 1800.
        figure_height: Height of the output figure in pixels. Default is 1500.
        colorscale: Colorscale for the spatial heatmaps. Default is "Viridis".
        generate_html: If True, save HTML output. Default is True.
        generate_png: If True, save PNG output. Default is True.
        save_metadata: If True, save metadata JSON. Default is True.
    
    Returns:
        None. Outputs are saved to disk:
        - compare_{phrase1}_vs_{phrase2}.html: Interactive Plotly visualization
        - compare_{phrase1}_vs_{phrase2}.png: Static image (requires kaleido)
        - compare_{phrase1}_vs_{phrase2}_meta.json: Visualization metadata and statistics
    
    Raises:
        FileNotFoundError: If fingerprints or metadata files are missing
        ValueError: If either phrase is not found in the vocabulary
    
    Example:
        >>> visualize_phrase_pair(
        ...     phrase1="machine learning",
        ...     phrase2="deep learning",
        ...     fingerprints_dir=Path("data/fingerprints"),
        ...     output_dir=Path("outputs/viz"),
        ...     grid_size=128,
        ...     use_morton=True,
        ...     threshold=0.01
        ... )
    """
    logger.info(f"Loading fingerprints for comparison: '{phrase1}' vs '{phrase2}'")
    
    # Define paths to fingerprint data and metadata
    npz_path = fingerprints_dir / "phrase_fingerprints.npz"
    meta_path = fingerprints_dir / "phrase_fingerprints_meta.json"
    
    # Validate that required files exist
    if not npz_path.exists() or not meta_path.exists():
        logger.error(f"Missing Step 4 outputs in: {fingerprints_dir}")
        raise FileNotFoundError(f"Missing phrase_fingerprints files in {fingerprints_dir}")
    
    # Load phrase-to-row mapping from metadata
    logger.debug("Loading phrase metadata...")
    phrase_to_row = load_phrase_metadata(meta_path)
    logger.debug(f"Loaded {len(phrase_to_row)} phrases from metadata")
    
    # Verify that both phrases exist in the vocabulary
    missing = [p for p in [phrase1, phrase2] if p not in phrase_to_row]
    if missing:
        logger.error(f"Phrases not found: {missing}")
        available = list(phrase_to_row.keys())[:10]
        logger.info(f"Available phrases (first 10): {available}")
        raise ValueError(f"Phrases not found in vocabulary: {missing}")
    
    # Load fingerprint array and extract both phrases' fingerprints
    logger.debug(f"Loading fingerprints from {npz_path}...")
    fingerprints = np.load(npz_path)['fingerprints']
    logger.debug(f"Fingerprints shape: {fingerprints.shape}")
    
    idx1, idx2 = phrase_to_row[phrase1], phrase_to_row[phrase2]
    fp1, fp2 = fingerprints[idx1], fingerprints[idx2]
    
    # Log fingerprint characteristics for debugging
    logger.debug(f"Phrase 1 '{phrase1}': idx={idx1}, nnz={np.count_nonzero(fp1)}, "
                 f"max={fp1.max():.4f}")
    logger.debug(f"Phrase 2 '{phrase2}': idx={idx2}, nnz={np.count_nonzero(fp2)}, "
                 f"max={fp2.max():.4f}")
    
    # Reconstruct 2D grids from flattened fingerprints
    logger.info(f"Reconstructing 2D grids (size={grid_size}, morton={use_morton})")
    grid1 = inverse_flatten(fp1, grid_size, use_morton)
    grid2 = inverse_flatten(fp2, grid_size, use_morton)
    logger.debug(f"Grid 1 shape: {grid1.shape}, Grid 2 shape: {grid2.shape}")
    
    # Extract top active cells for each phrase and their overlap
    logger.debug("Extracting top active cells...")
    top_cells_1 = get_top_active_cells(grid1, top_n=20)
    top_cells_2 = get_top_active_cells(grid2, top_n=20)
    top_overlapped = get_top_overlapped_cells(grid1, grid2, top_n=20)
    
    logger.debug(f"Top 20 overlapped cells: {top_overlapped[:5]}...")
    
    # Identify activated cells for matrix views (cells exceeding threshold)
    logger.debug(f"Identifying activated cells with threshold={threshold}...")
    activated_coords_1 = np.argwhere(grid1 > threshold)
    activated_coords_2 = np.argwhere(grid2 > threshold)
    
    # Compute overlap and difference grids for comparison
    logger.debug("Computing overlap and difference grids...")
    overlap = np.minimum(grid1, grid2)
    activated_coords_overlap = np.argwhere(overlap > threshold)
    diff = grid1 - grid2
    
    logger.info(f"Phrase 1: {len(activated_coords_1)} activated cells (threshold={threshold})")
    logger.info(f"Phrase 2: {len(activated_coords_2)} activated cells (threshold={threshold})")
    logger.info(f"Overlap: {len(activated_coords_overlap)} activated cells (threshold={threshold})")
    
    # Compute comprehensive statistics for both phrases
    logger.debug("Computing statistics...")
    active1 = fp1[fp1 > 0]
    active2 = fp2[fp2 > 0]
    
    # Compute similarity metrics
    norm1, norm2 = np.linalg.norm(fp1), np.linalg.norm(fp2)
    cos_sim = np.dot(fp1, fp2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    grid_stats = {
        "phrase1": {
            "active_cells": int(len(active1)),
            "activated_cells_threshold": int(len(activated_coords_1)),
            "max_activation": float(fp1.max()),
            "mean_activation": float(active1.mean()) if len(active1) > 0 else 0.0,
            "grid_nonzero": int(np.count_nonzero(grid1))
        },
        "phrase2": {
            "active_cells": int(len(active2)),
            "activated_cells_threshold": int(len(activated_coords_2)),
            "max_activation": float(fp2.max()),
            "mean_activation": float(active2.mean()) if len(active2) > 0 else 0.0,
            "grid_nonzero": int(np.count_nonzero(grid2))
        },
        "comparison": {
            "cosine_similarity": float(cos_sim),
            "overlap_cells": int(np.count_nonzero(overlap)),
            "overlap_activated_threshold": int(len(activated_coords_overlap)),
            "overlap_max": float(overlap.max()),
            "difference_range": [float(diff.min()), float(diff.max())],
            "euclidean_distance": float(np.linalg.norm(fp1 - fp2)),
            "threshold": float(threshold)
        }
    }
    
    logger.debug(f"Comparison statistics: {grid_stats['comparison']}")
    logger.info(f"Cosine similarity: {cos_sim:.4f}")
    
    # ========================================================================
    # Create 9-panel interactive figure (3 rows × 3 columns)
    # ========================================================================
    logger.debug("Creating subplot structure...")
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            f'Matrix: "{phrase1}"',
            f'Matrix: "{phrase2}"',
            'Matrix: Overlap',
            f'Spatial: "{phrase1}"',
            f'Spatial: "{phrase2}"',
            'Spatial: Overlap',
            'Difference Map',
            'Similarity Metrics',
            'Activation Distribution'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.06,      # Reduced from 0.08
        horizontal_spacing=0.05,    # Reduced from 0.08
        row_heights=[0.33, 0.33, 0.34],
        column_widths=[0.33, 0.33, 0.34]  # Add explicit column widths
    )
    
    shape_count = 0
    
    # ========================================================================
    # ROW 1: Matrix Views with 4×4 Block Borders
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # Panel 1 (Row 1, Col 1): Matrix view of phrase 1
    # ------------------------------------------------------------------------
    logger.debug("Adding Panel 1: Matrix view of phrase 1...")
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
            colorbar=dict(
                title="Activation",
                x=0.29,
                len=0.28,
                y=0.83
            ),
            hovertemplate='Cell: (%{x}, %{y})<br>Activation: %{z:.4f}<extra></extra>',
            xgap=1, ygap=1
        ),
        row=1, col=1
    )
    
    # Draw 4×4 block borders for phrase 1
    if grid_borders:
        logger.debug("Drawing 4×4 block borders for phrase 1...")
        block_size = 4
        num_blocks = grid_size // block_size
        
        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                logger.warning(f"Reached max_shapes limit ({max_shapes}), skipping remaining borders")
                break
            
            x_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=x_pos, y0=-0.5,
                x1=x_pos, y1=grid_size-0.5,
                line=dict(color=border_color, width=border_width),
                layer="above",
                xref="x", yref="y"
            )
            shape_count += 1
        
        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            
            y_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=-0.5, y0=y_pos,
                x1=grid_size-0.5, y1=y_pos,
                line=dict(color=border_color, width=border_width),
                layer="above",
                xref="x", yref="y"
            )
            shape_count += 1
        
        logger.debug(f"Panel 1: Drew {(num_blocks + 1) * 2} border lines")
    
    # ------------------------------------------------------------------------
    # Panel 2 (Row 1, Col 2): Matrix view of phrase 2
    # ------------------------------------------------------------------------
    logger.debug("Adding Panel 2: Matrix view of phrase 2...")
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
            colorbar=dict(
                title="Activation",
                x=0.63,
                len=0.28,
                y=0.83
            ),
            hovertemplate='Cell: (%{x}, %{y})<br>Activation: %{z:.4f}<extra></extra>',
            xgap=1, ygap=1
        ),
        row=1, col=2
    )
    
    # Draw 4×4 block borders for phrase 2
    if grid_borders and shape_count < max_shapes:
        logger.debug("Drawing 4×4 block borders for phrase 2...")
        block_size = 4
        num_blocks = grid_size // block_size
        
        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                logger.warning(f"Reached max_shapes limit ({max_shapes}), skipping remaining borders")
                break
            
            x_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=x_pos, y0=-0.5,
                x1=x_pos, y1=grid_size-0.5,
                line=dict(color=border_color, width=border_width),
                layer="above",
                xref="x2", yref="y2"
            )
            shape_count += 1
        
        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            
            y_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=-0.5, y0=y_pos,
                x1=grid_size-0.5, y1=y_pos,
                line=dict(color=border_color, width=border_width),
                layer="above",
                xref="x2", yref="y2"
            )
            shape_count += 1
        
        logger.debug(f"Panel 2: Drew {(num_blocks + 1) * 2} border lines")
    
    # ------------------------------------------------------------------------
    # Panel 3 (Row 1, Col 3): Matrix view of overlap
    # ------------------------------------------------------------------------
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
            colorbar=dict(
                title="Overlap",
                x=0.97,
                len=0.28,
                y=0.83
            ),
            hovertemplate='Cell: (%{x}, %{y})<br>Overlap: %{z:.4f}<extra></extra>',
            xgap=1, ygap=1
        ),
        row=1, col=3
    )
    
    # Draw 4×4 block borders for overlap
    if grid_borders and shape_count < max_shapes:
        logger.debug("Drawing 4×4 block borders for overlap...")
        block_size = 4
        num_blocks = grid_size // block_size
        
        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                logger.warning(f"Reached max_shapes limit ({max_shapes}), skipping remaining borders")
                break
            
            x_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=x_pos, y0=-0.5,
                x1=x_pos, y1=grid_size-0.5,
                line=dict(color=border_color, width=border_width),
                layer="above",
                xref="x3", yref="y3"
            )
            shape_count += 1
        
        for i in range(num_blocks + 1):
            if shape_count >= max_shapes:
                break
            
            y_pos = i * block_size - 0.5
            fig.add_shape(
                type="line",
                x0=-0.5, y0=y_pos,
                x1=grid_size-0.5, y1=y_pos,
                line=dict(color=border_color, width=border_width),
                layer="above",
                xref="x3", yref="y3"
            )
            shape_count += 1
        
        logger.debug(f"Panel 3: Drew {(num_blocks + 1) * 2} border lines")
    
    logger.debug(f"Total shapes drawn: {shape_count}")
    
    # ========================================================================
    # ROW 2: Continuous Spatial Heatmaps
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # Panel 4 (Row 2, Col 1): Continuous heatmap of phrase 1
    # ------------------------------------------------------------------------
    logger.debug("Adding Panel 4: Continuous heatmap of phrase 1...")
    fig.add_trace(
        go.Heatmap(
            z=grid1,
            colorscale='Blues',
            zmin=0, zmax=1,
            colorbar=dict(
                title="Activation",
                x=0.29,
                len=0.28,
                y=0.5
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Activation: %{z:.4f}<extra></extra>',
            xgap=0, ygap=0
        ),
        row=2, col=1
    )
    
    # ------------------------------------------------------------------------
    # Panel 5 (Row 2, Col 2): Continuous heatmap of phrase 2
    # ------------------------------------------------------------------------
    logger.debug("Adding Panel 5: Continuous heatmap of phrase 2...")
    fig.add_trace(
        go.Heatmap(
            z=grid2,
            colorscale='Oranges',
            zmin=0, zmax=1,
            colorbar=dict(
                title="Activation",
                x=0.63,
                len=0.28,
                y=0.5
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Activation: %{z:.4f}<extra></extra>',
            xgap=0, ygap=0
        ),
        row=2, col=2
    )
    
    # ------------------------------------------------------------------------
    # Panel 6 (Row 2, Col 3): Continuous heatmap of overlap
    # ------------------------------------------------------------------------
    logger.debug("Adding Panel 6: Continuous heatmap of overlap...")
    fig.add_trace(
        go.Heatmap(
            z=overlap,
            colorscale='Purples',
            zmin=0, zmax=1,
            colorbar=dict(
                title="Overlap",
                x=0.97,
                len=0.28,
                y=0.5
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Overlap: %{z:.4f}<extra></extra>',
            xgap=0, ygap=0
        ),
        row=2, col=3
    )
    
    # ========================================================================
    # ROW 3: Analysis Panels
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # Panel 7 (Row 3, Col 1): Difference map (phrase1 - phrase2)
    # ------------------------------------------------------------------------
    logger.debug("Adding Panel 7: Difference map...")
    fig.add_trace(
        go.Heatmap(
            z=diff,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1, zmax=1,
            colorbar=dict(
                title="Difference",
                x=0.29,
                len=0.28,
                y=0.17
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Difference: %{z:.4f}<extra></extra>',
            xgap=0, ygap=0
        ),
        row=3, col=1
    )
    
    # ------------------------------------------------------------------------
    # Panel 8 (Row 3, Col 2): Similarity metrics text display
    # ------------------------------------------------------------------------
    logger.debug("Adding Panel 8: Similarity metrics...")
    metrics_text = (
        f"<b>Cosine Similarity</b><br><br>"
        f"<span style='font-size:32px'>{cos_sim:.4f}</span><br><br>"
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
    
    # ------------------------------------------------------------------------
    # Panel 9 (Row 3, Col 3): Histogram comparison of activation distributions
    # ------------------------------------------------------------------------
    logger.debug("Adding Panel 9: Activation histograms...")
    if len(active1) > 0:
        fig.add_trace(
            go.Histogram(
                x=active1,
                nbinsx=30,
                name=phrase1,
                marker=dict(color='blue', opacity=0.6, line=dict(color='black', width=1)),
                hovertemplate='Activation: %{x:.4f}<br>Count: %{y}<extra></extra>'
            ),
            row=3, col=3
        )
    
    if len(active2) > 0:
        fig.add_trace(
            go.Histogram(
                x=active2,
                nbinsx=30,
                name=phrase2,
                marker=dict(color='darkorange', opacity=0.6, line=dict(color='black', width=1)),
                hovertemplate='Activation: %{x:.4f}<br>Count: %{y}<extra></extra>'
            ),
            row=3, col=3
        )
    
    # ========================================================================
    # Update axes for all panels with proper constraints
    # ========================================================================
    logger.debug("Updating axes with scaling constraints...")
    
    # Panels 1-6: Heatmaps (rows 1-2, all columns) - enforce square aspect ratio
    for row in [1, 2]:
        for col in [1, 2, 3]:
            subplot_num = (row - 1) * 3 + col
            xaxis_ref = f"x{subplot_num}" if subplot_num > 1 else "x"
            yaxis_ref = f"y{subplot_num}" if subplot_num > 1 else "y"
            
            fig.update_xaxes(
                title_text="X",
                row=row, col=col,
                constrain="domain",
                showgrid=False
            )
            fig.update_yaxes(
                title_text="Y",
                row=row, col=col,
                scaleanchor=xaxis_ref,
                scaleratio=1,
                constrain="domain",
                showgrid=False
            )
    
    # Panel 7 (Row 3, Col 1): Difference heatmap - also square
    fig.update_xaxes(
        title_text="X",
        row=3, col=1,
        constrain="domain",
        showgrid=False
    )
    fig.update_yaxes(
        title_text="Y",
        row=3, col=1,
        scaleanchor="x7",
        scaleratio=1,
        constrain="domain",
        showgrid=False
    )
    
    # Panel 8 (Row 3, Col 2): Metrics bar chart - no aspect ratio constraint
    fig.update_xaxes(visible=False, row=3, col=2)
    fig.update_yaxes(visible=False, row=3, col=2)
    
    # Panel 9 (Row 3, Col 3): Histogram - standard axes
    fig.update_xaxes(
        title_text="Activation",
        row=3, col=3,
        showgrid=True,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        title_text="Count",
        row=3, col=3,
        showgrid=True,
        gridcolor='lightgray'
    )
    
    # ========================================================================
    # Update global layout with fixed dimensions
    # ========================================================================
    logger.debug("Updating layout...")
    logger.debug(f"Setting figure dimensions: {figure_width}x{figure_height}")
    fig.update_layout(
        title=dict(
            text=f'<b>Comparative Analysis: "{phrase1}" vs "{phrase2}"</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        height=figure_height,
        width=figure_width,
        showlegend=True,
        legend=dict(
            x=0.85,
            y=0.17,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        template='plotly_white',
        autosize=False,
        margin=dict(l=60, r=60, t=100, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # ========================================================================
    # Save outputs to disk
    # ========================================================================
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name1 = phrase1.replace(' ', '_').replace('/', '_')
    safe_name2 = phrase2.replace(' ', '_').replace('/', '_')
    output_path = output_dir / f"compare_{safe_name1}_vs_{safe_name2}"
    
    if generate_html:
        html_path = output_path.with_suffix('.html')
        logger.debug(f"Saving HTML to {html_path}...")
        fig.write_html(
            str(html_path),
            config={
                'displayModeBar': True,
                'responsive': False,
                'displaylogo': False
            }
        )
        logger.info(f"Saved interactive HTML: {html_path}")
    
    if generate_png:
        png_path = output_path.with_suffix('.png')
        logger.debug(f"Saving PNG to {png_path} with dimensions {figure_width}x{figure_height} ...")
        try:
            fig.write_image(
                str(png_path), 
                width=figure_width, 
                height=figure_height,
                scale=1
            )
            logger.info(f"Saved static PNG: {png_path}")
        except Exception as e:
            logger.warning(f"Could not save PNG (kaleido required): {e}")
    
    if save_metadata:
        meta_path = output_dir / f"{output_path.stem}_meta.json"
        
        logger.debug(f"Saving metadata to {meta_path}...")
        save_visualization_metadata(
            output_path=meta_path,
            mode="comparative",
            phrases={
                "phrase1": phrase1,
                "phrase2": phrase2
            },
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
            top_cells={
                "phrase1": top_cells_1,
                "phrase2": top_cells_2
            }
        )
        logger.info(f"Saved metadata: {meta_path}")
    
    logger.info(f"Visualization complete for '{phrase1}' vs '{phrase2}'")

# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Visualize phrase fingerprints with interactive Plotly graphics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single phrase visualization
  python phrase_visualizer.py \\
      --fingerprints outputs/run/phrase_fingerprints \\
      --phrase "machine learning" \\
      --output ./visualizations

  # Compare two phrases
  python phrase_visualizer.py \\
      --fingerprints outputs/run/phrase_fingerprints \\
      --phrase1 "linguistic" \\
      --phrase2 "language" \\
      --output ./visualizations

  # With custom styling
  python phrase_visualizer.py \\
      --fingerprints outputs/run/phrase_fingerprints \\
      --phrase1 "AI" \\
      --phrase2 "ML" \\
      --output ./viz \\
      --border-color darkgray \\
      --border-width 1.5 \\
      --colorscale Viridis

  # With debug logging
  LOG_LEVEL=DEBUG python phrase_visualizer.py \\
      --fingerprints outputs/run/phrase_fingerprints \\
      --phrase "AI" \\
      --output ./viz
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--fingerprints', '-f', type=Path, required=True,
        help='Directory containing Step 4 outputs'
    )
    parser.add_argument(
        '--output', '-o', type=Path, required=True,
        help='Output directory for visualizations'
    )
    
    # Phrase selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--phrase', type=str, help='Single phrase to visualize')
    group.add_argument('--phrase1', type=str, help='First phrase for comparison')
    
    parser.add_argument('--phrase2', type=str, help='Second phrase for comparison')
    
    # Grid configuration
    parser.add_argument('--grid-size', type=int, default=128, help='Grid dimension (default: 128)')
    
    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument('--morton', action='store_true', default=True, dest='morton')
    exclusive_group.add_argument('--no-morton', action='store_false', dest='morton')
    
    # Activation threshold
    parser.add_argument('--threshold', type=float, default=0.0, 
                        help='Activation threshold (default: 0.0)')
    
    # Border styling
    parser.add_argument('--no-grid-borders', action='store_true', 
                        help='Disable 4×4 block borders')
    parser.add_argument('--border-color', type=str, default='lightgray',
                        help='Color of 4×4 block borders (default: lightgray)')
    parser.add_argument('--border-width', type=float, default=1.0,
                        help='Width of block border lines (default: 1.0)')
    
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
    
    args = parser.parse_args()
    
    if args.phrase1 and not args.phrase2:
        parser.error("--phrase2 is required when using --phrase1")
    
    logger.info("=" * 60)
    logger.info("Phrase Fingerprint Visualization Tool (Plotly)")
    logger.info("=" * 60)
    logger.info(f"Fingerprints dir: {args.fingerprints}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Grid size: {args.grid_size}")
    logger.info(f"Encoding: {'Row-major' if not args.morton else 'Morton (Z-order)'}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Grid borders: {'Disabled' if args.no_grid_borders else f'Enabled ({args.border_color}, width={args.border_width})'}")
    logger.info(f"Figure size: {args.width}×{args.height}")
    logger.info(f"Colorscale: {args.colorscale}")
    logger.info(f"Max shapes: {args.max_shapes}")
    logger.info("=" * 60)
    
    try:
        if args.phrase:
            logger.info("Mode: Single-phrase visualization")
            visualize_single_phrase(
                phrase=args.phrase.lower(),
                fingerprints_dir=args.fingerprints,
                output_dir=args.output,
                grid_size=args.grid_size,
                use_morton=args.morton,
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
        else:
            logger.info("Mode: Comparative two-phrase visualization")
            visualize_phrase_pair(
                phrase1=args.phrase1.lower(),
                phrase2=args.phrase2.lower(),
                fingerprints_dir=args.fingerprints,
                output_dir=args.output,
                grid_size=args.grid_size,
                use_morton=args.morton,
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
        
        logger.info("=" * 60)
        logger.info("Visualization completed successfully.")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.exception(f"Visualization failed: {e}")
        sys.exit(1)



if __name__ == '__main__':
    main()
