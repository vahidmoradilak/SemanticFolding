"""
Document Fingerprint Visualizer
PhD Thesis: Semantic Folding for Closed-Domain QA
Step 5: Document Fingerprint Analysis Dashboard

Visualizes document-level semantic fingerprints with spatial analysis.
Handles sparse TF-IDF weighted fingerprints with proper normalization.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, Tuple, List
import argparse
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from doc_fingerprints import morton_to_xy
# Import from your lib module
from lib import load_document_fingerprints, load_phrase_fingerprints_sparse, get_logger

# Initialize logger
logger = get_logger("doc_visualizer")


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
            'row': int(r),
            'col': int(c),
            'value': float(fingerprint_2d[r, c])
        })
    
    logger.debug(f"Extracted top {len(cells)} active cells, "
                f"highest value: {cells[0]['value']:.6f}" if cells else "No active cells")
    
    return cells


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
    Create comprehensive document fingerprint visualization.
    
    Generates a 2×3 dashboard with:
    - Row 1: Document matrix (with 4×4 borders) | Spatial heatmap | Metrics
    - Row 2: Activation histogram | Spatial density | Top active cells
    
    Args:
        fingerprint: 1D flattened fingerprint vector (TF-IDF weighted)
        doc_id: Document identifier
        metadata: Dict with grid_size, num_docs, etc.
        grid_size: Dimension of the square grid
        grid_borders: If True, draw 4×4 block borders on matrix view
        border_color: Color for 4×4 block borders
        border_width: Width of 4×4 block borders
        max_shapes: Maximum number of shapes to draw (safety limit)
        
    Returns:
        Plotly Figure object
    """
    logger.info(f"Creating visualization for document: {doc_id}")
    
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
            'Document Fingerprint Matrix (4×4 Grid)',
            'Spatial Activation Heatmap',
            'Document Metrics',
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
    # Row 1, Col 1: Document Matrix View with 4×4 block borders
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
        ['Document ID', doc_id],
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
            [f"({cell['row']}, {cell['col']})" for cell in top_cells],
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
            text=f'Document Fingerprint Analysis: {doc_id}: {doc_text[:128]}',
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

def main():
    """
    Main entry point for document fingerprint visualization.
    
    Usage:
        python doc_visualizer.py --run-dir outputs/20260423_023143/ \
                                 --doc-id doc_001 \
                                 --output visualizations/
    """
    parser = argparse.ArgumentParser(
        description='Visualize document semantic fingerprints'
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        required=True,
        help='Path to run output directory (e.g., outputs/20260423_023143/)'
    )
    parser.add_argument(
        '--doc-id',
        type=str,
        required=True,
        help='Document ID to visualize'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output folder path for visualizations'
    )
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
    
    args = parser.parse_args()
    
    logger.info(f"Starting document visualization for {args.doc_id}")
    logger.debug(f"Run directory: {args.run_dir}")
    logger.debug(f"Output directory: {args.output}")
    
    # Construct paths from run directory
    doc_fp_dir = args.run_dir / 'doc_fingerprints'
    
    # Validate directories exist
    if not doc_fp_dir.exists():
        logger.error(f"Document fingerprints directory not found: {doc_fp_dir}")
        print(f"Error: Document fingerprints directory not found: {doc_fp_dir}")
        return
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {args.output}")
    
    # Load document fingerprints using lib function
    logger.info(f"Loading document fingerprints from {doc_fp_dir}...")
    print(f"Loading document fingerprints from {doc_fp_dir}...")
    
    try:
        doc_fingerprints, metadata = load_document_fingerprints(doc_fp_dir)
        use_morton = metadata.get('use_morton', False)   # fallback False (row-major) for old runs
        grid_size = metadata['grid_size']
        logger.info(f"Loaded {len(doc_fingerprints)} document fingerprints")
        logger.debug(f"Metadata: {metadata}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load document fingerprints: {e}")
        print(f"Error loading document fingerprints: {e}")
        return
    
    # Check if document exists
    if args.doc_id not in doc_fingerprints:
        available_docs = list(doc_fingerprints.keys())[:10]
        logger.error(f"Document ID '{args.doc_id}' not found in fingerprints")
        logger.debug(f"Available documents (first 10): {available_docs}")
        print(f"Error: Document ID '{args.doc_id}' not found.")
        print(f"Available documents: {available_docs}...")
        return
    
    # Get fingerprint (convert from sparse to dense)
    doc_fp_sparse = doc_fingerprints[args.doc_id]
    fingerprint = doc_fp_sparse.toarray().flatten()
    
    logger.debug(f"Converted sparse fingerprint to dense array: shape {fingerprint.shape}")
    
    grid_size = metadata['grid_size']
    
    text_dir = "data\\quran\\quran_ayahs_clean.txt" 
    # text_dir = "data\\quran\\quran_ayahs_NE.txt" 
    # text_dir = "data\\quran\\quran_ayahs_tail764.txt" 
    # text_dir = "data\\corpus.txt"
    doc_text = get_document_by_id(text_dir, args.doc_id)
            
    print(f"Visualizing customtext {args.doc_id}: {doc_text}")
    print(f"Grid size: {grid_size}×{grid_size}")
    print(f"Total documents: {metadata['num_docs']}")
    
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
    logger.info("Document visualization completed successfully")


if __name__ == '__main__':
    main()
