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
            # visualize_customtext_pair(
            #     phrase1=args.phrase1.lower(),
            #     phrase2=args.phrase2.lower(),
            #     fingerprints_dir=args.fingerprints,
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
            print("!!! visualize customtext pair not ready, now")
    
    except Exception as e:
        logger.exception(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
