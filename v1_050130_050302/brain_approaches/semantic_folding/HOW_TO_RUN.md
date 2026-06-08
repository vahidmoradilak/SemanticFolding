# Semantic Folding Pipeline: Complete Guide

## Introduction

Semantic Folding is a brain-inspired approach to semantic representation and retrieval that maps linguistic concepts into a discrete 2D semantic space. Unlike traditional vector embeddings, this method creates sparse, interpretable "fingerprints" that preserve semantic relationships while enabling efficient similarity search.

The pipeline transforms raw text into semantic fingerprints through dimensionality reduction and spatial encoding, mimicking how the cortex organizes conceptual knowledge. Documents and queries are represented as activation patterns over a shared semantic grid, enabling context-aware retrieval without dense vector operations.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Pipeline Overview](#pipeline-overview)
5. [Pipeline Steps](#pipeline-steps)
   - [Phase 1: Phrase Extraction](#phase-1-phrase-extraction)
   - [Phase 2: Term-Context Matrix](#phase-2-term-context-matrix)
   - [Phase 3: Semantic Space](#phase-3-semantic-space)
   - [Phase 4: Phrase Fingerprints](#phase-4-phrase-fingerprints)
   - [Phase 5: Document Fingerprints](#phase-5-document-fingerprints)
   - [Phase 6: Query Processing](#phase-6-query-processing)
6. [Utility Scripts](#utility-scripts)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

**System Requirements:**
- Python 3.11 or higher
- 8GB+ RAM (16GB recommended for large corpora)
- macOS, Linux, or Windows with WSL

**Core Dependencies:**
- `loguru` - Structured logging
- `scipy` - Sparse matrix operations
- `spacy` - NLP and phrase extraction
- `scikit-learn` - Dimensionality reduction (t-SNE, PCA)
- `umap-learn` - UMAP dimensionality reduction
- `tqdm` - Progress bars
- `pyyaml` - Configuration file parsing

## Installation

Install `uv` (fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the project:

```bash
git clone https://github.com/yourusername/knowledge-graph-builder.git
cd knowledge-graph-builder
```

Install dependencies:

```bash
uv pip install loguru scipy spacy scikit-learn umap-learn tqdm pyyaml
python -m spacy download en_core_web_sm
```

Optional dependencies:

```bash
uv pip install matplotlib seaborn  # For visualizations
uv pip install networkx            # For semantic graphs
```

## Configuration

The pipeline uses a centralized YAML configuration file located at `config/semantic_folding.yml`. This file controls all pipeline parameters, from phrase extraction to query processing.

**Configuration File Structure:**

Create `config/semantic_folding.yml` with the following structure:

```yaml
# ============================================================
# Semantic Folding Pipeline Configuration
# ============================================================
# Production-ready settings for brain-inspired semantic retrieval

# ============================================================
# INPUT/OUTPUT PATHS
# ============================================================
paths:
  # Input data files
  corpus_path: "data/corpus.txt"
  queries_path: null  # Optional: path to queries file
  
  # Output directory structure
  output_base: "outputs"
  output_timestamp: true  # Append timestamp to output folder names
  
  # Optional: Pre-computed artifacts (skip earlier stages)
  phrases_path: null
  matrix_path: null
  coordinates_path: null
  phrase_fingerprints_path: null
  doc_fingerprints_path: null

# ============================================================
# PHASE 1: PHRASE EXTRACTION
# ============================================================
phrase_extraction:
  # Frequency filtering
  min_freq: 2                    # Minimum phrase frequency to keep
  min_word_length: 3             # Minimum character length for single words
  
  # Extraction method
  use_spacy: true                # Use spaCy (fallback to NLTK if unavailable)
  filter_generic: true           # Remove generic words (get, do, make, etc.)
  
  # N-gram settings (fallback mode)
  max_ngram: 3                   # Maximum n-gram size for extraction
  
  # Output options
  show_stats: true               # Print extraction statistics

# ============================================================
# PHASE 2: TERM-CONTEXT MATRIX
# ============================================================
term_context_matrix:
  # Matrix construction
  min_phrase_freq: 0             # Additional frequency filter for matrix
  use_tfidf: true                # Apply TF-IDF normalization
  word_boundaries: true          # Use word boundary matching (not substring)
  keep_verbs: false              # Include verbal elements in phrases
  
  # Processing
  batch_size: 1000               # Chunk size for large datasets
  sparse_format: "csr"           # Sparse matrix format (csr, csc, lil)

# ============================================================
# PHASE 3: SEMANTIC SPACE
# ============================================================
semantic_space:
  # Dimensionality reduction method
  method: "tsne"                 # Options: tsne, umap, pca
  
  # Grid configuration
  grid_size: 32                  # Semantic space resolution (8, 16, 32, 64)
  grid_padding: 0                # Padding around grid edges
  enable_grid: true              # Generate discrete grid coordinates
  
  # t-SNE specific parameters
  tsne:
    perplexity: 30               # Balance local vs global structure (5-50)
    max_iter: 1000               # Maximum iterations
    learning_rate: 200           # Step size (auto, or 10-1000)
    early_exaggeration: 12       # Initial separation strength
    metric: "cosine"             # Distance metric
    
  # UMAP specific parameters
  umap:
    n_neighbors: 15              # Local neighborhood size (2-100)
    min_dist: 0.1                # Minimum separation (0.0-0.99)
    metric: "cosine"             # Distance metric
    
  # PCA specific parameters
  pca:
    n_components: 2              # Number of dimensions (always 2 for 2D space)
    
  # Collision resolution
  collision_resolution: true     # Use Morton Z-order curve for collisions
  
  # Performance
  n_jobs: 1                      # Parallel jobs (-1 = all cores)
  use_sparse: false              # Use sparse matrices (UMAP/PCA only)
  
  # Visualization
  visualize: true                # Generate PNG plots
  show_density: false            # Add density heatmap overlay

# ============================================================
# PHASE 4: PHRASE FINGERPRINTS
# ============================================================
phrase_fingerprints:
  # Fingerprint representation
  binary: false                  # Use presence/absence (not weights)
  normalization: "none"          # Options: none, l1, l2, binary
  
  # Sparsification
  sparsify_threshold: 0.0        # Zero out values below threshold (0.0 = disabled)
  
  # Grid indexing
  use_morton: true               # Enable Morton Z-order indexing
  
  # Testing/debugging
  max_phrases: null              # Limit phrases for testing (null = all)

# ============================================================
# PHASE 5: DOCUMENT FINGERPRINTS
# ============================================================
document_fingerprints:
  # Thresholding strategy
  enable_threshold: true         # Apply thresholding to reduce fingerprint size
  threshold_method: "z_order"    # Options: z_order (locality-preserving), value (magnitude)
  top_percent: 0.05              # Keep top N% of cells (0.01-0.20 recommended)
  
  # Weighting
  use_idf: false                 # Apply IDF weighting to phrases
  normalization: "l2"            # Options: none, l1, l2, binary
  
  # Phrase extraction from documents
  max_ngram: 2                   # Maximum n-gram size (1-3)
  min_word_length: 2             # Minimum token length
  
  # Export options
  export_numpy: false            # Export as NumPy array (.npy)
  export_statistics: true        # Generate corpus statistics JSON
  
  # Testing/debugging
  max_docs: null                 # Limit documents for testing (null = all)

# ============================================================
# PHASE 6: QUERY PROCESSING
# ============================================================
query_processing:
  # Weighting scheme
  weighting: "uniform"           # Options: uniform, frequency, idf
  normalization: "l2"            # Options: none, l1, l2, binary
  
  # Query phrase extraction
  max_ngram: 3                   # Maximum n-gram size for query
  min_word_length: 2             # Minimum token length
  
  # Spatial spreading (context expansion)
  spreading:
    enabled: true                # Enable spreading to neighboring cells
    radius: 1                    # Spreading radius (0 = exact match, 1-3 = recall boost)
    decay: 0.5                   # Decay factor per distance (0.0-1.0)
    normalize_after: false       # Renormalize fingerprint after spreading
  
  # Ranking
  top_k: 10                      # Number of results to return
  min_similarity: 0.0            # Minimum similarity threshold (0.0-1.0)
  use_batch: true                # Use batch cosine similarity (faster)
  
  # Output
  verbose: false                 # Show detailed query construction info
  output_json: null              # Optional: save results to JSON file

# ============================================================
# SEMANTIC GRAPH (VISUALIZATION)
# ============================================================
semantic_graph:
  enabled: false                 # Generate semantic graph visualization
  max_edges: 200                 # Maximum edges in graph
  edge_threshold: 0.05           # Minimum similarity for edge creation
  layout: "force"                # Layout algorithm (force, circular, spring)

# ============================================================
# LOGGING & DEBUGGING
# ============================================================
logging:
  level: "INFO"                  # Console log level (DEBUG, INFO, WARNING, ERROR)
  debug: false                   # Enable debug mode with stack traces
  log_file: null                 # Optional: write logs to file
  progress_bars: true            # Show tqdm progress bars

# ============================================================
# PERFORMANCE & OPTIMIZATION
# ============================================================
performance:
  # Memory management
  max_memory_gb: null            # Maximum memory usage (null = unlimited)
  clear_intermediate: false      # Delete intermediate files after each phase
  
  # Parallel processing
  n_jobs: 1                      # Number of parallel workers (-1 = all cores)
  
  # Batch processing
  batch_size: 1000               # Default batch size for chunked operations

# ============================================================
# EXPERIMENTAL FEATURES
# ============================================================
experimental:
  # Advanced phrase extraction
  use_dependency_parsing: false  # Use dependency trees for phrase extraction
  
  # Alternative similarity metrics
  similarity_metric: "cosine"    # Options: cosine, euclidean, manhattan
  
  # Fingerprint compression
  quantize_fingerprints: false   # Quantize values to reduce storage
  quantization_bits: 8           # Bits per value (4, 8, 16)

# ============================================================
# PRESETS (QUICK CONFIGURATIONS)
# ============================================================
# Uncomment one preset to override individual settings

# presets:
#   # Fast prototyping (small grid, minimal processing)
#   fast:
#     semantic_space:
#       grid_size: 16
#       tsne:
#         max_iter: 500
#     document_fingerprints:
#       top_percent: 0.10
#     query_processing:
#       spreading:
#         radius: 0
  
#   # Balanced quality (recommended for most use cases)
#   balanced:
#     semantic_space:
#       grid_size: 32
#       tsne:
#         perplexity: 30
#         max_iter: 1000
#     document_fingerprints:
#       top_percent: 0.05
#     query_processing:
#       spreading:
#         radius: 1
#         decay: 0.5
  
#   # High precision (large grid, aggressive filtering)
#   precision:
#     semantic_space:
#       grid_size: 64
#       tsne:
#         perplexity: 50
#         max_iter: 2000
#     document_fingerprints:
#       top_percent: 0.03
#       use_idf: true
#     query_processing:
#       weighting: "idf"
#       spreading:
#         radius: 1
#         decay: 0.3
  
#   # High recall (spreading enabled, relaxed thresholds)
#   recall:
#     semantic_space:
#       grid_size: 32
#     document_fingerprints:
#       top_percent: 0.10
#     query_processing:
#       spreading:
#         radius: 2
#         decay: 0.6
#       top_k: 20
```

**Loading Configuration in Scripts:**

All pipeline scripts support loading configuration via the `--config` flag:

```bash
uv run python scripts/semantic_folding/phrase_extraction.py --config config/semantic_folding.yml
```

Configuration values can be overridden via command-line arguments:

```bash
# Override grid_size from config
uv run python scripts/semantic_folding/semantic_space.py \
  --config config/semantic_folding.yml \
  --grid-size 64
```

**Configuration Guidelines:**

Grid Size Recommendations:
- 16: Fast prototyping, small corpora (<1K docs)
- 32: Balanced quality, medium corpora (1K-10K docs)
- 64: High precision, large corpora (>10K docs)

Spreading Radius Guidelines:
- 0: Exact cell matching (high precision, low recall)
- 1: Moderate context expansion (balanced)
- 2-3: Aggressive expansion (high recall, lower precision)

Top Percent Guidelines:
- 0.03-0.05: Sparse, distinctive fingerprints
- 0.05-0.10: Balanced representation
- 0.10-0.20: Dense, comprehensive coverage

Memory Considerations:
- Grid 16: ~256 cells, minimal memory
- Grid 32: ~1,024 cells, moderate memory
- Grid 64: ~4,096 cells, high memory usage

## Pipeline Overview

The Semantic Folding pipeline consists of 6 sequential phases:

Raw Corpus (corpus.txt)
    ↓
[Phase 1: Phrase Extraction]
    → phrases.json (25,572 phrases)
    ↓
[Phase 2: Term-Context Matrix]
    → term_context_matrix.npz (11656 × 25572, sparsity: 0.0028)
    ↓
[Phase 3: Semantic Space]
    → phrase_coordinates.json (2D embeddings)
    → semantic_space.png (visualization)
    ↓
[Phase 4: Phrase Fingerprints]
    → phrase_fingerprints.npz (sparse grid activations)
    ↓
[Phase 5: Document Fingerprints]
    → doc_fingerprints.npz (aggregated phrase fingerprints)
    ↓
[Phase 6: Query Processing]
    → query_results.json (ranked document matches)


Each phase produces artifacts that serve as inputs to subsequent stages, enabling incremental processing and debugging.

## Pipeline Steps

### Phase 1: Phrase Extraction

Extracts meaningful phrases from raw text using spaCy's linguistic analysis or n-gram fallback.

**Script:** `scripts/semantic_folding/phrase_extraction.py`

**Command-Line Arguments:**

```bash
--corpus PATH              # Input corpus file (required)
--output PATH              # Output JSON file for phrases
--config PATH              # Configuration file
--min-freq INT             # Minimum phrase frequency (default: 2)
--min-word-length INT      # Minimum word length (default: 3)
--use-spacy                # Use spaCy for extraction (default: True)
--max-ngram INT            # Maximum n-gram size for fallback (default: 3)
```

**Key Features:**

- Noun phrase extraction via spaCy dependency parsing
- Frequency-based filtering to remove rare phrases
- Generic word filtering (e.g., "get", "do", "make")
- Fallback to n-gram extraction if spaCy unavailable

**Usage Example:**

```bash
uv run python scripts/semantic_folding/phrase_extraction.py \
  --corpus data/corpus.txt \
  --output outputs/phrases.json \
  --min-freq 2 \
  --min-word-length 3
```

**Expected Output:**

Phrases extracted: 25,572
Output saved to: outputs/phrases.json


**Output Format (phrases.json):**

```json
{
  "neural network": 1247,
  "machine learning": 892,
  "deep learning": 634,
  "artificial intelligence": 521
}
```

### Phase 2: Term-Context Matrix

Constructs a sparse term-context co-occurrence matrix with TF-IDF weighting.

**Script:** `scripts/semantic_folding/term_context_matrix.py`

**Command-Line Arguments:**

```bash
--corpus PATH              # Input corpus file (required)
--phrases PATH             # Phrases JSON from Phase 1 (required)
--output PATH              # Output .npz file for matrix
--config PATH              # Configuration file
--use-tfidf                # Apply TF-IDF normalization (default: True)
--min-phrase-freq INT      # Additional frequency filter (default: 0)
--batch-size INT           # Processing batch size (default: 1000)
```

**Key Features:**

- Sparse matrix representation (CSR format)
- TF-IDF normalization for semantic weighting
- Word boundary matching (not substring matching)
- Memory-efficient batch processing

**Usage Example:**

```bash
uv run python scripts/semantic_folding/term_context_matrix.py \
  --corpus data/corpus.txt \
  --phrases outputs/phrases.json \
  --output outputs/term_context_matrix.npz \
  --use-tfidf
```

**Expected Output:**

Matrix created: 11656 × 25572
Sparsity: 0.0028
Output saved to: outputs/term_context_matrix.npz


**Output Format:**

- Sparse matrix in `.npz` format (SciPy `save_npz`)
- Rows: Context windows (documents/sentences)
- Columns: Phrases
- Values: TF-IDF weighted co-occurrence counts

### Phase 3: Semantic Space

Reduces the term-context matrix to 2D coordinates using t-SNE, UMAP, or PCA, then maps to a discrete grid.

**Script:** `scripts/semantic_folding/semantic_space.py`

**Command-Line Arguments:**

```bash
--matrix PATH              # Term-context matrix from Phase 2 (required)
--phrases PATH             # Phrases JSON from Phase 1 (required)
--output PATH              # Output JSON file for coordinates
--config PATH              # Configuration file
--method STR               # Reduction method: tsne, umap, pca (default: tsne)
--grid-size INT            # Grid resolution (default: 32)
--perplexity FLOAT         # t-SNE perplexity (default: 30)
--n-neighbors INT          # UMAP neighbors (default: 15)
--visualize                # Generate PNG visualization (default: True)
```

**Key Features:**

- Multiple dimensionality reduction algorithms
- Discrete grid quantization with collision resolution
- Morton Z-order curve for spatial locality preservation
- Automatic visualization with density heatmaps

**Usage Example:**

```bash
uv run python scripts/semantic_folding/semantic_space.py \
  --matrix outputs/term_context_matrix.npz \
  --phrases outputs/phrases.json \
  --output outputs/phrase_coordinates.json \
  --method tsne \
  --grid-size 32 \
  --perplexity 30 \
  --visualize
```

**Expected Output:**

Dimensionality reduction: t-SNE (perplexity=30)
Grid size: 32x32 (1024 cells)
Collisions resolved: 142 using Morton Z-order
Visualization saved to: outputs/semantic_space.png
Output saved to: outputs/phrase_coordinates.json


**Output Format (phrase_coordinates.json):**

```json
{
  "neural network": {"x": 15, "y": 22, "morton": 734},
  "machine learning": {"x": 14, "y": 23, "morton": 742},
  "deep learning": {"x": 16, "y": 21, "morton": 726}
}
```

**Visualization:**

- PNG scatter plot with phrase labels
- Optional density heatmap overlay
- Color-coded by semantic clusters

### Phase 4: Phrase Fingerprints

Converts phrase coordinates into sparse grid-based fingerprints.

**Script:** `scripts/semantic_folding/phrase_fingerprints.py`

**Command-Line Arguments:**

```bash
--coordinates PATH         # Phrase coordinates from Phase 3 (required)
--output PATH              # Output .npz file for fingerprints
--config PATH              # Configuration file
--grid-size INT            # Grid resolution (must match Phase 3)
--binary                   # Use binary fingerprints (default: False)
--normalization STR        # Normalization: none, l1, l2, binary (default: none)
```

**Key Features:**

- Sparse fingerprint representation (one-hot or weighted)
- Optional L1/L2 normalization
- Morton Z-order indexing for spatial queries
- Memory-efficient storage

**Usage Example:**

```bash
uv run python scripts/semantic_folding/phrase_fingerprints.py \
  --coordinates outputs/phrase_coordinates.json \
  --output outputs/phrase_fingerprints.npz \
  --grid-size 32
```

**Expected Output:**

Fingerprints created: 25,572 phrases
Grid size: 32x32 (1024 cells)
Average sparsity: 0.0009 (1 active cell per phrase)
Output saved to: outputs/phrase_fingerprints.npz


**Output Format:**

- Sparse matrix (CSR format): `(num_phrases, grid_size^2)`
- Each row is a phrase fingerprint
- Values represent activation strength at grid cells

### Phase 5: Document Fingerprints

Aggregates phrase fingerprints into document-level representations with thresholding.

**Script:** `scripts/semantic_folding/doc_fingerprints.py`

**Command-Line Arguments:**

```bash
--corpus PATH              # Input corpus file (required)
--phrases PATH             # Phrases JSON from Phase 1 (required)
--phrase-fps PATH          # Phrase fingerprints from Phase 4 (required)
--output PATH              # Output .npz file for document fingerprints
--config PATH              # Configuration file
--threshold-method STR     # Thresholding: z_order, value (default: z_order)
--top-percent FLOAT        # Keep top N% of cells (default: 0.05)
--use-idf                  # Apply IDF weighting (default: False)
--normalization STR        # Normalization: none, l1, l2 (default: l2)
```

**Key Features:**

- Phrase-to-document aggregation
- Locality-preserving thresholding (z_order method)
- Optional IDF weighting for rare phrases
- Corpus statistics export

**Usage Example:**

```bash
uv run python scripts/semantic_folding/doc_fingerprints.py \
  --corpus data/corpus.txt \
  --phrases outputs/phrases.json \
  --phrase-fps outputs/phrase_fingerprints.npz \
  --output outputs/doc_fingerprints.npz \
  --threshold-method z_order \
  --top-percent 0.05 \
  --normalization l2
```

**Expected Output:**

Documents processed: 1,000
Average phrases per document: 23.4
Fingerprint sparsity: 0.05 (51 active cells per document)
Output saved to: outputs/doc_fingerprints.npz
Statistics saved to: outputs/corpus_stats.json


**Output Format:**

- Sparse matrix (CSR format): `(num_documents, grid_size^2)`
- Each row is a document fingerprint
- Thresholded to top N% most active cells

**Thresholding Methods:**

- `z_order`: Preserves spatial locality using Morton curve ordering
- `value`: Keeps highest magnitude cells (may fragment spatial structure)

### Phase 6: Query Processing

Processes natural language queries and retrieves similar documents using fingerprint similarity.

**Script:** `scripts/semantic_folding/query_processing.py`

**Command-Line Arguments:**

```bash
--query STR                # Query string (required)
--phrases PATH             # Phrases JSON from Phase 1 (required)
--phrase-fps PATH          # Phrase fingerprints from Phase 4 (required)
--doc-fps PATH             # Document fingerprints from Phase 5 (required)
--corpus PATH              # Original corpus for result display (optional)
--config PATH              # Configuration file
--weighting STR            # Weighting: uniform, frequency, idf (default: uniform)
--spreading-radius INT     # Spatial spreading radius (default: 1)
--spreading-decay FLOAT    # Decay factor per distance (default: 0.5)
--top-k INT                # Number of results (default: 10)
--output-json PATH         # Save results to JSON (optional)
```

**Key Features:**

- Automatic query phrase extraction
- Spatial spreading for context expansion
- Multiple weighting schemes (uniform, frequency, IDF)
- Cosine similarity ranking

**Usage Example:**

```bash
uv run python scripts/semantic_folding/query_processing.py \
  --query "neural networks for image classification" \
  --phrases outputs/phrases.json \
  --phrase-fps outputs/phrase_fingerprints.npz \
  --doc-fps outputs/doc_fingerprints.npz \
  --corpus data/corpus.txt \
  --weighting uniform \
  --spreading-radius 1 \
  --spreading-decay 0.5 \
  --top-k 10
```

**Expected Output:**

Query: "neural networks for image classification"
Extracted phrases: ['neural networks', 'image classification']
Query fingerprint sparsity: 0.012 (12 active cells after spreading)

Top 10 Results:
1. Document 42 (similarity: 0.847)
   "Convolutional neural networks have revolutionized image classification..."
2. Document 156 (similarity: 0.792)
   "Deep learning approaches to computer vision tasks..."
3. Document 89 (similarity: 0.734)
   "Neural network architectures for visual recognition..."


**Output Format (query_results.json):**

```json
{
  "query": "neural networks for image classification",
  "extracted_phrases": ["neural networks", "image classification"],
  "query_fingerprint_sparsity": 0.012,
  "results": [
    {
      "doc_id": 42,
      "similarity": 0.847,
      "text": "Convolutional neural networks have revolutionized..."
    },
    {
      "doc_id": 156,
      "similarity": 0.792,
      "text": "Deep learning approaches to computer vision..."
    }
  ],
  "parameters": {
    "weighting": "uniform",
    "spreading_radius": 1,
    "spreading_decay": 0.5,
    "top_k": 10
  }
}
```

**Spreading Mechanism:**

Spreading expands the query fingerprint to neighboring grid cells, increasing recall:

Original query cell: (15, 22)
Radius 1 spreading: (14,21), (14,22), (14,23), (15,21), (15,22), (15,23), (16,21), (16,22), (16,23)
Weights decay by factor 0.5 per Manhattan distance


## Utility Scripts

### Context Similarity Analysis

Analyzes semantic similarity between phrase contexts in the term-context matrix.

**Script:** `scripts/semantic_folding/context_similarity.py`

**Usage:**

```bash
uv run python scripts/semantic_folding/context_similarity.py \
  --matrix outputs/term_context_matrix.npz \
  --phrases outputs/phrases.json \
  --phrase1 "neural network" \
  --phrase2 "deep learning"
```

**Output:**

Cosine similarity: 0.823
Shared contexts: 142
Unique to "neural network": 58
Unique to "deep learning": 34


### Phrase Fingerprint Inspector

Visualizes individual phrase fingerprints on the semantic grid.

**Script:** `scripts/semantic_folding/inspect_fingerprint.py`

**Usage:**

```bash
uv run python scripts/semantic_folding/inspect_fingerprint.py \
  --phrase-fps outputs/phrase_fingerprints.npz \
  --phrases outputs/phrases.json \
  --phrase "machine learning" \
  --grid-size 32
```

**Output:**

- PNG heatmap showing active grid cells
- List of neighboring phrases in adjacent cells

### Document Fingerprint Comparator

Compares two document fingerprints and identifies shared semantic regions.

**Script:** `scripts/semantic_folding/compare_docs.py`

**Usage:**

```bash
uv run python scripts/semantic_folding/compare_docs.py \
  --doc-fps outputs/doc_fingerprints.npz \
  --doc1 42 \
  --doc2 156 \
  --grid-size 32
```

**Output:**

Cosine similarity: 0.734
Shared active cells: 23
Unique to doc 42: 28
Unique to doc 156: 31
Overlap visualization saved to: outputs/doc_comparison.png


## Troubleshooting

**Issue: "Phrases extracted: 0"**

- Check corpus file encoding (must be UTF-8)
- Verify minimum frequency threshold is not too high
- Ensure spaCy model is installed: `python -m spacy download en_core_web_sm`

**Issue: "Matrix sparsity too high (>0.99)"**

- Increase `min_freq` in phrase extraction
- Reduce corpus size or use more focused domain text
- Check for data quality issues (e.g., excessive boilerplate)

**Issue: "t-SNE convergence warning"**

- Increase `max_iter` (try 2000-5000)
- Adjust `perplexity` (try 10-50 range)
- Switch to UMAP for faster convergence: `--method umap`

**Issue: "Query returns no results"**

- Reduce `spreading_radius` to 0 for exact matching
- Check if query phrases exist in `phrases.json`
- Lower `min_similarity` threshold
- Verify document fingerprints are not over-thresholded (increase `top_percent`)

**Issue: "Out of memory during semantic space generation"**

- Reduce `grid_size` (try 16 instead of 32)
- Use PCA instead of t-SNE: `--method pca`
- Enable sparse matrices: set `use_sparse: true` in config
- Process in batches by limiting `max_phrases` temporarily

**Issue: "Configuration file not loading"**

- Verify YAML syntax (use online validator)
- Check file path is correct relative to script location
- Ensure all required sections are present
- Use `--config` flag explicitly in command

Configuration complete. The pipeline now supports centralized YAML-based configuration with command-line overrides.