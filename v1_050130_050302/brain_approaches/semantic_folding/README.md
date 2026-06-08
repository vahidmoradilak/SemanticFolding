# Semantic Folding: Interpretable Semantic Space Embeddings

## Overview

**Status**: 🔄 Core Pipeline Complete | Evaluation & Knowledge Graph Integration Planned

Semantic Folding is a novel approach for constructing interpretable, grid-based semantic space embeddings from textual corpora. Unlike black-box embeddings (word2vec, BERT) or traditional dimensionality reduction (UMAP, t-SNE, PCA), semantic folding creates spatially-organized representations where:

- **Phrases** are represented as sparse spatial fingerprints showing their semantic distribution
- **Documents** are composite fingerprints aggregating constituent phrase semantics
- **Semantic relationships** are preserved through graph-based positioning and spatial proximity
- **Grid structure** enables efficient similarity computation and interpretable visualization

## Production-Ready Pipeline

### Core Components (✅ Completed)

The semantic folding pipeline has been modernized into a production-ready system with comprehensive engineering and academic documentation:

#### Pipeline Modules
- **`lib.py`**: Shared utility layer providing consistent implementations of core operations (normalization, validation, Morton encoding, sparsification)
- **`phrase_extractor.py`**: Modern phrase extraction with 1-4 word n-grams, quality filtering, and spaCy integration
- **`term_context.py`**: Sparse matrix term-context construction with TF-IDF normalization (95% memory savings)
- **`semantic_space.py`**: Force-directed graph layout with configurable grid sizes and spatial optimization
- **`phrase_fingerprints.py`**: Efficient fingerprint generation from semantic coordinates with validation and Morton encoding
- **`doc_fingerprints.py`**: Document fingerprint aggregation with IDF weighting, Z-order thresholding, and comprehensive metadata
- **`semantic_folder.py`**: Interactive TUI orchestration with progress tracking, resume capability, and error recovery
- **`scratchpad.py`**: Command-line orchestration with comprehensive logging and batch processing

#### Academic Documentation
- **`lib.md`**: Comprehensive documentation of shared utility functions with mathematical foundations
- **`phrase_extractor.md`**: Theoretical foundation and implementation details for phrase extraction
- **`term_context.md`**: Sparse matrix construction methodology and TF-IDF normalization theory
- **`sematic_space.md`**: Graph-based semantic space construction with force-directed layout algorithms
- **`phrase_fingerprints.md`**: Fingerprint generation theory with Morton encoding and sparsification
- **`doc_fingerprints.md`**: Document-level aggregation theory with Z-order thresholding and weighting schemes

#### Supporting Tools
- **`analyze_fingerprints.py`**: Statistical analysis and quality metrics for generated fingerprints
- **`context-similarity.py`**: Context similarity computation and validation
- **`query-processing.py`**: Query fingerprint generation and similarity search
- **`lance_storage.py`**: LanceDB integration for efficient vector storage and retrieval

### Key Improvements

- **Scalability**: Handles corpora from 1K to 50K+ documents efficiently
- **Memory Efficiency**: Sparse matrices reduce memory usage by 95%
- **Robustness**: Comprehensive validation, fallback mechanisms, and error handling
- **Consistency**: Shared `lib.py` ensures uniform behavior across all pipeline stages
- **Quality Optimization**: Tuned parameters for optimal semantic space distribution
- **Advanced Processing**: Morton encoding for spatial locality, Z-order thresholding for coherent sparsification
- **Comprehensive Logging**: Dual console/file output with color coding and debug modes
- **Resume Capability**: Automatic recovery from interruptions with checkpoint management

## Advanced Features

### Shared Utility Layer (`lib.py`)

All pipeline stages leverage a consistent set of core functions:

- **`normalize_phrase()`**: Consistent phrase key formatting (lowercase, whitespace normalization)
- **`is_valid_phrase_structure()`**: Structural validation to filter malformed phrases
- **`normalize_fingerprint()`**: Unified normalization (L1, L2, binary, raw) for activation maps
- **`xy_to_morton()`**: Z-order curve encoding for spatial locality preservation
- **`sparsify_fingerprint()`**: Value-based thresholding with configurable percentiles
- **`compute_idf_weights()`**: Corpus-level IDF statistics for phrase importance weighting
- **`compute_fingerprint_diversity()`**: Activation diversity metrics for quality assessment
- **`export_fingerprints_to_numpy()`**: Efficient serialization to NumPy format

### TF-IDF Matrix Normalization

Reduces the dominance of high-frequency words in semantic relationships:

- **Problem**: Common words like "it", "that", "the" can overwhelm meaningful semantic connections
- **Solution**: Applies TF-IDF weighting to term-context matrix entries
- **Formula**: $\text{TF-IDF} = \text{TF} \times \log\frac{N}{\text{DF}}$ where TF is term frequency, DF is document frequency
- **Benefit**: Balances semantic relationships by down-weighting ubiquitous terms
- **Implementation**: `normalize_matrix: true` in configuration (enabled by default)

### Z-order Curve Thresholding

Creates spatially coherent document fingerprints:

- **Problem**: Value-based thresholding fragments semantically coherent regions
- **Solution**: Traverses activation map via Z-order curve (Morton encoding) to preserve spatial locality
- **Method**: `xy_to_morton()` converts 2D coordinates to 1D indices, sorted traversal selects top activations
- **Benefit**: Maintains semantic coherence by preferentially selecting contiguous high-activation regions
- **Fallback**: `sparsify_fingerprint()` provides value-based thresholding when Z-order is disabled

### Document Fingerprint Sparsification

Creates focused and interpretable document representations:

- **Problem**: Document fingerprints may contain noise from weakly relevant terms
- **Solution**: Retains only the top N% most activated cells after aggregation
- **Method**: Z-order traversal or value-based thresholding depending on configuration
- **Benefit**: Improves semantic specificity and reduces storage requirements
- **Configuration**: `doc_top_percent: 0.05` (keeps top 5%, configurable)

### Phrase Structure Validation

Ensures quality and consistency across the pipeline:

- **Problem**: Malformed phrases (empty strings, excessive whitespace, invalid characters) can corrupt fingerprints
- **Solution**: `is_valid_phrase_structure()` validates phrases after normalization
- **Checks**: Non-empty, reasonable length, valid characters, proper whitespace
- **Benefit**: Early filtering prevents downstream errors and improves fingerprint quality
- **Metrics**: Validation rate tracked in pipeline statistics

## Quick Start

### Interactive TUI (Recommended)
```bash
# Install dependencies
uv sync

# Launch interactive interface
uv run python brain_approaches/semantic_folding/semantic_folder.py

### Command Line (Advanced)

# Run on MuSiQue corpus (default)
uv run python brain_approaches/semantic_folding/scratchpad.py \
  --corpus_path data/HippoRAG2/dataset/musique_corpus.json

# Run on custom corpus
uv run python brain_approaches/semantic_folding/scratchpad.py \
  --corpus_path /path/to/your/corpus.json \
  --grid_size 32

# Check results
ls outputs/$(date +%Y%m%d)_*/fingerprints/ | wc -l  # 25K+ fingerprint files
```
## Configuration

The semantic folding pipeline is highly configurable through `config/semantic_folding.yml`:

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 32 | Semantic space grid size (8, 16, 32, 64, 128) |
| `max_edges` | 200 | Maximum edges in semantic graph |
| `edge_threshold` | 0.05 | Minimum similarity for graph connections |
| `normalize_matrix` | true | Apply TF-IDF normalization to term-context matrix |
| `doc_top_percent` | 0.05 | Document fingerprint threshold percentage (Z-order or value-based) |
| `use_z_order` | true | Enable Z-order curve thresholding for spatial coherence |
| `normalization` | l2 | Fingerprint normalization method (l1, l2, binary, raw) |

### Quality vs Speed Trade-offs

**Maximum Quality** (Recommended):
yaml
grid_size: 64
max_edges: 500
edge_threshold: 0.01
normalize_matrix: true
doc_top_percent: 0.03
use_z_order: true
normalization: l2

**Balanced Performance**:
yaml
grid_size: 32
max_edges: 200
edge_threshold: 0.05
normalize_matrix: true
doc_top_percent: 0.05
use_z_order: true
normalization: l2

**Fast Processing**:
yaml
grid_size: 16
max_edges: 100
edge_threshold: 0.1
normalize_matrix: false
doc_top_percent: 0.1
use_z_order: false
normalization: binary

### Output Structure

outputs/YYYYMMDD_HHMMSS/
├── corpus.txt                 # Processed corpus
├── phrases.txt               # 25K+ filtered phrases with frequencies
├── term_context_matrix.npz   # Sparse matrix (827K entries, 0.28% density)
├── context_coordinates.csv   # Grid coordinates for semantic space
├── fingerprints/             # 25K+ phrase fingerprint files
├── doc_fingerprints/         # 11K+ document fingerprints + metadata
├── logs/pipeline.log         # Comprehensive execution log
└── visualizations/           # Heatmaps and graphs (if matplotlib available)

## Core Algorithm

### Pipeline Overview

The semantic folding pipeline consists of six core stages with comprehensive validation:

1. **Phrase Extraction** (`phrase_extractor.py`): Extract 1-4 word n-grams with quality filtering and frequency ranking
2. **Term-Context Matrix** (`term_context.py`): Sparse matrix construction with optional TF-IDF normalization
3. **Semantic Space Construction** (`semantic_space.py`): Force-directed graph layout mapping contexts to grid coordinates
4. **Phrase Fingerprint Generation** (`phrase_fingerprints.py`): Convert phrase occurrences to spatial fingerprints with validation
5. **Document Fingerprint Aggregation** (`doc_fingerprints.py`): Combine phrase fingerprints with weighting and thresholding
6. **Storage & Retrieval** (`lance_storage.py`): Efficient vector storage and similarity search via LanceDB

### Technical Details

#### 1. Phrase Extraction
**Input**: Text corpus (JSON/CSV format)
**Output**: Frequency-sorted phrase list (`phrases.txt`)

**Algorithm**:
- Uses spaCy NLP pipeline for linguistic analysis
- Extracts 1-4 word n-grams with configurable limits
- Filters by frequency, length, and structural validity
- Applies `normalize_phrase()` for consistent formatting
- Validates with `is_valid_phrase_structure()` before output

**Key Parameters**:
- `max_ngram`: Maximum phrase length (default: 4)
- `min_phrase_freq`: Minimum occurrence threshold
- `use_spacy`: Enable advanced linguistic processing

#### 2. Term-Context Matrix Construction
**Input**: Phrases list, corpus text
**Output**: Sparse matrix (`term_context_matrix.npz`)

**Algorithm**:
- Creates sparse CSR matrix: contexts × phrases
- Cell values: phrase occurrence counts in each context
- Optional TF-IDF normalization via `normalize_matrix` parameter
- 95% memory savings compared to dense representation

**Matrix Structure**:

Context ID | phrase_1 | phrase_2 | ... | phrase_n
-----------|----------|----------|-----|----------
1          | 2.3      | 0        | ... | 1.7
2          | 0        | 4.1      | ... | 0
...

#### 3. Semantic Space Construction
**Input**: Term-context matrix
**Output**: Semantic coordinates (`context_coordinates.csv`)

**Algorithm**:
1. **Graph Construction**:
   - Nodes: contexts (documents)
   - Edges: weighted by cosine similarity of context vectors
   - Threshold: `edge_threshold` filters weak connections
   - Max edges: `max_edges` limits connectivity per node

2. **Force-Directed Layout**:
   - NetworkX `spring_layout` with tuned parameters
   - `k = grid_size / 2` for optimal spacing
   - Reproducible positioning via fixed seed

3. **Grid Mapping**:
   - Maps continuous 2D coordinates to discrete grid
   - Grid size: `grid_size × grid_size` (configurable)
   - Coordinate transformation preserves relative positions

**Key Parameters**:
- `grid_size`: Resolution of semantic space (8-128)
- `max_edges`: Connectivity constraint for graph sparsity
- `edge_threshold`: Minimum similarity for edge creation

#### 4. Phrase Fingerprint Generation
**Input**: Context coordinates, term-context matrix
**Output**: Phrase fingerprint matrices (`fingerprints/*.txt`)

**Algorithm**:
For each phrase:
1. Normalize phrase key via `normalize_phrase()`
2. Validate structure via `is_valid_phrase_structure()`
3. Find contexts where phrase appears (matrix lookup)
4. Map context IDs to semantic coordinates
5. Create sparse activation map on grid
6. Apply `normalize_fingerprint()` for consistent representation
7. Optional: Apply `sparsify_fingerprint()` for multi-level thresholding

**Integration with `lib.py`**:
- Consistent phrase normalization ensures vocabulary matching
- Early validation prevents malformed fingerprints
- Morton encoding via `xy_to_morton()` for spatial indexing
- Unified normalization for downstream compatibility

#### 5. Document Fingerprint Generation
**Input**: Corpus text, phrase fingerprints
**Output**: Document fingerprint matrices (`doc_fingerprints/*.txt`)

**Algorithm**:
For each document:
1. Extract phrases and normalize via `normalize_phrase()`
2. Validate phrases via `is_valid_phrase_structure()`
3. Match against phrase vocabulary
4. Aggregate phrase fingerprints with optional weighting:
   - Uniform: $w_i = 1$
   - Frequency: $w_i = \text{count}(p_i, D)$
   - IDF: $w_i = \log\frac{N}{df(p_i)}$ via `compute_idf_weights()`
5. Normalize via `normalize_fingerprint()`
6. Apply Z-order thresholding via `xy_to_morton()` or value-based via `sparsify_fingerprint()`
7. Compute diversity metrics via `compute_fingerprint_diversity()`

**Mathematical Foundation**:
$$F_D = \text{threshold}\left(\text{normalize}\left(\sum_{i=1}^{n} w_i \cdot F_{p_i}\right)\right)$$

## Key Innovations

### 1. Interpretable Semantic Spaces
- Unlike black-box embeddings (word2vec, BERT), semantic folding produces spatially-organized representations
- Each grid position has semantic meaning based on context proximity in the learned space
- Phrases cluster spatially based on co-occurrence patterns and relational semantics

### 2. Spatial Coherence via Z-order Curves
- Morton encoding (`xy_to_morton()`) preserves spatial locality during thresholding
- Contiguous high-activation regions are preferentially selected over scattered peaks
- Maintains topological structure of the semantic space in sparse representations

### 3. Consistent Pipeline Integration
- Shared `lib.py` ensures uniform behavior across all stages
- Phrase normalization and validation prevent downstream errors
- Unified normalization enables fair comparison and composition of fingerprints

### 4. Flexible Weighting and Thresholding
- Multiple weighting schemes (uniform, frequency, IDF) for different use cases
- Configurable normalization methods (L1, L2, binary, raw) for various similarity metrics
- Adaptive thresholding (Z-order or value-based) balances quality and sparsity

## Quality Metrics

### Sparsity
$$\text{Sparsity}(F) = 1 - \frac{|F|}{|G|}$$
where $|F|$ is active coordinates and $|G|$ is total grid size. Target: 95-99%.

### Coverage
$$\text{Coverage}(D) = \frac{|\{p \in D : p \in \text{Vocabulary}\}|}{|\text{unique phrases in } D|}$$
Measures proportion of document phrases matching the learned vocabulary.

### Validation Rate
$$\text{ValidationRate}(D) = \frac{|\{p \in D : \text{is\_valid\_phrase\_structure}(p)\}|}{|\text{normalized phrases in } D|}$$
Tracks fraction of phrases passing structural validation.

### Fingerprint Diversity
Computed via `compute_fingerprint_diversity()`, measures activation spread across the semantic grid. Low diversity indicates over-concentration; high diversity suggests broad coverage.

## Experimental Framework

### Comparative Notebooks (Archive)

The `notebooks/` directory contains original research implementations demonstrating the algorithm and comparative studies:

- **`Semantic_Space_Construction.ipynb`**: Complete end-to-end pipeline with Google Colab integration
- **`ground-up.ipynb`**: Foundational phrase extraction and processing experiments
- **`pre_processing.ipynb`**: Text preprocessing with spaCy lemmatization and cleaning
- **`Document_representation.ipynb`**: Traditional TF-IDF + SVD baseline
- **`Document_clustering.ipynb`**: Self-Organizing Maps (SOM) clustering
- **`Umap_fingerprint.ipynb`**: UMAP-based dimensionality reduction with grid fingerprinting
- **`tsne_fingerprint.ipynb`**: t-SNE visualization and fingerprint generation
- **`ISOMAP.ipynb`**: ISOMAP manifold learning comparison
- **`SOM_fingerprint.ipynb`**: SOM-based semantic fingerprinting

**Note**: These notebooks represent the original research phase. The production pipeline (`lib.py`, `phrase_extractor.py`, etc.) supersedes these implementations with improved engineering, consistency, and scalability.

## Technical Requirements

### Dependencies
```bash
# Core dependencies
uv add spacy networkx numpy scipy scikit-learn pyyaml questionary

# Optional visualization
uv add matplotlib seaborn plotly

# LanceDB integration
uv add lancedb pyarrow

# Download spaCy model
python -m spacy download en_core_web_sm
```
### Input Format Requirements
**Corpus File** (JSON):
```json
[
  {"id": "1", "text": "The machine learning algorithm processes data efficiently."},
  {"id": "2", "text": "Neural networks are powerful computational models."}
]
```
**Corpus File** (CSV):
```csv
id,text
1,The machine learning algorithm processes data efficiently.
2,Neural networks are powerful computational models.
```
## Performance Characteristics

### Computational Complexity
- **Phrase Extraction**: $O(N)$ where $N$ = corpus size
- **Matrix Construction**: $O(C \times P)$ where $C$ = contexts, $P$ = phrases
- **Graph Construction**: $O(C^2 \times P)$ - quadratic in contexts (mitigated by `max_edges`)
- **Fingerprint Generation**: $O(P \times C \times D^2)$ where $D$ = grid dimensions
- **Document Aggregation**: $O(D_p \times D^2)$ where $D_p$ = phrases per document

### Scalability Considerations
- **Memory**: Sparse matrices reduce memory by 95% compared to dense representation
- **Phrase Filtering**: Frequency thresholds and validation reduce vocabulary size
- **Graph Sparsity**: `max_edges` parameter limits connectivity for large corpora
- **Grid Resolution**: Higher `grid_size` increases precision but grows quadratically

### Optimization Strategies
- **Batch Processing**: Process documents in batches to manage memory
- **Cached Fingerprints**: Load phrase fingerprints once and reuse across documents
- **Sparse Accumulation**: Maintain activation maps as sparse dictionaries during aggregation
- **Early Validation**: Filter malformed phrases before any fingerprint operations

## Future Work

### Knowledge Graph and AI Agent Integration

**Vision**: Transform semantic folding from purely distributional semantics to knowledge-grounded spatial embeddings through AI agent-orchestrated knowledge graph construction and retrieval.

#### 1. AI Agent-Driven Knowledge Graph Construction

**Open Information Extraction (OIE) Pipeline**
- Extract (subject, relation, object) triples using LLM-based OIE (GPT-4o-mini, Llama-3.3-70B, or Phi-3-mini)
- Apply schema-constrained prompting with strict output formatting (following LightRAG approach)
- Normalize entities via embedding similarity (threshold: 0.85-0.95) to merge synonyms
- Build typed knowledge graphs with ontology-guided relation categories

**Integration with Semantic Folding**
- Use KG entities as high-quality phrase candidates
- Incorporate KG triples as relational contexts in term-context matrix
- Apply entity type constraints to force-directed layout positioning
- Link phrases to KG entities for semantic grounding

#### 2. HippoRAG-Inspired Retrieval Enhancement

**Graph-Augmented Retrieval**
- Implement Personalized PageRank (PPR) over knowledge graphs for associative retrieval
- Combine dense passage embeddings with graph structure traversal
- Apply LLM-based recognition memory for triple filtering
- Target ~7% improvement over pure embedding baselines on associative tasks

**Hybrid Architecture**
- Multi-modal retrieval: semantic folding fingerprints + dense embeddings + graph traversal
- Agent-orchestrated strategy selection based on query type
- Feedback-driven semantic space refinement
- Spatial structure for explainable retrieval results

#### 3. AI Agent Optimization Framework

**Adaptive Parameter Tuning**
- Optimize `grid_size`, `max_edges`, `edge_threshold` based on corpus characteristics
- Learn dynamic phrase weighting schemes from retrieval feedback
- Validate and iteratively refine semantic coherence
- Translate natural language queries to optimal fingerprint representations

#### Implementation Roadmap

**Phase 1: OIE Foundation** (Months 1-2)
- Deploy LLM-based triple extraction with constrained prompting
- Implement entity normalization and synonym merging
- Build initial knowledge graph storage (python-igraph)

**Phase 2: KG-Semantic Folding Integration** (Months 3-4)
- Extract KG entities as phrase candidates
- Enrich term-context matrix with relational contexts
- Apply entity-aware constraints to spatial layout

**Phase 3: HippoRAG Retrieval** (Months 5-6)
- Implement PPR-based graph traversal
- Integrate dense embeddings with graph structure
- Benchmark on associative retrieval tasks

**Phase 4: AI Agent Orchestration** (Months 7-8)
- Build agent framework for parameter optimization
- Implement feedback-driven refinement loops
- Deploy hybrid retrieval strategy selection

**Phase 5: Evaluation & Refinement** (Months 9-10)
- Benchmark on standard datasets (HotpotQA, MuSiQue, 2WikiMultihopQA)
- Analyze explainability and interpretability
- Optimize for production deployment

#### Research Questions

- Can KG structure inform grid positioning beyond co-occurrence patterns?
- What triple extraction accuracy is sufficient for meaningful retrieval improvements?
- How does entity normalization threshold affect downstream semantic space quality?
- What is the optimal balance between fingerprint sparsity and graph density?
- Can spatial fingerprints enable more interpretable AI agent reasoning chains?

#### Technical Requirements

**Models & Tools**
- OIE: GPT-4o-mini (cost-effective, 82.1% accuracy) or Phi-3-mini (3.8B params)
- Graph: python-igraph for PPR computation
- Embeddings: sentence-transformers for entity similarity
- Storage: LanceDB for hybrid vector-graph indexing

**Key Design Principles**
- Prompt engineering with strict delimiters (e.g., `<|>` tuples, `##` records)
- Entity normalization before graph construction
- Minimal, normalized triple schemas
- Incremental graph updates for streaming corpora

### Additional Extensions

**Beyond Knowledge Graphs**
- **Multi-Modal Integration**: Combine text with structured data and images
- **Temporal Dynamics**: Track semantic space evolution over time
- **Hierarchical Spaces**: Multi-resolution grids for different granularities
- **Cross-Lingual Alignment**: Project multiple languages into shared semantic space
- **Distributed Processing**: Parallel fingerprint generation for massive corpora
- **Advanced Compression**: Ultra-sparse representations for billion-scale vocabularies

## References and Related Work

### Foundational Work
- **Distributional Semantics**: Harris (1954), Firth (1957)
- **Vector Space Models**: Salton et al. (1975)
- **Graph-based Methods**: Sahlgren (2006) random indexing
- **Manifold Learning**: Roweis & Saul (2000) locally linear embedding
- **Sparse Distributed Representations**: Kanerva (2009) hyperdimensional computing

### Knowledge Graph & Retrieval
- **GraphRAG**: Microsoft Research (2024) - Graph-augmented retrieval
- **LightRAG**: Schema-constrained OIE with LLMs
- **CLARE**: Constrained prompting for triple extraction (82.1% accuracy with GPT-4o-mini)
- **HippoRAG**: PPR-based associative retrieval with 7% improvement over baselines

### Implementation References
- **spaCy**: Linguistic processing and NLP pipelines
- **NetworkX**: Graph algorithms and force-directed layouts
- **SciPy Sparse**: Efficient sparse matrix operations
- **LanceDB**: Vector database for similarity search
- **python-igraph**: High-performance graph computations

### Key Technical Insights
1. **Preprocessing Importance**: Lemmatization and stop-word removal significantly impact phrase quality
2. **Parameter Sensitivity**: Grid dimensions affect fingerprint resolution vs. sparsity trade-off
3. **Spatial Coherence**: Z-order thresholding preserves semantic structure better than value-based methods
4. **Validation Benefits**: Early phrase validation prevents downstream errors
5. **Prompt Design**: Constrained prompting yields cleaner triples regardless of model size
6. **Entity Normalization**: Embedding similarity (0.85-0.95) effectively merges synonyms
7. **Small Models Suffice**: GPT-4o-mini achieves near-GPT-4o quality at lower cost

---

**Note**: This implementation provides a foundation for semantic folding research and applications. The knowledge graph integration roadmap represents a significant enhancement that transforms distributional semantics into knowledge-grounded spatial embeddings. Parameters should be tuned for specific domains and corpus characteristics. The production pipeline supersedes experimental notebooks with improved engineering, consistency, and scalability.
