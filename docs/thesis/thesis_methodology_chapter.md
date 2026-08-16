# Methodology Chapter Outline

## 3. Semantic Folding Pipeline (Detailed)

### 3.1 Overview
The Semantic Folding pipeline transforms textual documents into 2D grid representations enabling similarity computation via structural metrics. The pipeline consists of seven sequential steps, each building upon the previous to create fingerprints suitable for efficient retrieval.

### 3.2 Step 1: Phrase Extraction
- **Objective**: Extract meaningful phrases from raw text for further processing
- **Arabic (lib.py:extract_raw_phrases_ar_fa)**:
  - Unigrams: tokens ≥ 2 chars, not in `_AR_FUNCTION_WORDS`
  - Bigrams: NLTK POS pattern `[NN/NNS/NNP/NNPS/JJ/JJR/JVS/VBN] + [N*]`, both tokens must not be in `_AR_FUNCTION_WORDS`
  - Trigrams: strict POS filter `[NN/JJ/VBN] + [NN/JJ/VBN] + [NN/JJ/VBN]`, plus function-word guard
  - Clitic stripping guard: stem ≥ 2 chars AND not a function word (preserves `الله`)
  - `_AR_FUNCTION_WORDS` includes `ال`, `و`, `ف`, `ب`, `ل`, `ك`, `ک`, `س`, `بال`, `فل`, `ول`, `فب`
  - `_AR_CLITICS` includes `ال`, `و`, `ف`, `ب`, `ل`, `ک`, `ک`, `س`, `بال`, `فل`, `ول`, `فب`
  - `min_word_length = 2` for Arabic (preservesحروف مقطعه like `طه`, `يس`)
- **English (phrase_extractor.py:extract_raw_phrases_spacy)**:
  - Unigrams: tokens ≥ 3 chars, `pos_` ∈ {NOUN,PROPN,ADJ,VERB,ADV}, not `is_stop`
  - Bigrams: spaCy noun chunks + left modifiers + compound chains + conjunction expansion
  - Trigrams: spaCy noun chunks + left-anchored sub-spans
- **Parameters**: `min_word_length = 2` (Arabic), default English stopword removal

### 3.3 Step 2: Term Context Construction
- **Objective**: Build term-document-context matrix with TF-IDF weighting
- **Process**:
  - Construct vocabulary from extracted phrases across corpus
  - Compute phrase-document frequency counts
  - Apply IDF weighting: `idf(t) = log(N/n_t)` where N = total documents, n_t = docs containing term t
  - Construct term-context matrix with TF-IDF weights
- **Parameters**: 
  - `min_freq`: minimum document frequency to keep a phrase (default: 1)
  - `max_doc_freq`: maximum document frequency (0 = unlimited, default: 0)
  - `weighting`: "uniform", "frequency", or "idf"

### 3.4 Step 3: Semantic Space Construction
- **Objective**: Reduce high-dimensional term-context matrix to 2D grid via dimensionality reduction
- **Process**:
  - Apply UMAP/t-SNE/PCA for dimensionality reduction to grid_size × grid_size (default 64×64)
  - Morton (Z-order) ordering for spatial indexing
  - Key parameters:
    - `grid_size`: 64×64 (optimal for 20-doc corpus; must match across Steps 3–6)
    - `use_morton`: true (Morton order) or false (row-major)
    - `method`: "umap" (default), "pca", "tsne", or "learned"
    - `umap_n_neighbors`: 15 (default)
    - `umap_min_dist`: 0.1 (default)
    - `umap_metric`: "cosine" (default)
- **Sweep results**: 64×64 beats 128×128 (MRR 1.000 vs 0.900, AP 0.869 vs 0.836)

### 3.4.1 Morton Encoding
- Maps 2D coordinates (x, y) to 1D index via `morton_encode(x, y, grid_size)`
- Maps 1D index to 2D via `morton_to_xy(flat_idx, grid_size)`
- Enables efficient spatial queries and grid construction

### 3.5 Step 4: Phrase Fingerprints
- **Objective**: Convert phrase embeddings into sparse distributed representations (SDRs) in the 2D grid
- **Process**:
  - Build 2D grid from phrase positions using `build_document_fingerprint_2d`
  - Each phrase is placed at its `(y, x)` coordinate via `index_to_xy[row_idx]`
  - Phrase weights are accumulated: `grid_2d[y, x] += phrase_fp_matrix[row_idx]`
  - Apply Gaussian smoothing: `sparsify_to_sdr_topological(grid_2d, top_percent=0.05, grid_size=64, min_peak_distance=2, smoothing_sigma=1.5)`
  - Convert to sparse matrix: `csr_matrix(result_1d.reshape(1, -1))`
- **Key parameters**:
  - `top_percent`: 0.05 (0.10 dilutes signal; 0.05 gives better discrimination)
  - `smoothing_sigma`: 1.5 (Gaussian window width)
  - `no_smooth`: false (to enable Gaussian blur)

### 3.6 Step 5: Document Fingerprints
- **Objective**: Construct document-level fingerprints from the semantic space
- **Process**:
  - For each document, aggregate phrase fingerprints weighted by document frequency
  - Apply document-specific smoothing: `smoothing_sigma` (default 1.5)
  - Add `--normalize-after-spreading` option (default true: L2-normalize after spreading)
  - Output: `doc_fingerprints` directory containing sparse matrices
- **Key parameters**:
  - `--grid-size`: 64 (must match Step 3)
  - `--top-percent`: 0.10 (from Step 3)
  - `--normalize-method`: "l2"
  - `--min-peak-distance`: 2

### 3.6.1 Document Grid Construction
- For each document fingerprint:
  - Load flat vector from Step 4 output
  - Reshape or unflatten using `build_index_to_xy_table(grid_size, use_morton)`
  - Populate 2D grid: `grid[y, x] = flat[idx]` where `(y, x) = build_index_to_xy_table(grid_size, use_morton)[idx]`
  - Result: `doc_grids_2d[doc_id]` = `np.ndarray` of shape `(grid_size, grid_size)`

### 3.7 Step 6: Phrase Fingerprints (Custom/Ar-En)
- **Custom AR-EN benchmark**: Bilingual Arabic-English query processing
- **Process**:
  - Extract mixed-language queries (Arabic + English phrases)
  - Build bilingual corpus from user-supplied corpus.txt
  - Run Steps 1–5 on the bilingual corpus
  - Generate query strings via `make_mixed_query()` function
- **Variants evaluated**:
  - Pure SF
  - SF+SPLADE Linear (α=0.3)
  - SF+SPLADE RRF (k=60)
  - BM25 baseline

### 3.8 Step 7: Query Processing & Similarity Scoring
- **Objective**: Rank documents against a query using the pre-constructed fingerprints
- **Key parameters** (set in config/exec_state.yml):
  - `grid_size`: 64 (must match Steps 3–5)
  - `spreading_steps`: 1 (default)
  - `spreading_decay`: 0.5 (default)
  - `min_similarity`: 0.0 (default)
  - `weighting`: "idf" (best; uniform drops C17 ranking and loses C00)
  - `normalization`: "l2" (query), "√nnz" (document)
  - `geometric`: false (optional 3×3 spatial adjacency kernel)
  - `hybrid`: false (optional SF+BM25 scoring)
  - `splade`: false (optional SF+SPLADE)
  - `fusion_method`: "linear" or "rrf" (when splade enabled)
  - `rrf_k`: 60 (when rrf fusion enabled)
  - `min_word_length`: 3 (default)
  - `min_phrase_length`: 2 (default)
  - `retrieval_backend`: "numpy" (default) or "lancedb"
  - `image_search`: false (new; enables SSIM-based ranking)
  - `ssim_sigma`: 1.5 (Gaussian window sigma)
  - `ssim_region`: "active" (masks to cells active in either image) or "full"

### 3.8.1 Similarity Metrics

#### 3.8.1.1 Cosine Similarity
```
cos(q, d) = (q · d) / (||q|| ||d||)
```
- Standard vector cosine similarity between query and document fingerprints
- Fast computation, permutation-invariant (order-independent)

#### 3.8.1.2 Structural Similarity (SSIM)
```
SSIM(q, d) = [ (2μ_qμ_d + C1) / (μ_q² + μ_d² + C1) ] ×
             [ (2σ_qσ_d + C2) / (σ_q² + σ_d² + C2) ] ×
             [ (σ_qd + C3) / (σ_qσ_d + C3) ]
```
where:
- μ_q, μ_d = local means (gaussian filtered)
- σ_q, σ_d = local standard deviations
- σ_qd = local covariance
- C1 = (K1·L)², C2 = (K2·L)², with K1=0.01, K2=0.03, L=255 (image dynamic range)
- Alternatively, with normalized activations: C1, C2 adjusted accordingly

**SSIM Region Options**:
- `"active"`: Mask to cells active (non-zero) in either query or document image → recommended for sparse grids
- `"full"`: Use entire grid → broader comparison, less sparse-sensitive

#### 3.8.1.3 Hybrid Scoring
```
score = α · cosine(q, d) + (1 − α) · SSIM(q, d)
```
- `α` (hybrid_alpha): 0.3 (best per tuning), 0.5, 0.7
- Combines vector similarity with spatial structure

#### 3.8.1.4 SPLADE Fusion
- Linear: `score = α · SF_score + (1 − α) · SPLADE_score`, α=0.3 (default)
- RRF (Reciprocal Rank Fusion): `score = sum(1/(k + rank_i))` over multiple rankers, k=60

### 3.8.2 Retrieval Backends
- **numpy**: Flat vector cosine SSIM via `rank_documents()` — default, works with all configs
- **lancedb**: ANN index for large corpora; **incompatible with image_search** (requires numpy backend)

### 3.9 Evaluation Metrics
- **MRR** (Mean Reciprocal Rank): `1 / rank_of_first_relevant`
- **AP** (Average Precision): Area under precision-recall curve
- **P@K** (Precision at K): `relevant_docs_in_top_K / K`
- **R@K** (Recall at K): `relevant_docs_in_top_K / total_relevant`
- **NDCG@K** (Normalized Discounted Cumulative Gain): 
  ```
  DCG@K = sum_{i=1}^{K} (rel_i / log2(i+1))
  IDCG@K = DCG@K with ideal ordering
  NDCG@K = DCG@K / IDCG@K
  ```

### 3.10 Experimental Setup
- **Datasets**: QA-sample (20 docs, C00–C19), Quran (6,236 ayahs), Belebele, PubMedQA, PopQA, NarrativeQA
- **Hardware**: CPU-based (no GPU acceleration)
- **Software**: Python 3.x, numpy, scipy, scikit-learn, plotly, pyyaml
- **Environment**: `.venv\scripts\python` (Windows), `en_core_web_sm` spaCy model

### 3.11 Parameter Tuning
- Key grid findings: 64×64 optimal (beats 128×128: MRR 1.000 vs 0.900, AP 0.869 vs 0.836)
- Spreading: radius=1, decay=0.5 effective; spread=2 doesn't improve
- top_percent: 0.05 better than 0.10 on Quran corpus
- Query weighting: IDF best; uniform drops C17 ranking and loses C00
- Normalization: L2 for query, √nnz for document fingerprints
- Geometric: optional 3×3 kernel, rewards nearby (not just exact) cell overlap
- SPLADE fusion: Linear α=0.3 best on 3/4 datasets (Belebele, NarrativeQA, PubMedQA); PopQA stays pure SF

---

## 3.12 Summary
The Semantic Folding pipeline transforms textual documents into 2D grid representations where semantic content is encoded spatially. The seven-step process—from phrase extraction through similarity scoring—enables efficient retrieval using both traditional cosine similarity and newer SSIM-based image similarity metrics. Parameter tuning significantly impacts retrieval effectiveness, with the 64×64 grid and IDF weighting emerging as particularly important factors. The pipeline supports both pure symbolic retrieval and learned neural hybrid approaches via SPLADE fusion.

---

## 3.13 Summary of Key Variables

| Variable | Description | Default | Range |
|----------|-------------|---------|-------|
| `grid_size` | Grid dimensions (N×N) | 64 | 32–128 |
| `use_morton` | Morton order indexing | true | true/false |
| `spreading_steps` | Number of spreading iterations | 1 | 0–5 |
| `spreading_decay` | Decay factor per step | 0.5 | 0.1–1.0 |
| `top_percent` | Percent of peaks to retain | 0.05 | 0.01–0.20 |
| `weighting` | TF-IDF weighting scheme | "idf" | "uniform"/"frequency"/"idf" |
| `normalization` | Query/doc normalization | "l2"/"√nnz" | — |
| `weighting` | Query weighting | "idf" | — |
| `geometric` | 3×3 spatial adjacency kernel | false | true/false |
| `hybrid` | SF+BM25 hybrid | false | true/false |
| `splade` | SF+SPLADE | false | true/false |
| `fusion_method` | Fusion type: "linear" or "rrf" | "linear" | — |
| `rrf_k` | RRF constant | 60 | 10–100 |
| `min_word_length` | Minimum phrase length | 3 (Arabic: 2) | 1–5 |
| `min_phrase_length` | Minimum phrase length | 2 | 1–5 |
| `image_search` | SSIM-based ranking | false | true/false |
| `ssim_sigma` | Gaussian sigma for SSIM | 1.5 | 0.5–3.0 |
| `ssim_region` | SSIM region: "active" or "full" | "active" | — |

---