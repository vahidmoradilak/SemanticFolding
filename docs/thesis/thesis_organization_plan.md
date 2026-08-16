# Thesis Organization Plan

## Document Structure

This thesis presents the Semantic Folding (SF) pipeline for information retrieval, exploring how 2D grid-based semantic representations can be used for document ranking and retrieval. The pipeline consists of seven sequential steps, from phrase extraction through query processing and similarity scoring.

### Chapter Breakdown

**Chapter 1: Introduction**
- Background on information retrieval and text mining
- Overview of semantic folding approach
- Research questions and contributions
- Thesis structure overview

**Chapter 2: Related Work**
- Traditional information retrieval (TF-IDF, BM25)
- Vector space models and semantic embeddings (Word2Vec, BERT)
- Graph-based and topological approaches to text representation
- Previous work on fingerprints and sparse distributed representations
- Image similarity and SSIM in retrieval contexts
- Summary of gaps this work addresses

**Chapter 3: Semantic Folding Pipeline (Methodology)**
- Step 1: Phrase extraction (Arabic/English)
- Step 2: Term context construction with TF-IDF
- Step 3: Semantic space construction (Morton ordering, t-SNE/UMAP reduction)
- Step 4: Phrase fingerprints and topological sparsification
- Step 5: Document fingerprints construction
- Step 6: Query processing (query processor Step 7)
- Step 7: Similarity scoring (cosine, SSIM, hybrid methods)
- Experimental setup (grid size, spreading parameters, weighting schemes)
- Evaluation metrics (MRR, AP, P@K, R@K, NDCG@K)

**Chapter 4: Experimental Results (Evaluation)**
- Benchmark results across datasets (Quran, Belebele, PubMedQA, PopQA, NarrativeQA)
- Impact of parameter variations (grid_size, spreading_steps, top_percent, weighting)
- SSIM vs. cosine comparison results
- Error analysis and failure patterns
- SF+SPLADE fusion results
- Comparison with BM25 baseline

**Chapter 5: Conclusions and Future Work**
- Summary of key findings
- Contributions of the thesis
- Limitations and open problems
- Future research directions (multi-modal extensions, improved query simplification, etc.)

---

## Appendix A: Pipeline Parameter Reference
- Grid size: 64×64 (optimal for 20-doc corpus)
- Encoding: Morton Z-order (use_morton: true)
- Smoothing: Gaussian blur, sigma=1.5 (no_smooth: false to enable)
- TF-IDF: Applied in Step 2
- Dimensionality reduction: t-SNE (default), also supports UMAP, PCA
- Spreading: radius=1, decay=0.5 (in query processor)
- top_percent: 0.05 (0.10 dilutes signal)
- Query weighting: IDF (best; uniform drops C17 ranking and loses C00)
- Normalization: L2 for query, √nnz for document fingerprints
- Geometric scoring: Optional 3×3 spatial adjacency kernel

---

## Appendix B: Benchmark Datasets
- QA-sample.md: 5 test queries with 3 relevant documents each (C00–C19)
- Quran benchmark: 30 QA pairs, 6,236 ayahs
- Belebele: 100 queries
- NarrativeQA: 50 queries
- PubMedQA: 311 queries
- PopQA: 1,000 queries
- BM25 baseline comparisons