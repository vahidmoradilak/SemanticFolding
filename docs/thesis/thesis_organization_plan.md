# Thesis Organization Plan

## 1. Overview

This thesis presents the **Semantic Folding (SF)** pipeline for information retrieval — a brain-inspired approach that transforms textual documents into **sparse distributed representations (SDRs)** on a discrete 2D grid. The pipeline maps semantically related concepts to adjacent grid cells via dimensionality reduction and Morton (Z-order) encoding, then ranks documents by normalized dot-product similarity between query and document fingerprints.

The pipeline consists of **seven sequential steps**, from phrase extraction through query processing and similarity scoring:

```
Step 1  Phrase Extraction            phrase_extractor.py
Step 2  Term-Context Matrix          term_context.py
Step 3  Semantic Space               semantic_space.py
Step 4  Phrase Fingerprints          phrase_fingerprints.py
Step 5  Document Fingerprints        doc_fingerprints.py
Step 6  Custom-Text Fingerprints     customtext_fingerprints.py
Step 7  Query Processing & Scoring   query_processor.py
```

The work is evaluated against **10 benchmarks** (Quran, Belebele, NarrativeQA, PubMedQA, PopQA, SciFact, SciDocs, nfcorpus, MuSiQue, and a custom bilingual Arabic-English corpus), compared against a **BM25 baseline**, and extended with **SPLADE fusion** (Linear α=0.3 and RRF k=60).

---

## 2. Chapter Breakdown

### Chapter 1: Introduction

- **1.1 Background**: Information retrieval (IR) fundamentals; the lexical gap between TF-IDF/BM25 keyword matching and semantic understanding.
- **1.2 Motivation**: Why 2D spatial representation? Biological inspiration (grid cells in the entorhinal cortex; Kanerva's Hyperdimensional Computing), and the limits of dense embeddings (opacity, cost, hardware requirements).
- **1.3 The Semantic Folding Approach**: Preview of the 7-step pipeline; document fingerprints as sparse grids.
- **1.4 Research Questions**:
  - RQ1: Can a 2D-grid SDR pipeline achieve competitive retrieval accuracy against BM25 and SPLADE-augmented baselines?
  - RQ2: How do grid size, spreading, top_percent, and weighting affect retrieval quality?
  - RQ3: Does SPLADE fusion (Linear/RRF) improve accuracy over pure Semantic Folding?
  - RQ4: Can structural (image-based) similarity — SSIM — compete with cosine similarity on grid fingerprints?
  - RQ5: How does the pipeline generalize to Arabic and bilingual (Arabic–English) corpora?
- **1.5 Contributions**:
  1. A complete, reproducible 7-step unsupervised Semantic Folding pipeline.
  2. A systematic parameter-tuning study (grid 64 vs 128, spreading 0–2, top_percent, weighting, σ).
  3. SPLADE fusion achieving MRR=1.000 on Belebele, NarrativeQA, and PubMedQA.
  4. Image-similarity (SSIM) ranking as an alternative scoring mode.
  5. Cross-lingual (Arabic–English) benchmark and analysis.
- **1.6 Thesis Structure**: Roadmap of Chapters 2–5.
- **1.7 Publications**: Optional list of derived papers.

---

### Chapter 2: Related Work

- **2.1 Classical IR**: TF-IDF (Salton & Buckley, 1988); BM25 (Robertson & Zaragoza, 2009); Okapi ranking; the lexical-gap limitation.
- **2.2 Dense & Neural Retrieval**: Word2Vec (Mikolov et al., 2013); BERT / sentence-transformers (Devlin et al., 2019; Reimers & Gurevych, 2019); bi-encoder retrieval; SPLADE (Formal et al., 2021) sparse neural encoding; DPR (Karpukhin et al., 2020).
- **2.3 Sparse Distributed Representations & Hyperdimensional Computing**: Kanerva (2009) HDC; SDRs in Numenta HTM; Semantic Folding origin (Vermeulen & Bezobrazov, 2017); Fingerprints (Widdows & Cohen, 2015).
- **2.4 Graph & Multi-hop Retrieval**: HippoRAG (Gutiérrez et al., 2024) and HippoRAG 2 (2025) — Personalized PageRank over knowledge graphs; MuSiQue (Trivedi et al., 2022) multi-hop QA.
- **2.5 Image-Similarity in Non-Visual Domains**: SSIM (Wang et al., 2004); structural similarity applied to grid representations — a gap this work fills.
- **2.6 Cross-Lingual & Arabic IR**: Arabic NLP challenges (clitics, function words); bilingual retrieval approaches.
- **2.7 Summary of Gaps**: (a) no prior work applies SSIM to semantic grids; (b) limited unsupervised alternatives to dense retrieval; (c) no systematic comparison of grid-based SDR retrieval against SPLADE/BEIR benchmarks.

---

### Chapter 3: Semantic Folding Pipeline (Methodology)

*Detailed outline: see `docs/thesis/thesis_methodology_chapter.md`*

- **3.1 Overview** — 7-step architecture and design rationale.
- **3.2 Step 1: Phrase Extraction** — Arabic (`lib.py:extract_raw_phrases_ar_fa`) and English (`phrase_extractor.py:extract_raw_phrases_spacy`) POS-pattern bigram/trigram extraction; clitic stripping; function-word filtering.
- **3.3 Step 2: Term-Context Matrix** — TF-IDF weighting: `idf(t) = log(N/n_t)`.
- **3.4 Step 3: Semantic Space Construction** — Dimensionality reduction (t-SNE default in pipeline config; UMAP in benchmark runner; PCA); Morton Z-order encoding; grid_size=64.
- **3.5 Step 4: Phrase Fingerprints** — 2D grid accumulation; Gaussian smoothing (σ=1.5); topological sparsification (`top_percent`, `min_peak_distance`).
- **3.6 Step 5: Document Fingerprints** — document-level aggregation; L2 / √nnz normalization; precomputed norms.
- **3.7 Step 6: Custom-Text Fingerprints** — bilingual (Arabic–English) fingerprint construction.
- **3.8 Step 7: Query Processing & Similarity Scoring** — query fingerprint construction; spreading (radius=1, decay=0.5); scoring formulas:
  - Cosine: `cos(q,d) = (q·d)/(‖q‖·‖d‖)`
  - Score (asymmetric): `score(Q,D_i) = (q·d_i)/(‖q‖₂·√nnz(d_i))`
  - SSIM (structural): luminance × contrast × structure terms
  - Hybrid: `α·cosine + (1−α)·SSIM`
  - SPLADE Linear: `α·SF + (1−α)·SPLADE`, α=0.3
  - RRF: `Σ 1/(k + rank_i)`, k=60
- **3.9 Retrieval Backends** — numpy (default) and lancedb (ANN; incompatible with image_search).
- **3.10 Evaluation Metrics** — P@K, R@K, MRR, AP, NDCG@K (formulas in Appendix C).
- **3.11 Experimental Setup** — hardware (CPU), software (Python 3, numpy/scipy/spacy/plotly/sklearn), datasets, reproducibility (random_seed=42).

---

### Chapter 4: Experimental Results (Evaluation)

*Consolidated verified numbers: see `docs/thesis/thesis_final_results.md`*

- **4.1 QA-Sample Parameter Tuning** (20-doc corpus, 5 queries):
  - Grid sweep: **64×64 → MRR 1.000, AP 0.869, NDCG@5 0.919** vs 128×128 → 0.900 / 0.836 / 0.888.
  - Spreading, top_percent, weighting, smoothing sensitivity; geometric kernel neutrality.
- **4.2 Quran Benchmark** (30 QA pairs, 6,236 ayahs):
  - Pure SF 0.3344 / 0.1203; **SF+SPLADE RRF 0.3579 / 0.2181** (+7% MRR, +81% AP); BM25 0.1550 / 0.0723 (2.31× MRR gain).
- **4.3 Multi-Dataset Benchmarks** (top_k=100):
  - Belebele (100): SF+Linear/RRF **MRR=1.000** (BM25 0.995)
  - NarrativeQA (50): SF+RRF **MRR=1.000, AP=0.2996**
  - PubMedQA (172): SF+RRF **MRR=1.000, AP=0.946** (BM25 1.000)
  - PopQA (200): SF+RRF 0.990 (BM25 1.000 — entity-centric)
  - SciFact (200): SF+Linear **0.966 / 0.966** (beats BM25 0.947)
  - SciDocs: SF+Linear 0.947 / 0.644 (beats BM25 0.946)
  - nfcorpus (200): SF+Linear 0.655 / 0.423 (BM25 0.686 wins)
  - MuSiQue (87): Pure SF 0.507 / 0.306 (BM25 0.622)
- **4.4 Custom Arabic-English Benchmark** (488 bilingual passages):
  - Cosine vs SSIM: **0.8166 vs 0.6747** MRR; SF+SPLADE Linear **0.8231** best.
- **4.5 Cross-Lingual Arabic→English**: failed (MRR≈0.02) — languages occupy disjoint semantic spaces.
- **4.6 Error Analysis**: Quran 11/30 success (thematic query broadening, plural/singular mismatch); PopQA SPLADE noise; cross-lingual failure.
- **4.7 BM25 Comparison**: SF+SPLADE beats BM25 on 6/10 benchmarks, ties 1, loses 3.

---

### Chapter 5: Conclusions and Future Work

- **5.1 Summary of Findings**: SF+SPLADE achieves MRR=1.000 on 3 benchmarks; grid=64 and top_k=100 are decisive; SSIM viable but inferior to cosine; cross-lingual retrieval remains unsolved.
- **5.2 Contributions Restated**: (map back to §1.5).
- **5.3 Limitations**:
  - Corpus-size dependence of grid resolution.
  - t-SNE/UMAP seed sensitivity.
  - Binary relevance (no graded relevance for NDCG).
  - No GPU acceleration.
  - Cross-lingual failure.
- **5.4 Future Work**:
  - Learned / asymmetric geometric kernels; larger grids (256+).
  - Graded-relevance evaluation; multi-resolution fingerprints.
  - Stemming for Arabic query simplification.
  - Fusion with dense (bi-encoder) retrieval; ensemble methods.
  - Cross-lingual shared semantic space (alignment / embedding bridges).

---

## 3. Appendix A: Pipeline Parameter Reference

| Parameter | Step | Default | Range | Notes |
|-----------|------|---------|-------|-------|
| `grid_size` | 3–6 | 64 | 32–128 | 64 optimal for small corpora; must match Steps 3–6 |
| `use_morton` | 3–5 | true | true/false | Morton Z-order vs row-major |
| `spreading_steps` | 7 | 1 | 0–2 | Moore-neighbourhood expansion; 1 optimal |
| `spreading_decay` | 7 | 0.5 | 0.1–1.0 | Per-step decay |
| `top_percent` | 5–6 | 0.05 (pipeline) / 0.10 (benchmark) | 0.03–0.20 | Sparsity after peak detection |
| `smoothing_sigma` | 4–5 | 1.5 | 1.0–2.0 | Gaussian blur width |
| `min_peak_distance` | 5–6 | 2 | — | Minimum distance between hotspots |
| `weighting` | 7 | idf | idf/uniform | IDF essential for discriminative terms |
| `normalization` | 5,7 | l2 (query) / √nnz (doc) | — | Asymmetric normalization |
| `min_word_length` | 1 | 3 (EN), 2 (AR) | 1–5 | Preserves Arabic حروف مقطعه |
| `min_freq` | 1–2 | 1 | — | Keep hapax |
| `keep_verbs` | 1 | true | true/false | Include verbal phrases |
| `method` | 3 | tsne (config) / umap (benchmark) | tsne/umap/pca | Reduction algorithm |
| `geometric` | 7 | false | true/false | 3×3 adjacency kernel (no gain with spreading) |
| `splade` | 7 | false | true/false | Enable SPLADE fusion |
| `fusion_method` | 7 | linear | linear/rrf | Fusion type |
| `hybrid_alpha` | 7 | 0.3 | 0.1–0.7 | α for linear fusion (optimal 0.25–0.30) |
| `rrf_k` | 7 | 60 | 10–100 | RRF constant |
| `image_search` | 7 | false | true/false | SSIM-based ranking (requires numpy backend) |
| `ssim_sigma` | 7 | 1.5 | 0.5–3.0 | SSIM Gaussian window |
| `ssim_region` | 7 | active | active/full | Mask to active cells (recommended) |
| `top_k` | 7 | 5 | 5–100 | Candidate pool size; **100 is the key improvement** |

---

## 4. Appendix B: Benchmark Datasets

| Dataset | Corpus | Queries | Gold per Query | Notes |
|---------|--------|---------|----------------|-------|
| QA-sample | 20 docs (C00–C19) | 5 | 3 docs | Parameter-tuning ground truth |
| Quran | 6,236 ayahs | 30 | variable | Arabic; simplified queries |
| Belebele | 476–488 passages | 100 | 1 doc | Reading comprehension |
| NarrativeQA | 19,585 docs | 50 | many | Multi-gold per query (low AP) |
| PubMedQA | 1,475 docs | 172–552 | 1 doc | Biomedical |
| PopQA | 1,475 docs | 200–1000 | 1 doc | Entity-centric |
| SciFact (BEIR) | — | 200 | 1+ | Claim verification |
| SciDocs (BEIR) | — | 100–300 | 1+ | Document classification |
| nfcorpus (BEIR) | — | 200 | multiple | Medical IR |
| MuSiQue | 20-passage pool/query | 87 | 2–5 | Multi-hop QA |
| AR-EN (custom) | 488 bilingual | 488 | 1 doc | Arabic\|English |
| Cross-AR→EN | — | 50 | 1 | Arabic query → EN corpus (failed) |

---

## 5. Appendix C: Evaluation Metric Formulas

| Metric | Formula |
|--------|---------|
| Precision@K | `P@K = |R ∩ top-K| / K` |
| Recall@K | `R@K = |R ∩ top-K| / |R|` |
| Mean Reciprocal Rank | `MRR = (1/|Q|) Σ 1/rank_q` |
| Average Precision | `AP = (1/|R|) Σ_{k=1..N} P@k · rel(k)` |
| NDCG@K | `DCG@K / IDCG@K`; `DCG@K = Σ rel_i/log₂(i+1)` |
| Score (SF) | `score(Q,D_i) = (q · d_i) / (‖q‖₂ · √nnz(d_i))` |
| SPLADE Linear | `score = α·SF + (1−α)·SPLADE`, α=0.3 |
| RRF | `score = Σ 1/(k + rank_i)`, k=60 |
| SSIM | luminance × contrast × structure; C1=(K1·L)², C2=(K2·L)², K1=0.01, K2=0.03, L=255 |

---

## 6. Suggested Writing Order & Timeline

1. **Chapter 3 (Methodology)** — most self-contained; mirrors the codebase (`thesis_methodology_chapter.md`).
2. **Chapter 4 (Results)** — insert verified tables from `thesis_final_results.md`; add plots (MRR/AP bar charts, α-sweep curves, Quran success/failure distribution).
3. **Chapter 2 (Related Work)** — literature survey; cite §2 sources.
4. **Chapter 1 (Introduction)** — write last, after contributions crystallize.
5. **Chapter 5 (Conclusions)** — draft from §4 takeaways.

---

## 7. Cross-References to Repository Files

| Topic | File |
|-------|------|
| Methodology outline | `docs/thesis/thesis_methodology_chapter.md` |
| Consolidated results (verified highest) | `docs/thesis/thesis_final_results.md` |
| Results chapter | `docs/thesis/thesis_results_chapter.md` |
| Parameter tuning study | `semantic_folding/parameters_tuning.md` |
| Metrics definitions | `semantic_folding/metrics.md` |
| Benchmark methodology | `semantic_folding/benchmarks.md` |
| Final benchmark numbers | `outputs/FINAL_BENCHMARK_RESULTS.md` |
| Paper tables | `outputs/PAPER_TABLE.md` |
| SPLADE fusion analysis | `outputs/SPLADE_FUSION_RESULTS.md` |
| Block diagram (results) | `docs/block_diagram.md` |
| MuSiQue benchmark | `semantic_folding/dataset_benchmark/musique/README.md` |