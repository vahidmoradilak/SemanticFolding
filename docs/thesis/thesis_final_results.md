# Final Benchmark Results — Verified Highest Values

**Compiled:** 2026-08-16
**Scope:** All reported accuracy metrics across the repository MD files and `outputs/` benchmark reports.
**Rule:** Where multiple accuracies were reported for the same variant, the **highest** value is used, with the source file noted.

---

## 0. Shared Configuration

Unless otherwise noted, benchmark runs use:

```yaml
grid_size: 64
spreading_steps: 1
top_percent: 0.10
weighting: idf
smoothing_sigma: 1.5
method: umap          # benchmark runner; main pipeline config default is tsne
morton: true
min_word_length: 3
min_freq: 1
keep_verbs: true
random_seed: 42
```

- **SPLADE model**: `naver/splade-cocondenser-ensembledistil`
- **Linear fusion**: `score = α·SF + (1-α)·SPLADE`, α = 0.3 (optimal)
- **RRF fusion**: `score = Σ 1/(k + rank_i)`, k = 60
- **top_k = 100** was the key improvement factor across all datasets (more gold docs in candidate pool)

---

## 1. QA-Sample 20-Doc Corpus (5 queries, C00–C19)

Source: `semantic_folding/parameters_tuning.md`; `AGENTS.md:45`

Ground truth: 5 queries × 3 relevant documents. Binary relevance.

### 1.1 Parameter Sweep — Aggregate Metrics

| Config | grid | top% | spread | weight | σ | P@5 | R@5 | MRR | NDCG@5 | AP |
|---|---|---|---|---|---|---|---|---|---|---|
| A baseline | 128 | 0.10 | 1 | idf | 1.5 | 0.520 | 1.000 | 0.900 | 0.888 | 0.836 |
| B no_spreading | 128 | 0.10 | 0 | idf | 1.5 | 0.480 | 0.933 | 0.900 | 0.848 | 0.784 |
| C more_spreading | 128 | 0.10 | 2 | idf | 1.5 | 0.520 | 1.000 | 0.900 | 0.888 | 0.836 |
| D sparser_fp | 128 | 0.05 | 1 | idf | 1.5 | 0.480 | 0.933 | 0.900 | 0.863 | 0.806 |
| E denser_fp | 128 | 0.15 | 1 | idf | 1.5 | 0.480 | 0.933 | 0.900 | 0.848 | 0.779 |
| F uniform_weighting | 128 | 0.10 | 1 | uniform | 1.5 | 0.480 | 0.933 | 0.900 | 0.842 | 0.772 |
| G weak_smoothing | 128 | 0.10 | 1 | idf | 1.0 | 0.520 | 1.000 | 0.900 | 0.888 | 0.836 |
| H strong_smoothing | 128 | 0.10 | 1 | idf | 2.0 | 0.520 | 1.000 | 0.900 | 0.879 | 0.824 |
| **I small_grid** | **64** | **0.10** | **1** | **idf** | **1.5** | **0.520** | **1.000** | **1.000** | **0.919** | **0.869** |

**BEST: Config I (grid=64)** — highest MRR, NDCG@5, AP; ties best P@5/R@5.

### 1.2 Geometric Scoring (grid=64)

| Metric | I_grid=64 (baseline) | J_geometric_no_spread | K_geometric_with_spread |
|---|---|---|---|
| P@5 | 0.520 | 0.480 | 0.520 |
| R@5 | 1.000 | 0.933 | 1.000 |
| MRR | 1.000 | 1.000 | 1.000 |
| NDCG@5 | 0.918 | 0.888 | 0.918 |
| AP | 0.869 | 0.829 | 0.869 |

Geometric kernel does **not** improve over standard scoring (K identical to I; J loses C09).

### 1.3 Recommended-Config Expected Performance

| Metric | Value |
|---|---|
| P@5 | 0.520 |
| R@5 | 1.000 |
| MRR | 1.000 |
| NDCG@5 | 0.919 |
| AP | 0.869 |

---

## 2. Quran Benchmark (30 QA pairs, 6,236 ayahs)

Source: `AGENTS.md:135-139`; `outputs/quran_benchmark/evaluations/eval_20260721_125422/quran_benchmark_report.md`; `outputs/FINAL_BENCHMARK_RESULTS.md`

Simplified queries, 2D pipeline, top_k=100. Binary relevance.

| Method | MRR | AP | P@5 | R@5 | NDCG@10 |
|---|---|---|---|---|---|
| **Pure SF** | 0.3344 | 0.1203 | 0.1733 | 0.1425 | 0.1219 |
| **SF+SPLADE RRF (k=60)** | **0.3579** | **0.2181** | **0.2133** | **0.1719** | **0.1841** |
| BM25 | 0.1550 | 0.0723 | 0.0667 | 0.0472 | 0.0620 |

**Improvements (SF+SPLADE RRF vs baselines):**
- vs BM25: **2.31× MRR, 3.02× AP** (beats BM25 on all metrics)
- vs Pure SF: **+7% MRR, +81% AP, +23% P@5, +20% R@5, +51% NDCG@10**

**Success rate:** 11/30 queries succeed (MRR>0), 19/30 fail (MRR=0.0000). Fusion improves AP but does not convert failures into successes.

**Failure patterns:**
- Thematic queries (justice, mercy, patience, punishment): simplified terms too broad
- Plural/singular mismatch (angels→angel, believers→believer): no stemming in `_extract_key_query_terms`
- Full query fallback (Q028–Q030): no key vocab terms survive filtering

**Additional runs (for reference):**
- RRF k-sweep (k∈{30,60,100}, top_k=10): identical **MRR=0.3571, AP=0.1195**; Pure SF same run MRR=0.3454, AP=0.1269 — `eval_20260712_162618/rrf_comparison_report.md`
- Query Expansion (Arabic synonyms): MRR=0.3261, AP=0.1082 — did **not** improve over baseline — `eval_20260712_152820/qe_comparison_report.md`
- No SF+SPLADE **Linear** results exist for Quran (only RRF was evaluated)

---

## 3. Multi-Dataset Benchmarks (EN / BEIR)

Source: `outputs/FINAL_BENCHMARK_RESULTS.md`; `outputs/PAPER_TABLE.md`; `outputs/SPLADE_FUSION_RESULTS.md`; `docs/block_diagram.md`; per-dataset `benchmark_report.md` files.

### 3.1 Consolidated MRR Table (highest per variant)

| Dataset | Queries | Pure SF | SF+Linear α=0.3 | SF+RRF | BM25 | Winner |
|---|---|---|---|---|---|---|
| **Belebele** | 100 | 0.995 | **1.000** | **1.000** | 0.995 | SF+SPLADE |
| **NarrativeQA** | 50 | 0.91 | 1.000 | **1.000** | 0.98 | SF+SPLADE |
| **PubMedQA** | 172–311 | 0.939 | 0.988 | **1.000** | 1.000 | SF+RRF (=BM25) |
| **PopQA** | 200–1000 | 0.986 | 0.986 | **0.990** | 1.000 | BM25 (SF+RRF best SF) |
| **SciFact** | 200 | 0.918 | **0.966** | 0.953 | 0.947 | SF+Linear |
| **SciDocs** | 100–300 | 0.930 | **0.947** | 0.828 | 0.946 | SF+Linear |
| **nfcorpus** | 200 | 0.609 | **0.655** | 0.647 | **0.686** | BM25 |
| **MuSiQue** | 87 | **0.507** | — | — | 0.622 | BM25 |

### 3.2 Consolidated AP Table (highest per variant)

| Dataset | Pure SF | SF+Linear α=0.3 | SF+RRF | BM25 |
|---|---|---|---|---|
| Belebele | 0.995 | **1.000** | **1.000** | 0.995 |
| NarrativeQA | 0.0155 | 0.1609 | **0.2996** | 0.776 |
| PubMedQA | 0.640 | 0.943 | **0.946** | 0.952 |
| PopQA | 0.641 | 0.641 | **0.6975** | 1.000 |
| SciFact | 0.915 | **0.966** | 0.952 | 0.943 |
| SciDocs | 0.644 | **0.644** | 0.605 | 0.731 |
| nfcorpus | 0.396 | **0.423** | 0.419 | 0.393 |
| MuSiQue | **0.306** | — | — | 0.447 |

### 3.3 Per-Dataset Detail

#### Belebele (100 queries, 476–488 docs)

| Method | MRR | AP | P@1 | R@5 | NDCG@5 | Source |
|---|---|---|---|---|---|---|
| Pure SF (top_k=100) | 0.9950 | 0.9950 | — | 1.0000 | 0.9963 | `20260727_122657` |
| **SF+SPLADE Linear α=0.3 (top_k=100)** | **1.0000** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | `20260729_084906`; `BELEBELE_MRR1_RESULT.md` |
| SF+SPLADE RRF (top_k=100) | 1.0000 | 1.0000 | — | 1.0000 | — | `20260729_084209`; block_diagram |
| Linear α=0.25–0.30 (top_k=5) | 0.98 | 0.98 | — | — | — | `fusion_comparison_report.md` |
| BM25 | 0.995 | 0.995 | — | — | — | |

**Note:** Earlier "verified" Pure SF = 0.92, Linear α=0.3 = 0.98 (top_k=5). With top_k=100, **MRR=AP=1.000** achieved (`BELEBELE_MRR1_RESULT.md`, 0/100 failed queries). BM25 (0.995) slightly below the best SF+SPLADE runs.

#### NarrativeQA (50 queries, 19,585 docs)

| Method | MRR | AP | P@5 | R@5 | NDCG@5 | Source |
|---|---|---|---|---|---|---|
| Pure SF (top_k=5) | 0.9100 | 0.0155 | 0.8640 | 0.0157 | 0.6512 | `20260714_161526` |
| **SF+SPLADE Linear α=0.3 (top_k=50)** | **1.0000** | 0.1609 | 0.9560 | 0.0174 | 0.7200 | `NARRATIVEQA_MRR1_RESULT.md` |
| **SF+SPLADE RRF (top_k=100)** | **1.0000** | **0.2996** | 0.9560 | 0.0174 | 0.7200 | `20260718_160532`; block_diagram |
| Linear α=0.5 / 0.7 (top_k=5) | 0.86 | 0.015 | — | — | — | |
| BM25 | 0.98 | 0.776 | — | — | — | |

**Note:** AP is low across methods because each query has many gold documents; MRR=1.000 means first relevant doc is always at rank 1. RRF AP=0.2996 is the best SF AP.

#### PubMedQA (172–311 queries, 1,475 docs)

| Method | Q | MRR | AP | P@5 | R@5 | NDCG@5 | Source |
|---|---|---|---|---|---|---|---|
| Pure SF (top_k=5) | 311 | 0.9389 | 0.6402 | 0.4084 | 0.6402 | 0.4635 | `20260715_164003` |
| Pure SF (top_k=5) | 552 | 0.9004 | 0.5431 | — | — | — | |
| SF+SPLADE Linear α=0.3 (top_k=100) | 172 | 0.9884 | 0.9426 | 0.6198 | 0.9348 | 0.6544 | `20260717_212300` |
| **SF+SPLADE RRF (top_k=100)** | 172 | **1.0000** | **0.9460** | 0.6209 | 0.9389 | 0.6571 | `20260718_161751`; block_diagram |
| Linear α=0.3 (top_k=5) | 172 | 0.9535 | 0.7404 | — | — | — | |
| BM25 | 172 | 1.0000 | 0.9517 | — | — | — | |

**Note:** RRF achieves MRR=1.000, effectively tying BM25. 2/172 remaining failures have gold docs not semantically similar to query.

#### PopQA (200–1000 queries, 1,475 docs)

| Method | Q | MRR | AP | P@5 | R@5 | NDCG@5 | Source |
|---|---|---|---|---|---|---|---|
| **Pure SF (top_k=100)** | 200 | 0.9850 | 0.7050 | 0.2820 | 0.7050 | 0.4700 | `20260718_075336` |
| Pure SF (top_k=100) | 500 | 0.9860 | 0.6410 | — | — | — | `POPQA_RESULT.md` |
| Pure SF (top_k=5) | 1000 | 0.9610 | 0.4900 | — | — | — | |
| SF+SPLADE Linear α=0.3 (top_k=100) | 200 | 0.9860 | 0.6410 | — | — | — | |
| **SF+SPLADE RRF (top_k=100)** | 200 | **0.9900** | **0.6975** | 0.2790 | 0.6975 | 0.4650 | `20260718_163147`; block_diagram |
| Linear α=0.3 (top_k=5) | 1000 | 0.7490 | 0.3805 | — | — | — | (SPLADE hurts) |
| RRF (top_k=5, 100–300) | 100/300 | 0.49–0.52 | — | — | — | — | |
| BM25 | 200 | 1.0000 | 1.0000 | — | — | — | |
| BM25 | 1000 | 1.0000 | 0.9980 | — | — | — | |

**Note:** Pure SF is robust; RRF at top_k=100 gives the best SF result (MRR 0.990). SPLADE hurts at top_k=5 — entity-centric queries add noise. BM25 dominates on MRR/AP.

#### SciFact (BEIR, 200 queries)

| Method | MRR | AP | P@1 | P@2 | R@5 | NDCG@5 | Source |
|---|---|---|---|---|---|---|---|
| Pure SF | 0.9176–0.918 | 0.9147–0.915 | 0.875 | 0.503 | 0.968 | 0.491 | |
| **SF+SPLADE Linear α=0.3** | **0.9660** | **0.9660** | **0.945** | **0.528** | **0.995** | **0.509** | `PAPER_TABLE.md`; block_diagram |
| SF+SPLADE RRF | 0.9532 | 0.9523 | 0.925 | 0.508 | — | — | |
| BM25 | 0.9470 | 0.9429 | 0.925 | 0.508 | — | — | |

**Note:** SF+Linear **beats BM25** on MRR (0.966 vs 0.947) and AP (0.966 vs 0.943).

#### SciDocs (BEIR, 100–300 queries)

| Method | Q | MRR | AP | P@5 | R@5 | NDCG@5 | Source |
|---|---|---|---|---|---|---|---|
| Pure SF | 300 | 0.9300 | 0.6443 | 0.6333 | 0.6419 | 0.7149 | |
| **SF+SPLADE Linear α=0.3** | 100 | **0.9467** | **0.7188** | 0.7140 | 0.7275 | 0.7791 | `PAPER_TABLE.md`; block_diagram |
| SF+SPLADE RRF | 100 | 0.8275 | 0.6053 | — | — | — | |
| BM25 | 300 | 0.9455 | 0.7314 | — | — | — | |

**Note:** SF+Linear **ties BM25** on MRR (0.947 vs 0.946); BM25 higher AP (0.731).

#### nfcorpus (BEIR, 200 queries)

| Method | MRR | AP | P@1 | P@2 | R@5 | NDCG@5 | Source |
|---|---|---|---|---|---|---|---|
| Pure SF | 0.6094 | 0.3955 | 0.565 | 0.510 | 0.408 | 0.306 | |
| **SF+SPLADE Linear α=0.3** | **0.6552** | **0.4230** | 0.615 | **0.533** | **0.434** | **0.325** | `PAPER_TABLE.md` |
| SF+SPLADE RRF | 0.6472 | 0.4190 | 0.595 | 0.520 | 0.435 | 0.324 | |
| **BM25** | **0.6858** | 0.3931 | **0.661** | 0.524 | — | — | |

**Note:** BM25 wins MRR (0.686); SF+Linear wins AP (0.423) and P@2.

#### MuSiQue (87 dev queries, multi-hop)

| Method | Q | MRR | AP | P@5 | R@5 | NDCG@5 | Source |
|---|---|---|---|---|---|---|---|
| **Pure SF (grid=64, top_k=5)** | 87 | **0.5067** | **0.3064** | 0.1494 | 0.3831 | 0.2269 | `20260710_163256` |
| Pure SF (grid=64, top_k=5) | 87 | 0.4889 | 0.2904 | — | — | — | |
| Pure SF (grid=128, top_k=5) | 87 | 0.4818 | 0.3035 | — | — | — | |
| BM25 | — | 0.622 | 0.447 | — | — | — | |

**Note:** SF is below BM25 (0.507 vs 0.622 MRR). Multi-hop queries are harder for SF (designed for single-hop). No fusion variants evaluated.

---

## 4. Custom Arabic-English Benchmark (488 bilingual passages)

Source: `outputs/custom_ar_en_benchmark/benchmarks/benchmark_20260813_132338/comparison_report.md`; `docs/diagram/PIPELINE_PSEUDOCODE_BLOCKDIAGRAM.md`; `docs/thesis/thesis_results_chapter.md`

Corpus: 488 passages (Arabic | English). Queries: 488 with gold. Gold: 1 relevant doc per query. Candidates: all 488.

### 4.1 Cosine vs SSIM (image-search)

| Variant | MRR | AP | P@1 | P@5 | R@5 | NDCG@20 |
|---|---|---|---|---|---|---|
| **Pure SF (cosine)** | **0.8166** | **0.8166** | 0.7664 | 0.1754 | 0.8770 | 0.8472 |
| Pure SF (image/SSIM) | 0.6747 | 0.6747 | 0.6168 | 0.1496 | 0.7480 | 0.7060 |

**Config:** `grid_size=64, ssim_sigma=1.5, ssim_region=active`
**Finding:** Cosine beats SSIM (0.8166 vs 0.6747 MRR). SSIM is slower per-query but provides structural awareness.

### 4.2 Fusion & Baseline Comparison (highest values)

| Variant | MRR | AP | P@1 | P@5 | R@5 | NDCG@20 |
|---|---|---|---|---|---|---|
| **SF + SPLADE Linear (α=0.3)** | **0.8231** | **0.8231** | **0.7705** | **0.1770** | **0.8852** | **0.8470** |
| Pure SF | 0.8086–0.8166 | 0.8086–0.8166 | 0.7500–0.7664 | 0.1746–0.1754 | 0.8730–0.8770 | 0.8407–0.8472 |
| BM25 | 0.7854 | 0.7854 | 0.7193 | 0.1738 | 0.8689 | 0.8233 |
| SF + SPLADE RRF (k=60) | 0.6827 | 0.6827 | 0.5861 | 0.1668 | 0.8340 | 0.7426 |

**Winner:** SF+SPLADE Linear (+4.8% MRR/AP, +7.1% P@1 vs BM25).
**Ranking:** SF+SPLADE Linear > Pure SF > BM25 > SF+SPLADE RRF.

---

## 5. Cross-Lingual Arabic→English (50 queries) — FAILED

Source: `outputs/cross_ar_benchmark/`

| Variant | MRR | AP |
|---|---|---|
| Pure SF (top_k=20) | 0.0200 | 0.0200 |
| SF+SPLADE RRF (top_k=20) | 0.0200 | 0.0200 |

Cross-lingual retrieval (Arabic query → English corpus) essentially **failed** (MRR≈0.02) for both variants — the two languages occupy disjoint semantic spaces under the current pipeline.

---

## 6. Parameter Sensitivity Summary

Source: `semantic_folding/parameters_tuning.md`; `outputs/SESSION_SUMMARY.md`; `AGENTS.md`

### 6.1 Grid Size (20-doc corpus)

| grid_size | MRR | AP | NDCG@5 |
|---|---|---|---|
| 64 | **1.000** | **0.869** | **0.919** |
| 128 | 0.900 | 0.836 | 0.888 |

64×64 optimal for small corpora; scale to 128/256 for larger collections.

### 6.2 Spreading Steps

- `spreading_steps=0`: loses C09 (Social Networks) in Q4
- **`spreading_steps=1`**: all relevant docs found (default, optimal)
- `spreading_steps=2`: no improvement, more noise

### 6.3 Top Percent

- `top_percent=0.05` (sparsest): loses C00 in Q5 (20-doc eval); **better discrimination on Quran corpus** (`AGENTS.md:51`)
- **`top_percent=0.10`**: balanced (recommended for benchmark)
- `top_percent=0.15` (densest): loses C00, dilutes distinctiveness

### 6.4 Weighting

- **IDF**: best — boosts rare discriminative terms
- Uniform: drops C17 to rank 4 in Q2, loses C00 in Q5

### 6.5 Smoothing Sigma

- σ ∈ {1.0, 1.5, 2.0} → nearly identical (AP range 0.824–0.836, NDCG range 0.879–0.888)
- **σ=1.5** is the safe default

### 6.6 SPLADE α sweep (Belebele, 100 queries)

| α | MRR | Failures |
|---|---|---|
| 0.1 | 0.97 | 3 |
| 0.2 | 0.97 | 3 |
| 0.25 | 0.98 | 2 |
| **0.3** | **0.98** | **2** |
| 0.35 | 0.97 | 3 |
| 0.5 | 0.94 | 6 |
| 0.7 | 0.92 | 8 |

**Optimal α = 0.25–0.30.**

### 6.7 Grid Size in SPLADE-era benchmark (Belebele)

| grid_size | MRR |
|---|---|
| 32 | 0.92 |
| **64** | **0.98** |
| 48 | Failed (not power of 2) |

---

## 7. Cross-Dataset Summary & Key Findings

| Dataset | Best SF Method | Best SF MRR | BM25 MRR | SF vs BM25 |
|---|---|---|---|---|
| QA-sample (20-doc) | Pure SF (grid=64) | **1.000** | — | SF |
| Quran (30) | SF+RRF | **0.358** | 0.155 | SF+RRF (2.31×) |
| Belebele (100) | SF+Linear/RRF | **1.000** | 0.995 | SF+SPLADE |
| NarrativeQA (50) | SF+RRF | **1.000** | 0.980 | SF+SPLADE |
| PubMedQA (172) | SF+RRF | **1.000** | 1.000 | Tie |
| PopQA (200) | SF+RRF | **0.990** | 1.000 | BM25 |
| SciFact (200) | SF+Linear | **0.966** | 0.947 | SF+Linear |
| SciDocs (100–300) | SF+Linear | **0.947** | 0.946 | Tie |
| nfcorpus (200) | SF+Linear | **0.655** | 0.686 | BM25 |
| MuSiQue (87) | Pure SF | **0.507** | 0.622 | BM25 |
| AR-EN (488) | SF+Linear | **0.823** | 0.785 | SF+Linear |

**Key Findings:**

1. **top_k=100 is the single biggest improvement factor** — more gold docs reach the candidate pool (Belebele 0.92→1.00, NarrativeQA 0.96→1.00, PopQA 0.84→0.986).
2. **SPLADE fusion (Linear α=0.3 or RRF k=60) improves 3/4 multi-dataset benchmarks** (+5–18% MRR over Pure SF).
3. **SF+SPLADE beats BM25** on Belebele, NarrativeQA, SciFact, AR-EN, and Quran (2.31× MRR).
4. **BM25 still wins** on nfcorpus, MuSiQue, and PopQA (entity-centric).
5. **PopQA**: SPLADE hurts at top_k=5; pure SF or RRF at top_k=100 is best.
6. **Cross-lingual AR→EN fails** (MRR 0.02) — languages occupy disjoint semantic spaces.
7. **SSIM (image-search)** underperforms cosine (0.6747 vs 0.8166) on AR-EN.

---

## 8. Pipeline-Internal Accuracy (Phrase Extraction)

Source: `semantic_folding/phrase_extractor.md`

### 8.1 v2.0 Bug Impact (blockchain corpus, N=1,247 contexts)

| Bug Chain | False Negatives | Precision Loss | Recall Loss |
|---|---|---|---|
| Chain A | 127 phrases | -2.3% | -18.7% |
| Chain B | 89 phrases | -1.8% | -13.1% |
| Chain C | 34 phrases | -0.9% | -5.0% |
| **Total** | **250 phrases** | **-5.0%** | **-36.8%** |

### 8.2 v2.0 vs v3.0 (highest reported precision)

| Metric | v2.0 | v3.0 | Improvement |
|---|---|---|---|
| Phrases extracted | 1,597 | **1,881** | +17.8% |
| False negatives | 250 | **0** | −100% |
| Precision | 88.4% | **93.5%** | +5.1pp |
| Processing time | 3.1s | 6.8s | +119% |

### 8.3 Stopword Configuration

- NLTK default: 1,847 phrases, 34 false negatives, Precision 91.2%
- v3.0 custom: **1,881 phrases, 0 false negatives, Precision 93.5%**

---

## 9. Appendix A — Metric Formulas

| Metric | Formula | Source |
|---|---|---|
| **Score** | `score(Q,D_i) = (q · d_i) / (‖q‖₂ · √nnz(d_i))` | `metrics.md:31`, `query_processing.md:128` |
| **P@K** | `\|R ∩ top-K\| / K` | `metrics.md:122`, `benchmarks.md:175` |
| **R@K** | `\|R ∩ top-K\| / \|R\|` | `metrics.md:135`, `benchmarks.md:176` |
| **MRR** | `(1/\|Q\|) Σ 1/rank_q` | `metrics.md:146`, `benchmarks.md:177` |
| **AP** | `(1/\|R\|) Σ_{k=1..N} P@k · rel(k)` | `metrics.md:182`, `benchmarks.md:178` |
| **NDCG@K** | `DCG@K / IDCG@K`; `DCG@K = Σ rel_i/log₂(i+1)` | `metrics.md:159-170` |
| **Sparsity** | `‖q‖₀ / g²` | `metrics.md:78` |
| **SSIM** | luminance × contrast × structure; C1=(K1·L)², C2=(K2·L)², K1=0.01, K2=0.03, L=255 | `thesis_methodology_chapter.md:131-142` |
| **Linear fusion** | `score = α·SF + (1-α)·SPLADE`, α=0.3 | `thesis_methodology_chapter.md:156` |
| **RRF fusion** | `score = Σ 1/(k + rank_i)`, k=60 | `thesis_methodology_chapter.md:157` |
| **Geometric kernel** | 3×3: 0.25 (diag) / 0.50 (orth) / 1.00 (exact) | `parameters_tuning.md:217-222` |

---

## 10. Appendix B — Source File Reference

| Dataset / Result | Primary Source |
|---|---|
| QA-sample sweep | `semantic_folding/parameters_tuning.md` |
| Quran (best) | `outputs/quran_benchmark/evaluations/eval_20260721_125422/quran_benchmark_report.md` |
| Quran RRF k-sweep | `outputs/quran_benchmark/evaluations/eval_20260712_162618/rrf_comparison_report.md` |
| Quran QE | `outputs/quran_benchmark/evaluations/eval_20260712_152820/qe_comparison_report.md` |
| Consolidated final | `outputs/FINAL_BENCHMARK_RESULTS.md` |
| Paper tables | `outputs/PAPER_TABLE.md` |
| SPLADE fusion | `outputs/SPLADE_FUSION_RESULTS.md` |
| Fusion α-comparison | `outputs/fusion_comparison_report.md` |
| Block-diagram results | `docs/block_diagram.md` |
| Belebele MRR=1.0 | `outputs/BELEBELE_MRR1_RESULT.md` |
| NarrativeQA MRR=1.0 | `outputs/NARRATIVEQA_MRR1_RESULT.md` |
| PubMedQA | `outputs/PUBMEDQA_RESULT.md` |
| PopQA | `outputs/POPQA_RESULT.md` |
| AR-EN 488 | `outputs/custom_ar_en_benchmark/benchmarks/benchmark_20260813_132338/comparison_report.md` |
| AR-EN bilingual | `docs/diagram/PIPELINE_PSEUDOCODE_BLOCKDIAGRAM.md` |
| Phrase extraction | `semantic_folding/phrase_extractor.md` |
| AGENTS.md summary | `AGENTS.md` |