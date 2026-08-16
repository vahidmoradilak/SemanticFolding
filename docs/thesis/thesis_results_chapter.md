# Results Chapter Summary

## Overview of Benchmark Results

This chapter summarizes evaluation results across multiple datasets and benchmark configurations. The Semantic Folding pipeline was evaluated on several standard QA/retrieval benchmarks, with particular focus on the impact of parameter configurations and the introduction of image similarity (SSIM) ranking.

### 4.1 Quran Benchmark (30 QA pairs, 6,236 ayahs)

**Pure Semantic Folding (baseline)**:
- MRR: 0.3344
- AP: 0.1203
- P@5: 0.1733
- R@5: 0.1425
- NDCG@10: 0.1219

**SF+SPLADE RRF Fusion**:
- MRR: **0.3579** (+7% over pure SF)
- AP: **0.2181** (+81% over pure SF)
- P@5: **0.2133** (+23% over pure SF)
- R@5: **0.1719** (+20% over pure SF)
- NDCG@10: **0.1841** (+51% over pure SF)

**BM25 Baseline**:
- MRR: 0.1550
- AP: 0.0723
- P@5: 0.0667
- R@5: 0.0472
- NDCG@10: 0.0620

**Key Observations**:
- SPLADE fusion improves all metrics (2.31× MRR, 3.02× AP over BM25)
- +7% MRR, +81% AP over Pure SF
- **11/30 queries succeed** (MRR>0), **19/30 fail** (MRR=0.0000)
- **Common failure patterns**:
  - Thematic queries (justice, mercy, patience, punishment): simplified terms too broad
  - Plural/singular mismatch (angels→angel, believers→believer): no stemming in query term simplification
  - Full query fallback (Q028–Q030): no key vocab terms survive filtering
- **Best config**: `grid_size=64, spreading_steps=1, top_percent=0.10, weighting=idf`

**Query Term Simplification Impact**:
- Only rare proper nouns (joseph, solomon, cave) benefit from single-term simplification
- Thematic queries use 2–3 key terms
- Full query fallback (Q028–Q030) fails because no key vocab terms survive filtering

### 4.2 Multi-Dataset Benchmark Results

**Belebele (100 queries)**:
| Method | MRR | AP |
|--------|-----|-----|
| Pure SF | 0.92 | 0.92 |
| RRF | 0.94 | 0.94 |
| **Linear α=0.3** | **0.98** | **0.98** |
| Linear α=0.5 | 0.94 | 0.94 |
| Linear α=0.7 | 0.92 | 0.92 |
| BM25 | 0.995 | 0.995 |

**NarrativeQA (50 queries)**:
| Method | MRR | AP |
|--------|-----|-----|
| Pure SF | 0.91 | 0.015 |
| RRF | 0.95 | 0.015 |
| **Linear α=0.3** | **0.96** | 0.0157 |
| Linear α=0.5 | 0.86 | 0.0151 |
| Linear α=0.7 | 0.86 | 0.0150 |

**PubMedQA (311 queries)**:
| Method | MRR | AP |
|--------|-----|-----|
| Pure SF | 0.891 | 0.537 |
| RRF | 0.939 | 0.654 |
| **Linear α=0.3** | **0.954** | **0.740** |
| Linear α=0.5 | 0.939 | 0.640 |
| Linear α=0.7 | 0.939 | 0.640 |

**PopQA (1,000 queries)**:
| Method | MRR | AP |
|--------|-----|-----|
| **Pure SF** | **0.84** | **0.43** |
| Linear α=0.3 | 0.749 | 0.381 |
| Linear α=0.5 | 0.817 | 0.416 |
| Linear α=0.7 | 0.826 | 0.420 |
| RRF | ~0.50 | — |

**Key Takeaway**:
- **Linear α=0.3** best on 3/4 datasets (Belebele, NarrativeQA, PubMedQA)
- PopQA: SPLADE hurts regardless of method — stay pure SF
- Full report available at: `outputs/SPLADE_FUSION_RESULTS.md`

### 4.3 Custom Arabic-English Benchmark (New Evaluation)

**Image-Search vs. Cosine Comparison**:
- **Pure SF (cosine)**: MRR=0.8166, AP=0.8166
- **Pure SF (image/SSIM)**: MRR=0.6747, AP=0.6747
- **Difference**: Image search yields lower but discriminative scores
- **Config used**: `grid_size=64, ssim_sigma=1.5, ssim_region=active`
- **Performance note**: SSIM ranking is slower than cosine (per-query SSIM computation over all docs) but provides structural similarity awareness

**Without image-search (regression)**:
- Pure SF: MRR=0.8166, AP=0.8166 (unchanged behavior confirmed)

### 4.4 Parameter Sensitivity Summary

**Grid Size**:
- 64×64: MRR 1.000 (on small 20-doc eval), AP 0.869
- 128×128: MRR 0.900, AP 0.836
- **Conclusion**: 64×64 optimal for small corpora

**Spreading Steps**:
- `spreading_steps=1`: Best overall performance
- `spreading_steps>1`: Diminishing returns, potential degradation

**Top Percent**:
- `top_percent=0.05`: Better discrimination on Quran corpus
- `top_percent=0.10`: More candidates, may dilute signal

**Weighting**:
- `weighting=idf`: Best overall performance
- `weighting=uniform`: Drops C17 ranking, loses C00

**Normalization**:
- L2 (query), √nnz (document): Recommended default
- Other modes (l1, binary, none): Available but less effective

**Query Weighting**:
- IDF: Best; uniform weighting loses C17 ranking

**Geometric Scoring** (3×3 spatial adjacency kernel):
- Rewards nearby (not just exact) cell overlap
- Modest improvements on some datasets, neutral on others

**SSIM Parameters**:
- `ssim_sigma=1.5`: Default, good balance of spatial detail vs. noise
- `ssim_region=active`: Recommended for sparse grids (masks to active cells)
- `ssim_region=full`: Broader comparison, less sparse-sensitive

### 4.5 Error Analysis & Failure Patterns

**Quran Benchmark (19/30 queries fail)**:
- Thematic queries (justice, mercy, patience, punishment): simplified terms too broad
- Plural/singular mismatch: no stemming in `_extract_key_query_terms`
- Full query fallback (Q028–Q030): no key vocab terms survive filtering

**Cross-Dataset Patterns**:
- PopQA: Pure SF always best; SPLADE degrades results
- Belebele/NarrativeQA/PubMedQA: Linear α=0.3 best for hybrid approaches
- Arabic corpora: Image search provides alternative but lower MRR than cosine

### 4.6 Comparison with BM25 Baseline

| Dataset | SF MRR | BM25 MRR | Ratio (SF/BM25) |
|---------|--------|----------|-----------------|
| Quran | 0.3344 | 0.1550 | 2.16× |
| Belebele | 0.92 | 0.995 | 0.92× |
| NarrativeQA | 0.91 | (not separately reported) | — |
| PubMedQA | 0.891 | (not separately reported) | — |
| PopQA | 0.84 | (not separately reported) | — |

**Interpretation**: Semantic folding outperforms BM25 on Quran (2.16× MRR) but underperforms on high-resource benchmarks where BM25's lexical matching is competitive.

### 4.7 Visualizations & Additional Outputs
- `outputs/SPLADE_FUSION_RESULTS.md`: Full fusion results report
- `outputs/run_<timestamp>/`: Pipeline outputs per run
- `outputs/query_metrics/qa_evaluation_report.md`: Per-query evaluation metrics
- Benchmark analysis tools: `semantic_folding/dataset_benchmark/musique/benchmark_analyzer.py`

---

## 4.8 Thesis Contribution Summary

The Semantic Folding pipeline demonstrates:
1. **Effectiveness**: Competitive MRR/AP across diverse datasets (Quran: 0.3344 → 0.3579 with SPLADE; Belebele: 0.92 → 0.98 with linear fusion)
2. **Parameter Sensitivity**: Grid size, weighting, and spreading parameters significantly impact results
3. **Fusion Benefits**: SPLADE RRF and linear fusion consistently improve results on 3/4 multi-dataset benchmarks
4. **Image-Similarity Alternative**: SSIM-based ranking provides structural similarity awareness at computational cost
5. **Failure Pattern Analysis**: Thematic query broadening and plural/singular mismatches are primary failure modes
6. **Benchmark Diversity**: Results vary significantly by dataset, emphasizing the need for dataset-specific parameter tuning

---

## 4.9 Suggested Visualizations for Thesis

1. MRR/AP across datasets by method (bar chart)
2. Parameter sweep results (grid size, spreading, top_percent)
3. Quran query success/failure distribution (11 success / 19 failure)
4. Linear α vs. performance across datasets
5. PopQA: Pure SF vs. SPLADE comparison
6. Custom AR-EN: Cosine vs. SSIM ranking comparison
7. BM25 vs. SF comparison across datasets
8. Failure pattern distribution (thematic vs. lexical mismatches)