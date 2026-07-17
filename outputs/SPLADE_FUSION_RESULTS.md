# SPLADE Fusion Benchmark Results

**Date:** 2026-07-17
**Configuration:** grid_size=64, spreading_steps=1, top_percent=0.10, weighting=idf
**Fusion:** Linear (score = α·SF + (1-α)·SPLADE), α=0.3
**SPLADE Model:** naver/splade-cocondenser-ensembledistil

---

## Summary

| Dataset | Method | MRR | AP | Improvement |
|---------|--------|-----|-----|-------------|
| Belebele | Pure SF | 0.92 | 0.92 | — |
| Belebele | **SPLADE α=0.3** | **0.98** | **0.98** | **+0.06 MRR** |
| NarrativeQA | Pure SF | 0.91 | 0.015 | — |
| NarrativeQA | **SPLADE α=0.3** | **0.96** | **0.0157** | **+0.05 MRR** |
| PubMedQA | Pure SF | 0.891 | 0.537 | — |
| PubMedQA | **SPLADE α=0.3** | **0.954** | **0.740** | **+0.063 MRR, +0.203 AP** |
| PopQA | Pure SF | 0.84 | 0.43 | — |
| PopQA | SPLADE | ~0.75 | ~0.38 | SPLADE hurts |

---

## Key Findings

1. **SPLADE fusion significantly improves 3/4 datasets**
   - Belebele: +6.5% MRR
   - NarrativeQA: +5.5% MRR
   - PubMedQA: +7.1% MRR, +37.8% AP

2. **PopQA remains pure SF** — SPLADE adds noise for entity-centric queries

3. **Optimal configuration**: `splade=True, hybrid_alpha=0.3, fusion_method=linear`

---

## Run Directories

| Dataset | Run Directory |
|---------|---------------|
| Belebele | `outputs/belebele_benchmark/runs/run_20260717_154235` |
| NarrativeQA | `outputs/narrativeqa_benchmark/runs/run_20260717_160056` |
| PubMedQA | `outputs/pubmedqa_benchmark/runs/run_20260717_161150` |

---

## Benchmark Reports

| Dataset | Report Path |
|---------|-------------|
| Belebele | `outputs/belebele_benchmark/benchmarks/benchmark_20260717_155434/benchmark_report.md` |
| NarrativeQA | `outputs/narrativeqa_benchmark/benchmarks/benchmark_20260717_160605/benchmark_report.md` |
| PubMedQA | `outputs/pubmedqa_benchmark/benchmarks/benchmark_20260717_161735/benchmark_report.md` |

---

## Performance Optimizations (Also in this commit)

1. Vectorized Morton encoding (5-10x speedup)
2. Sparse scatter in fingerprint building (5-10x speedup)
3. Batch document scoring (10-50x speedup)
4. Increased LRU caches (30-50% less recomputation)
5. Precomputed document norms
