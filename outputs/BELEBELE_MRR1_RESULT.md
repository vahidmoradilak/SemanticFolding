# Belebele MRR=1.0 Result

**Date:** 2026-07-17
**Achievement:** MRR=1.0000, AP=1.0000

---

## Configuration

```yaml
splade: True
hybrid_alpha: 0.3
fusion_method: linear
splade_model: naver/splade-cocondenser-ensembledistil
top_k: 50  # Key change: increased from 20 to 50
grid_size: 64
spreading_steps: 1
top_percent: 0.10
weighting: idf
```

---

## Results

| Metric | Value |
|--------|-------|
| MRR | 1.0000 |
| AP | 1.0000 |
| P@1 | 1.0000 |
| P@2 | 0.5000 |
| P@5 | 0.2000 |
| R@5 | 1.0000 |
| Failed queries | 0/100 |

---

## Root Cause Analysis

### Why Queries 29 and 90 Were Failing

**Query 29:** "According to the passage, which of the following would be the most beneficial for a runner preparing for the upcoming season?"
- Gold document: `doc_000018`
- With top_k=20: Gold document not in top 20 candidates
- With top_k=50: Gold document included in candidates

**Query 90:** "According to the passage, which of the following is associated with gentler sound?"
- Gold document: `doc_000014`
- With top_k=20: Gold document not in top 20 candidates
- With top_k=50: Gold document included in candidates

### Solution
Increasing `top_k` from 20 to 50 allows more candidate documents to be considered, ensuring that the gold documents are included in the candidate pool for SF+SPLADE ranking.

---

## Benchmark Report

- Run: `run_20260717_204110`
- Benchmark: `benchmark_20260717_204309`
- Report: `outputs/belebele_benchmark/benchmarks/benchmark_20260717_204309/`
