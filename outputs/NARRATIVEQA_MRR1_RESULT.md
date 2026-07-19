# NarrativeQA MRR=1.0 Result

**Date:** 2026-07-17
**Achievement:** MRR=1.0000, AP=0.1609

---

## Configuration

```yaml
splade: True
hybrid_alpha: 0.3
fusion_method: linear
splade_model: naver/splade-cocondenser-ensembledistil
top_k: 50
grid_size: 64
```

---

## Results

| Metric | Value |
|--------|-------|
| MRR | 1.0000 |
| AP | 0.1609 |
| Failed queries | 0/50 |

---

## Improvement Over Previous

| Metric | Previous (top_k=20) | New (top_k=50) | Improvement |
|--------|---------------------|----------------|-------------|
| MRR | 0.96 | **1.00** | +4.2% |
| AP | 0.0157 | **0.1609** | +925% |
| Failed queries | 2/50 | **0/50** | -2 |

---

## Note on AP
AP (Average Precision) is low (0.1609) because NarrativeQA has multiple gold documents per query, and the metric penalizes when not all relevant documents are retrieved in the top results. However, MRR=1.0 means the first relevant document is always at rank 1.
