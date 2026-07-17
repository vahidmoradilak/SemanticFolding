# PubMedQA Result

**Date:** 2026-07-17
**Best Result:** MRR=0.9884, AP=0.9426

---

## Configuration

```yaml
splade: True
hybrid_alpha: 0.3
fusion_method: linear
top_k: 100
grid_size: 64
```

---

## Results Comparison

| top_k | MRR | AP | Failed |
|-------|-----|-----|--------|
| 20 | 0.954 | 0.740 | 6/172 |
| 50 | 0.988 | 0.924 | 2/172 |
| 100 | 0.988 | 0.943 | 2/172 |

---

## Remaining Failing Queries

| Query ID | Query | Issue |
|----------|-------|-------|
| 0005 | "Syncope during bathing in infants..." | Gold doc not in SF+SPLADE top results |
| 0118 | "The Main Gate Syndrome..." | Gold doc not in SF+SPLADE top results |

---

## Analysis

The 2 failing queries have a fundamental issue: the gold documents are not semantically similar enough to the queries for SF+SPLADE to find them. This is a limitation of the semantic matching approach, not a candidate size issue.
