# PopQA Result

**Date:** 2026-07-18
**Best Result:** MRR=0.9860, AP=0.6410 (500 queries)

---

## Configuration

```yaml
splade: False  # Pure SF - SPLADE hurts PopQA
top_k: 100
grid_size: 64
```

---

## Results Comparison

| Config | MRR | AP | Queries |
|--------|-----|-----|---------|
| Previous (top_k=20) | 0.84 | 0.43 | 1000 |
| **New (top_k=100)** | **0.986** | **0.641** | 500 |

---

## Improvement

- MRR: 0.84 → 0.986 (+17.4%)
- AP: 0.43 → 0.641 (+49.1%)

---

## Note

PopQA uses pure SF (no SPLADE) because SPLADE adds noise for entity-centric queries. The significant improvement comes from increasing top_k from 20 to 100.
