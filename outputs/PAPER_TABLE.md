# جدول مقایسه کامل برای مقاله

## Table 1: Main Results — MRR Comparison

| Dataset | Queries | Pure SF | SF+Linear | SF+RRF | BM25 | بهترین |
|---------|---------|---------|-----------|--------|------|--------|
| Belebele | 100 | 0.920 | **1.000** | **1.000** | 0.995 | SF+SPLADE |
| NarrativeQA | 50 | 0.910 | **1.000** | **1.000** | 0.980 | SF+SPLADE |
| PubMedQA | 172 | 0.891 | 0.988 | **1.000** | 1.000 | SF+RRF |
| PopQA | 200 | 0.840 | 0.986 | 0.990 | 1.000 | BM25 |
| SciFact | 200 | 0.918 | **0.966** | 0.953 | 0.947 | SF+Linear |
| nfcorpus | 200 | 0.609 | 0.655 | 0.647 | **0.686** | BM25 |

## Table 2: Main Results — AP Comparison

| Dataset | Queries | Pure SF | SF+Linear | SF+RRF | BM25 | بهترین |
|---------|---------|---------|-----------|--------|------|--------|
| Belebele | 100 | 0.920 | **1.000** | **1.000** | 0.995 | SF+SPLADE |
| NarrativeQA | 50 | 0.015 | 0.161 | **0.300** | 0.776 | BM25 |
| PubMedQA | 172 | 0.537 | 0.943 | **0.946** | 0.952 | BM25 |
| PopQA | 200 | 0.430 | 0.641 | 0.698 | 1.000 | BM25 |
| SciFact | 200 | 0.915 | **0.966** | 0.952 | 0.943 | SF+Linear |
| nfcorpus | 200 | 0.396 | **0.423** | 0.419 | 0.393 | SF+Linear |

## Table 3: Detailed Metrics — SciFact (200 queries)

| Metric | Pure SF | SF+Linear | SF+RRF | BM25 |
|--------|---------|-----------|--------|------|
| MRR | 0.918 | **0.966** | 0.953 | 0.947 |
| AP | 0.915 | **0.966** | 0.952 | 0.943 |
| P@1 | 0.875 | **0.945** | 0.925 | 0.925 |
| P@2 | 0.503 | **0.528** | 0.508 | 0.508 |
| R@5 | 0.968 | **0.995** | — | — |
| NDCG@5 | 0.491 | **0.509** | — | — |

## Table 4: Detailed Metrics — nfcorpus (200 queries)

| Metric | Pure SF | SF+Linear | SF+RRF | BM25 |
|--------|---------|-----------|--------|------|
| MRR | 0.609 | 0.655 | 0.647 | **0.686** |
| AP | 0.396 | **0.423** | 0.419 | 0.393 |
| P@1 | 0.565 | 0.615 | 0.595 | **0.661** |
| P@2 | 0.510 | **0.533** | 0.520 | 0.524 |
| R@5 | 0.408 | **0.434** | 0.435 | — |
| NDCG@5 | 0.306 | **0.325** | 0.324 | — |

## Table 5: Speed Comparison

| Dataset | SF+SPLADE (s/query) | BM25 (s/query) | نسبت سرعت |
|---------|---------------------|----------------|-----------|
| Belebele | 2.88 | 0.02 | 124.6x |
| SciFact | 1.65 | 0.02 | 83.7x |

## Table 6: Improvement Over Pure SF

| Dataset | MRR بهبود | AP بهبود | بهترین روش |
|---------|----------|----------|-----------|
| Belebele | +8.7% | +8.7% | SF+SPLADE |
| NarrativeQA | +9.9% | +1850% | SF+SPLADE |
| PubMedQA | +12.2% | +75.8% | SF+RRF |
| PopQA | +17.9% | +62.3% | SF+RRF |
| SciFact | +5.2% | +5.6% | SF+Linear |
| nfcorpus | +7.5% | +6.8% | SF+Linear |

---

## Configuration

```yaml
grid_size: 64
splade: True
top_k: 100
fusion_method: linear  # یا rrf
rrf_k: 60
spreading_steps: 1
top_percent: 0.10
weighting: idf
smoothing_sigma: 1.5
```
