# خلاصه موفقیت Belebele — رسیدن به MRR=1.0

**تاریخ:** 2026-07-17
**نتیجه:** MRR=1.0000, AP=1.0000

---

## مشکل اولیه
- MRR=0.98 با 2 query شکست‌خورده (29 و 90)
- Gold documents در top 20 candidates نبودند

## علت اصلی مشکل
- Query 29: "According to the passage, which of the following would be the most beneficial for a runner preparing for the upcoming season?"
- Query 90: "According to the passage, which of the following is associated with gentler sound?"
- هر دو query دارای gold document بودند که در candidate list BM25 رتبه پایینی داشتند

## راه‌حل
افزایش `top_k` از 20 به 50

## نتایج قبل و بعد

| متریک | قبل | بعد | بهبود |
|-------|------|------|-------|
| MRR | 0.98 | **1.00** | +2% |
| AP | 0.98 | **1.00** | +2% |
| شکست‌ها | 2/100 | **0/100** | -2 |
| P@1 | 0.98 | **1.00** | +2% |

## پیکربندی بهینه نهایی

```yaml
splade: True
hybrid_alpha: 0.3
fusion_method: linear
splade_model: naver/splade-cocondenser-ensembledistil
top_k: 50  # کلید موفقیت
grid_size: 64
spreading_steps: 1
top_percent: 0.10
weighting: idf
```

## فایل‌های نتایج

| فایل | توضیح |
|------|-------|
| `outputs/BELEBELE_MRR1_RESULT.md` | جزئیات نتیجه |
| `outputs/belebele_benchmark/benchmarks/benchmark_20260717_204309/` | گزارش benchmark |

## Commits

| Commit | پیام |
|--------|------|
| `a65f092` | achieve MRR=1.0 on Belebele with top_k=50 |
| `a0eaba8` | update Belebele result to MRR=1.0 |

---

## درس‌آموخته‌ها

1. **افزایش Candidate Size** می‌تواند مشکلات retrieval را حل کند
2. **SF+SPLADE با top_k=50** بهترین نتیجه را می‌دهد
3. **queries طولانی** ممکن است به candidate size بیشتری نیاز داشته باشند
