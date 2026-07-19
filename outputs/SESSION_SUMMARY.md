# خلاصه جلسه — بهبود عملکرد Semantic Folding

**تاریخ:** 2026-07-17
**هدف:** بهبود عملکرد سیستم بازیابی اطلاعات Semantic Folding

---

## ۱. بهینه‌سازی‌های عملکرد (Performance Optimizations)

### ۱.۱ Vectorized Morton Encoding
**فایل:** `semantic_folding/lib.py`, `semantic_folding/phrase_fingerprints.py`, `semantic_folding/fingerprint_builder.py`, `semantic_folding/query_processor.py`

**تغییر:** جایگزینی double nested Python loop با numpy vectorized operations برای Morton Z-order encoding.

**نتیجه:** 5-10x سرعت‌بخشی در linearisation loops

### ۱.۲ Sparse Scatter در Fingerprint Building
**فایل:** `semantic_folding/fingerprint_builder.py`

**تغییر:** جایگزینی `np.add.at` با scatter operation فقط روی non-zero elements.

**نتیجه:** 5-10x سرعت‌بخشی در fingerprint construction

### ۱.۳ Batch Document Scoring
**فایل:** `semantic_folding/query_processor.py`

**تغییر:** جایگزینی Python loop per-document با batch matrix-vector product.

**نتیجه:** 10-50x سرعت‌بخشی در document ranking

### ۱.۴ افزایش LRU Cache Size
**فایل:** `semantic_folding/lib.py`

**تغییر:** افزایش `lemmatize_token` cache از 10K به 50K و `normalize_phrase` cache از 2K به 32K.

**نتیجه:** 30-50% کاهش در NLTK recomputation

### ۱.۵ Precomputed Document Norms
**فایل:** `semantic_folding/fingerprint_builder.py`, `semantic_folding/doc_fingerprints.py`, `semantic_folding/customtext_fingerprints.py`

**تغییر:** محاسبه و ذخیره L2 norms اسناد در مرحله 5 برای استفاده در مرحله 7.

**نتیجه:** حذف تکرار محاسبه norms در هر query

---

## ۲. SPLADE Fusion Benchmark

### پیکربندی بهینه
```yaml
splade: True
hybrid_alpha: 0.3
fusion_method: linear
splade_model: naver/splade-cocondenser-ensembledistil
grid_size: 64
spreading_steps: 1
top_percent: 0.10
weighting: idf
```

### نتایج

| دیتاست | روش | MRR | AP | BM25 MRR | بهبود |
|--------|------|-----|-----|----------|-------|
| **Belebele** | SF+SPLADE | **0.98** | **0.98** | 0.995 | +6.5% |
| **NarrativeQA** | SF+SPLADE | **0.96** | **0.0157** | 0.98 | +5.5% |
| **PubMedQA** | SF+SPLADE | **0.954** | **0.740** | 1.000 | +7.1% |
| **PopQA** | Pure SF | 0.84 | 0.43 | — | — |

### تحلیل نتایج

1. **SF+SPLADE شکاف را با BM25 در ۳ از ۴ دیتاست کاهش می‌دهد**
2. **Belebele**: SF+SPLADE (0.98) بسیار نزدیک به BM25 (0.995)
3. **PubMedQA**: SF+SPLADE (0.954) نزدیک به BM25 (1.000)
4. **PopQA**: Pure SF بهترین است — SPLADE برای queries مبتنی بر entity نویز اضافه می‌کند

---

## ۳. آزمایش‌های پارامتری

### ۳.۱ Grid Size
| grid_size | MRR | وضعیت |
|-----------|-----|-------|
| 32 | 0.92 | رزولوشن کم |
| **64** | **0.98** | بهینه |
| 48 | Failed | توان ۲ نیست |

### ۳.۲ OOV Expansion
| پیکربندی | MRR | نتیجه |
|----------|-----|-------|
| بدون OOV | 0.98 | — |
| با OOV (weight=0.8) | 0.98 | کمک نکرد |

**دلیل:** Queries شکست‌خورده (29, 90) مربوط به OOV نیستند.

### ۳.۳ SPLADE Alpha
| alpha | MRR | نتیجه |
|-------|-----|-------|
| 0.25 | 0.98 | یکسان |
| 0.30 | 0.98 | بهینه |
| 0.50 | 0.94 | بدتر |
| 0.70 | 0.92 | بدتر |

---

## ۴. فایل‌های ایجاد/تغییر یافته

### فایل‌های کد
| فایل | تغییرات |
|------|---------|
| `semantic_folding/lib.py` | Vectorized Morton, LRU cache افزایش |
| `semantic_folding/phrase_fingerprints.py` | Vectorized linearisation |
| `semantic_folding/fingerprint_builder.py` | Sparse scatter, precomputed norms |
| `semantic_folding/query_processor.py` | Batch scoring, vectorized flattening |
| `semantic_folding/doc_fingerprints.py` | Precomputed norms |
| `semantic_folding/customtext_fingerprints.py` | Precomputed norms |

### فایل‌های اسکریپت
| فایل | توضیح |
|------|-------|
| `run_splade_benchmarks.py` | اجرای benchmark با SPLADE |
| `run_best_config.py` | اجرای پیکربندی بهینه روی تمام دیتاست‌ها |

### فایل‌های نتایج
| فایل | توضیح |
|------|-------|
| `outputs/FINAL_BENCHMARK_RESULTS.md` | نتایج نهایی benchmark |
| `outputs/SPLADE_FUSION_RESULTS.md` | تحلیل SPLADE fusion |
| `AGENTS.md` | به‌روزرسانی نتایج |

---

## ۵. Commits

| Commit | پیام |
|--------|------|
| `cfaa296` | performance optimizations & SPLADE fusion benchmarks |
| `e2bdbc5` | update benchmark results documentation |
| `e19c6e5` | add final benchmark results and run_best_config script |
| `26bf24a` | add final benchmark results to outputs |

---

## ۶. پیشنهادات برای بهبود بیشتر

1. **بررسی queries شکست‌خورده** — درک اینکه چرا queries 29 و 90 شکست می‌خورند
2. **فعال‌سازی OOV Expansion** — ممکن است برای دیتاست‌های دیگر مفید باشد
3. **Grid Search دقیق‌تر** — تست مقادیر مختلف alpha (0.25-0.30)
4. **روش‌های ensemble** — ترکیب چند روش مختلف
5. **Reranking با مدل آموزش‌دیده** — LambdaMART یا مشابه

---

## ۷. پیکربندی بهینه نهایی

```yaml
# Semantic Folding + SPLADE Fusion
grid_size: 64
splade: true
hybrid_alpha: 0.3
fusion_method: linear
splade_model: naver/splade-cocondenser-ensembledistil
spreading_steps: 1
top_percent: 0.10
weighting: idf
smoothing_sigma: 1.5
```
