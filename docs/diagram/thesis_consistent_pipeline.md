# Thesis Pipeline — Consistent Reference Version

## هدف
این فایل به‌عنوان **مرجع واحد (Single Source of Truth)** برای pipeline پایان‌نامه تنظیم شده است تا شماره‌گذاری مراحل، تنظیمات benchmark و اصطلاحات در block diagram، pseudocode، methodology و results یکسان باشند.

## Pipeline نهایی

```text
Input Corpus
    ↓
Step 1 — Phrase Extraction
    ↓
Step 2 — Term-Context Matrix
    ↓
Step 3 — Semantic Space
    ↓
Step 4 — Phrase Fingerprints
    ↓
Step 5 — Document Fingerprints
    ↓
Step 6 — Query Encoding & Retrieval
    ↓
Evaluation
```

> **Custom Text Fingerprint** یک قابلیت کمکی برای تولید/نمایش fingerprint متن دلخواه است و به‌عنوان یک Step مستقل از pipeline اصلی شمارش نمی‌شود.

---

## Step 1 — Phrase Extraction

### English
با spaCy و الگوهای POS، unigram، noun chunk، bigram، trigram، compound chain و conjunction-expanded phrase استخراج می‌شوند.

### Arabic/Persian
با tokenization، POS filtering، function-word filtering، نرمال‌سازی عربی و clitic stripping، phraseها استخراج می‌شوند.

خروجی:

```text
extracted_phrases/
```

---

## Step 2 — Term-Context Matrix

برای phraseهای استخراج‌شده، ماتریس context ساخته و TF-IDF اعمال می‌شود:

\[
IDF(t)=\lograc{N}{df(t)}
\]

که در آن N تعداد اسناد و df تعداد اسنادی است که phrase/term در آن ظاهر شده است.

خروجی:

```text
term_context_matrix/
├── term_context_matrix.npz
└── idf_weights.json
```

---

## Step 3 — Semantic Space

ماتریس sparse به فضای دوبعدی semantic نگاشت می‌شود.

تنظیم benchmark:

```text
UMAP
n_neighbors = 15
min_dist    = 0.1
metric      = cosine
```

سپس مختصات به grid نهایی نگاشت می‌شوند:

```text
64 × 64
```

بنابراین هر fingerprint دارای:

\[
64	imes64=4096
\]

cell است.

> آزمایش‌های 128×128 یا سایر اندازه‌ها باید به‌عنوان **preliminary/sensitivity experiments** گزارش شوند، نه configuration نهایی.

خروجی:

```text
semantic_space/
└── context_coordinates.json
```

---

## Step 4 — Phrase Fingerprints

هر phrase مختصات `(row, col)` خود را از semantic space می‌گیرد و cell متناظر روی grid فعال می‌شود.

در ادامه، بسته به configuration:

- Morton Z-order
- Gaussian smoothing
- sparsification / peak selection

اعمال می‌شوند.

خروجی:

```text
phrase_fingerprints/
├── phrase_fingerprints.npz
└── phrase_fingerprints_meta.json
```

---

## Step 5 — Document Fingerprints

برای هر document:

```text
document
  ↓
extract phrases
  ↓
retrieve phrase fingerprints
  ↓
weighted aggregation
  ↓
normalization
```

به‌صورت مفهومی:

\[
D_i=\sum_{p\in P_i}w(p)FP(p)
\]

که در آن `w(p)` وزن phrase، معمولاً بر اساس TF-IDF/IDF، است.

### Normalization

طبق configuration مورد استفاده:

```text
weighted aggregation
        ↓
L2 normalization
        ↓
sqrt(nnz) scaling
```

که:

\[
nnz(x)=|\{j:x_j
eq0\}|
\]

و:

\[
\sqrt{nnz(x)}
\]

ریشه دوم تعداد cellهای غیرصفر است.

خروجی:

```text
doc_fingerprints/
├── doc_fingerprints.npz
└── doc_fingerprints_meta.json
```

---

## قابلیت کمکی — Custom Text Fingerprint

متن دلخواه می‌تواند از همان pipeline برای ساخت fingerprint عبور کند:

```text
Custom Text
    ↓
Phrase Extraction
    ↓
Phrase Fingerprint Lookup
    ↓
Aggregation
    ↓
Custom Text Fingerprint
```

کاربردها:

- visualization
- debugging
- qualitative analysis
- تست queryهای دلخواه
- بررسی رفتار semantic representation

این قابلیت **Step مستقل pipeline اصلی نیست**.

---

## Step 6 — Query Encoding & Retrieval

Query با همان فضای semantic به fingerprint تبدیل می‌شود.

```text
Query
 ↓
Phrase Extraction
 ↓
Phrase Fingerprints
 ↓
Query Fingerprint
```

### Spreading

با تنظیم benchmark:

```text
spreading_steps = 1
spreading_decay = 0.5
```

cellهای فعال query به Moore neighbourhood گسترش می‌یابند.

### Similarity

معیار اصلی:

\[
cos(q,d)=rac{q\cdot d}{\|q\|_2\|d\|_2}
\]

### Candidate Pool

در benchmark نهایی:

```text
top_k = 100
```

> اگر در آزمایش‌های اولیه `top_k=5` یا `20` استفاده شده، آن‌ها به‌عنوان preliminary experiments گزارش شوند.

### SPLADE Fusion

#### Linear

\[
Score=lpha Score_{SF}+(1-lpha)Score_{SPLADE}
\]

تنظیم گزارش‌شده:

```text
alpha ≈ 0.3
```

#### RRF

\[
RRF(d)=\sum_irac{1}{k+rank_i(d)}
\]

با:

```text
k = 60
```

---

# Evaluation

معیارهای اصلی:

```text
MRR
AP
P@K
R@K
NDCG@K
```

نتایج به‌صورت per-query و aggregate قابل گزارش هستند.

---

# Final Benchmark Configuration

| Parameter | Final configuration |
|---|---|
| Grid | 64 × 64 |
| Dimensions | 4096 |
| Reduction | UMAP |
| UMAP n_neighbors | 15 |
| UMAP min_dist | 0.1 |
| UMAP metric | cosine |
| Spreading steps | 1 |
| Spreading decay | 0.5 |
| Weighting | TF-IDF / IDF-based |
| top_k | 100 |
| SPLADE | optional |
| Linear alpha | ≈ 0.3 |
| RRF k | 60 |

---

# Preliminary vs Final Experiments

برای جلوگیری از تناقض، نتایج باید به دو دسته تقسیم شوند.

### Preliminary / Sensitivity

مثلاً:

```text
grid = 64 / 128
top_k = 5 / 20 / 100
spreading = 0 / 1 / 2
different smoothing values
different weighting methods
different reduction methods
```

### Final Benchmark

```text
grid = 64 × 64
top_k = 100
spreading_steps = 1
spreading_decay = 0.5
UMAP:
    n_neighbors = 15
    min_dist = 0.1
    metric = cosine
```

---

# Benchmark Architecture

## Phase 1 — Index

```text
Dataset JSONL
    ↓
Build combined corpus
    ↓
Deduplicate passages
    ↓
Assign document IDs
    ↓
Steps 1–5
    ↓
Pre-built fingerprints
```

Artifacts:

```text
runs/run_*/
├── corpus.txt
├── config.yml
├── query_doc_map.json
├── query_gold.json
├── extracted_phrases/
├── term_context_matrix/
├── semantic_space/
├── phrase_fingerprints/
└── doc_fingerprints/
```

## Phase 2 — Retrieval Benchmark

```text
Queries
    ↓
Step 6: Query Encoding
    ↓
Spreading
    ↓
Document Scoring
    ↓
Top-100 candidates
    ↓
Optional SPLADE Fusion
    ↓
Metrics
```

## Phase 3 — Report

```text
Per-query results
    ↓
Aggregate metrics
    ↓
Dataset comparison
    ↓
Fusion comparison
    ↓
Benchmark report
```

---

# Baselines

مقایسه اصلی:

```text
Pure SF
SF + SPLADE Linear
SF + SPLADE RRF
BM25
```

---

# Terminology

| Term | Meaning |
|---|---|
| Term | واحد vocabulary در term-context space |
| Phrase | واحد زبانی استخراج‌شده از متن |
| Phrase Fingerprint | نمایش فضایی یک phrase |
| Document Fingerprint | نمایش تجمیعی یک document |
| Query Fingerprint | نمایش فضایی query |
| Semantic Space | فضای دوبعدی مختصات phraseها |
| Semantic Grid | grid گسسته 64×64 |
| Spreading | گسترش فضایی cellهای فعال |
| Morton Encoding | نگاشت Z-order مختصات دوبعدی |
| Candidate Pool | مجموعه اسناد نگه‌داشته‌شده برای ranking/reranking |
| Fusion | ترکیب امتیاز/رتبه SF و SPLADE |

---

# Canonical Pipeline

```text
Corpus
  ↓
Phrase Extraction
  ↓
Term-Context Matrix
  ↓
Semantic Space
  ↓
Phrase Fingerprints
  ↓
Document Fingerprints
  ↓
Query Encoding
  ↓
Spreading
  ↓
Similarity / Ranking
  ↓
Optional SPLADE Fusion
  ↓
Evaluation
```

این توالی، مرجع اصلی برای هماهنگ‌سازی block diagram، pseudocode، methodology و benchmark documentation است.
