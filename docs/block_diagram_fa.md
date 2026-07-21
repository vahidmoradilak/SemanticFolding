# خط‌لوله Semantic Folding — دیاگرام بلوکی

## 1. معماری کلی خط‌لوله

```mermaid
flowchart TD
    subgraph INPUT["📥 ورودی داده"]
        A1["متن\n(TXT / JSONL)"]
        A2["پرسش\n(تکی / دسته‌ای JSONL)"]
        A3["تنظیمات\nconfig/semantic_folding.yml"]
    end

    subgraph STEP1["🔧 گام 1\nاستخراج عبارت"]
        B1["انگلیسی: spaCy\nen_core_web_sm"]
        B2["عربی: NLTK\nاستخراج مبتنی بر POS"]
        B3["یک‌واژه‌ای: NOUN, PROPN,\nADJ, VERB, ADV (≥۲ حرف)"]
        B4["دو/سه‌واژه‌ای:\nگروه اسمی + ترکیبات"]
        B5["خروجی:\nextracted_phrases/"]
    end

    subgraph STEP2["📊 گام 2\nزمینه واژه"]
        C1["وزن‌دهی TF-IDF\n(آمار سراسری پیکره)"]
        C2["ماتریس هم‌رویدادی\n(اندازه پنجره=۵)"]
        C3["خروجی:\nterm_context_matrix/"]
    end

    subgraph STEP3["🗺️ گام 3\nفضای معنایی"]
        D1["کاهش ابعاد:\nt-SNE / UMAP / PCA"]
        D2["نگاشت شبکه‌ای:\n۶۴×۶۴ مرتبه Z مورتون"]
        D3["هموارسازی گاوسی\nσ=1.5"]
        D4["خروجی:\nsemantic_space/"]
    end

    subgraph STEP4["🧩 گام 4\nاثر انگاشت عبارت"]
        E1["شبکه دودویی هر عبارت\n(خانه=۱ اگر عبارت آنجاست)"]
        E2["خروجی:\nphrase_fingerprints/"]
    end

    subgraph STEP5["📄 گام 5\nاثر انگاشت سند"]
        F1["تجمع اثر انگاشت‌های عبارت\n(به‌ازای هر سند)"]
        F2["بهنجارش:\n√nnz / L2"]
        F3["خروجی:\ndoc_fingerprints/"]
    end

    subgraph STEP7["🔍 گام 7\nپردازش پرسش"]
        G1["رمزگذاری پرسش\n→ اثر انگاشت عبارت"]
        G2["گسترش\n(شعاع=۱، زوال=۰.۵)"]
        G3["امتیازدهی هندسی\n(اختیاری: هسته ۳×۳)"]
        G4["تشابه:\nکسینوسی با اثر انگاشت اسناد"]
        G5["۵٪ برتر + بازرتبه‌بندی ۲۰تایی"]
        G6["ادغام:\nSPLADE (خطی/RRF)"]
    end

    subgraph OUTPUT["📈 خروجی"]
        H1["اسناد رتبه‌بندی‌شده"]
        H2["معیارها:\nMRR, AP, P@K, NDCG"]
        H3["گزارش‌ها:\nqa_evaluation_report.md"]
    end

    A1 --> STEP1 --> STEP2 --> STEP3 --> STEP4 --> STEP5
    A2 --> STEP7
    STEP5 --> STEP7
    A3 --> STEP1
    A3 --> STEP7
    STEP7 --> OUTPUT
```

---

## 2. تقارن استخراج (انگلیسی در مقابل عربی)

```mermaid
flowchart LR
    subgraph COMMON["طراحی مشترک"]
        S["راهبرد استخراج دوتایی POS\nبرای هر دو زبان"]
    end

    subgraph EN["🇬🇧 انگلیسی — phrase_extractor.py"]
        E1["واژه‌پرداز: spaCy"]
        E2["یک‌واژه‌ای:\npos ∈ {NOUN, PROPN, ADJ,\nVERB, ADV}, نه is_stop"]
        E3["دوتایی:\nگروه اسمی spaCy +\nتوصیف‌گر چپ +\nزنجیره ترکیبی +\nگسترش حرف ربط"]
        E4["سه‌تایی:\nگروه اسمی +\nزیربازه‌های چپ‌مبدأ"]
    end

    subgraph AR["🇸🇦 عربی — lib.py:extract_raw_phrases_ar_fa"]
        A1["واژه‌پرداز: NLTK"]
        A2["یک‌واژه‌ای:\n≥۲ حرف، نه در\n_AR_FUNCTION_WORDS"]
        A3["دوتایی:\nالگوی POS [NN/JJ/VBN]\n+ [N*]، هر دو تابع‌واژه\nنباشند"]
        A4["سه‌تایی:\nفیلتر POS سختگیرانه\n[NN/JJ/VBN] × ۳"]
        A5["حذف وابسته‌ها:\nال/و/ف/ب/ل/ک/س/بال/فل/\nول/فب + نگهبان\n(بُن‌واژه≥۲ + تابع‌واژه نباشد)"]
    end

    subgraph OUTPUT_P["خروجی"]
        O["phrases.jsonl\n(قالب یکسان)"]
    end

    S --> EN
    S --> AR
    EN --> O
    AR --> O
```

---

## 3. چارچوب ارزیابی (۳ فاز)

```mermaid
flowchart TD
    subgraph P1["فاز 1: ایندکس"]
        I1["ورودی:\nJSONL دیتاست"]
        I2("build_combined_corpus()\n• پاراگراف‌های یکتا\n• حذف تکراری با paragraph_text\n• تخصیص شناسه doc_XXXXXX")
        I3["اجرای گام‌های ۱–۵\n(یک بار به ازای هر دیتاست)"]
        I4["مصنوعات:\nruns/run_<ts>/\n├ corpus.txt\n├ config.yml\n├ query_doc_map.json\n├ query_gold.json\n├ extracted_phrases/\n├ term_context_matrix/\n├ semantic_space/\n├ phrase_fingerprints/\n└ doc_fingerprints/"]
        I1 --> I2 --> I3 --> I4
    end

    subgraph P2["فاز 2: ارزیابی"]
        B1["ورودی:\nاثر انگاشت از پیش ساخته\n+ پرسش‌های JSONL"]
        B2("query_processor.py\n(گام ۷) به ازای هر پرسش")
        B3["پس‌فیلتر به\n۱۰۰ کاندید\n(top_k=100)"]
        B4["ادغام SPLADE\n(اختیاری: خطی/RRF)"]
        B5["محاسبه معیارها:\nMRR, AP, P@K, NDCG"]
        B6["مصنوعات:\nbenchmarks/benchmark_<ts>/\n├ queries.txt\n├ results.tsv\n├ summary.json\n└ benchmark_report.md"]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6
    end

    subgraph P3["فاز 3: گزارش"]
        R1["تولید خودکار\nbenchmark_report.md"]
        R2["جدول مقایسه\nچنددیتاستی"]
        R3["مقایسه روش‌های\nادغام"]
        R1 --> R2 --> R3
    end

    subgraph OPTIMIZATIONS["🚀 بهینه‌سازی کارایی"]
        O1["رمزگذاری برداری مورتون\n۵-۱۰× افزایش سرعت"]
        O2["درون‌ریزی تنک در\nساخت اثر انگاشت\n۵-۱۰× افزایش سرعت"]
        O3["امتیازدهی دسته‌ای اسناد\n۱۰-۵۰× افزایش سرعت"]
        O4["نهان‌گاه‌های LRU بزرگتر\n۳۰-۵۰٪ محاسبه مجدد کمتر"]
        O5["نرم‌های اسناد\nاز پیش محاسبه‌شده"]
    end

    P1 --> P2 --> P3
    OPTIMIZATIONS -.-> P2
```

---

## 4. تطبیق‌دهنده‌های دیتاست

```mermaid
flowchart LR
    subgraph ADAPTERS["تطبیق‌دهنده‌های موجود"]
        A1["belebele_adapter.py\n(۱۰۰ پرسش، ۴۷۶ سند)"]
        A2["narrativeqa_adapter.py\n(۵۰ پرسش، ۱۹۵۸۵ سند)"]
        A3["pubmedqa_adapter.py\n(۳۱۱ پرسش، ۱۴۷۵ سند)"]
        A4["popqa_adapter.py\n(۱۰۰۰ پرسش، ۱۴۷۵ سند)"]
        A5["beir_adapter.py\nSciDocs, NFCorpus,\nSciFact, Quora,\nTREC-COVID, DBPedia"]
        A6["bioasq_adapter.py"]
        A7["hotpotqa_adapter.py"]
        A8["nq_rear_adapter.py"]
        A9["twowiki_adapter.py"]
    end

    subgraph COMMON_STRUCT["ساختار مشترک"]
        C1["BaseAdapter\n(چکیده)"]
        C2("download()\n(دریافت)")
        C3("convert_to_musique_format()\n(تبدیل به قالب)")
        C4["خروجی:\nconverted/<name>.jsonl"]
    end

    ADAPTERS --> COMMON_STRUCT
```

---

## 5. روش‌های ادغام (SF + SPLADE)

```mermaid
flowchart TD
    subgraph METHODS["روش‌های ادغام"]
        M1["Pure SF\n(۱۰۰٪ Semantic Folding)"]
        M2["خطی α\nنمره = α·SF + (۱-α)·SPLADE\nα ∈ [0, 1]"]
        M3["RRF\nنمره = Σ ۱/(k + rank_i)\nk=۶۰"]
        M4["BM25 پایه\n(Okapi BM25)"]
    end

    subgraph KEY_CONFIG["تغییرات کلیدی تنظیمات"]
        K1["top_k: ۲۰ → ۱۰۰\n(اسناد طلایی بیشتر در استخر)"]
        K2["splade: True\n(فعال‌سازی ادغام)"]
        K3["عملیات برداری و دسته‌ای\n(۱۰-۵۰× افزایش سرعت)"]
    end

    subgraph RESULTS["نتایج نهایی (top_k=۱۰۰)"]
        R1["Belebele: RRF/خطی\nMRR=۱.۰۰, AP=۱.۰۰\n(+۸.۷٪ نسبت به Pure SF)"]
        R2["NarrativeQA: RRF\nMRR=۱.۰۰, AP=۰.۳۰\n(+۹.۹٪ نسبت به Pure SF)"]
        R3["PubMedQA: RRF\nMRR=۱.۰۰, AP=۰.۹۴۶\n(+۱۲.۲٪ نسبت به Pure SF)"]
        R4["PopQA: RRF\nMRR=۰.۹۹۰, AP=۰.۶۹۸\n(+۱۷.۹٪ نسبت به Pure SF)"]
        R5["SciFact: خطی\nMRR=۰.۹۶۶, AP=۰.۹۶۶\n(بهتر از BM25: ۰.۹۴۷)"]
        R6["SciDocs: خطی\nMRR=۰.۹۴۷, AP=۰.۶۴۴\n(مساوی BM25: ۰.۹۴۶)"]
        R7["MuSiQue: Pure SF\nMRR=۰.۵۰۷, AP=۰.۳۰۶\n(BM25: ۰.۶۲۲)"]
        R8["nfcorpus: RRF\nMRR=۰.۶۴۷, AP=۰.۴۱۹\n(BM25: ۰.۶۸۶)"]
    end

    subgraph VARS["متغیرهای کلیدی"]
        V1["مدل SPLADE:\nnaver/splade-cocondenser-\nensembledistil"]
        V2["نهان‌گاه بردار سند:\nsplade_doc_vectors.npy\n(استفاده مجدد خودکار)"]
        V3["فعال‌سازی ادغام:\n--splade --fusion-method\nrrf/linear --hybrid-alpha N"]
    end

    subgraph TAKEAWAY["نتیجه‌گیری"]
        T1["RRF > خطی در اکثر دیتاست‌ها\n(PubmedQA, PopQA, NarrativeQA)"]
        T2["top_k=۱۰۰ عامل اصلی بهبود\nدر تمام دیتاست‌ها"]
        T3["SF+SPLADE از BM25 بهتر است در\nBelebele, NarrativeQA, SciFact"]
    end

    METHODS --> RESULTS
    KEY_CONFIG --> RESULTS
    RESULTS --> VARS
    RESULTS --> TAKEAWAY
```

---

## 6. ساختار دایرکتوری پروژه

```mermaid
mindmap
  root((Semantic Folding\nریشه))
    semantic_folding/
      semantic_folder.py
      phrase_extractor.py
      term_context.py
      semantic_space.py
      phrase_fingerprints.py
      doc_fingerprints.py
      customtext_fingerprints.py
      query_processor.py
      lib.py
      phrase_visualizer.py
      doc_visualizer.py
      dataset_benchmark/
        generic_benchmark.py
        beir_benchmark_adapter.py
        bm25_benchmark.py
        run_all_benchmarks.py
        runs_registry.yml
        adapters/
          __init__.py
          base_adapter.py
          beir_adapter.py
          belebele_adapter.py
          bioasq_adapter.py
          hotpotqa_adapter.py
          narrativeqa_adapter.py
          nq_rear_adapter.py
          popqa_adapter.py
          pubmedqa_adapter.py
          twowiki_adapter.py
        musique/
          run_benchmark.py
          benchmark_analyzer.py
        quran/
          run_benchmark.py
        notebooks/
    config/
      semantic_folding.yml
      exec_state.yml
    data/
      datasets/
        <name>/
          converted/<name>.jsonl
        ...
      qa-sample.md
    outputs/
      <dataset>_benchmark/
        runs/run_<ts>/
          corpus.txt
          config.yml
          extracted_phrases/
          term_context_matrix/
          semantic_space/
          phrase_fingerprints/
          doc_fingerprints/
        benchmarks/benchmark_<ts>/
          queries.txt
          results.tsv
          summary.json
          benchmark_report.md
```

---

## 7. پارامترهای پیکربندی خط‌لوله

```mermaid
flowchart LR
    subgraph GRID["شبکه و فضا"]
        G1["grid_size: ۶۴×۶۴"]
        G2["morton: True\n(رمزگذاری مرتبه Z)"]
        G3["method: t-SNE\n(جایگزین: UMAP, PCA)"]
        G4["smoothing_sigma: ۱.۵"]
    end

    subgraph QUERY["پردازش پرسش"]
        Q1["spreading_steps: ۱"]
        Q2["spreading_radius: ۱"]
        Q3["spreading_decay: ۰.۵"]
        Q4["top_percent: ۰.۱۰"]
        Q5["top_k: ۱۰۰\n(بود ۲۰ — بهبود کلیدی)"]
        Q6["weighting: idf\n(جایگزین: uniform)"]
        Q7["doc_norm: sqrt(nnz)"]
        Q8["query_norm: l2"]
    end

    subgraph SPLADE["ادغام SPLADE"]
        S1["splade: True\n(فعال‌سازی ادغام)"]
        S2["مدل: naver/\nsplade-cocondenser-\nensembledistil"]
        S3["روش ادغام:\nrrf (بهترین) / خطی"]
        S4["hybrid_alpha: ۰.۳\n(برای خطی)"]
        S5["rrf_k: ۶۰"]
    end

    subgraph BENCH["ارزیابی"]
        B1["seed: ۴۲"]
        B2["sim_metric: کسینوسی"]
        B3["score_norm: none"]
        B4["rerank: False"]
    end

    GRID --> QUERY --> SPLADE --> BENCH
```

---

## 8. خط‌لوله معیارهای ارزیابی

```mermaid
flowchart TD
    subgraph INPUT_M["ورودی"]
        I1["پرسش Q"]
        I2["اسناد رتبه‌بندی‌شده:\n[d1, d2, ..., d100]"]
        I3["اسناد طلایی:\n[g1, g2, ...]"]
    end

    subgraph METRICS["محاسبه معیارها"]
        M1["MRR\nمیانگین وارون رتبه\n= ۱/رتبه_نخستین_طلایی"]
        M2["AP\nمیانگین دقت\n= Σ P@k / تعداد_طلایی"]
        M3["P@K\nدقت در K\n= مرتبط_در_K_بالا / K"]
        M4["R@K\nبازیابی در K\n= مرتبط_در_K_بالا / تعداد_طلایی"]
        M5["NDCG@K\nسود تجمعی تنزیل‌شده\nبهنجار"]
    end

    subgraph OUTPUT_M["خروجی"]
        O1["معیارهای هر پرسش\nبه results.tsv"]
        O2["معیارهای کلان\nبه summary.json"]
        O3["گزارش ارزیابی\nبه benchmark_report.md"]
    end

    INPUT_M --> METRICS --> OUTPUT_M
```

---

## 9. تاریخچه نسخه‌ها

```
v3.2 ─── جاری (نهایی)
  ├── top_k=۱۰۰: بهبود کلیدی (۲۰→۱۰۰)
  ├── ادغام SF+SPLADE (RRF > خطی)
  ├── RRF MRR=۱.۰۰ روی Belebele, NarrativeQA, PubMedQA
  ├── PopQA RRF MRR=۰.۹۹۰ (+۱۷.۹٪ نسبت به Pure SF)
  ├── ارزیابی SciDocs, SciFact, nfcorpus, MuSiQue
  ├── عملیات برداری (۵-۵۰× افزایش سرعت)
  ├── جستجوی α در Belebele: بهترین α=۰.۲۵–۰.۳۰
  ├── تحلیل شکست PopQA
  ├── نهان‌گاه بردار سند SPLADE
  └── پشتیبان‌گیری خودکار ثبات در خرابی

v3.1
  ├── چارچوب ارزیابی چنددیتاستی عمومی
  ├── چارچوب ۳ فاز (ایندکس→ارزیابی→گزارش)
  ├── پایه BM25
  └── ارزیابی قرآن (MRR=۰.۳۳۴)

v3.0
  ├── استخراج عبارت عربی (lib.py)
  ├── حذف وابسته‌ها + فیلتر تابع‌واژه
  ├── تقارن استخراج انگلیسی-عربی
  └── حذف گروه‌های دوزبانه
```
