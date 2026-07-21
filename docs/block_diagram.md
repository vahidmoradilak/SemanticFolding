# Semantic Folding Pipeline — Block Diagram

## 1. Overall Pipeline Architecture

```mermaid
flowchart TD
    subgraph INPUT["📥 Data Input"]
        A1["Corpus\n(TXT / JSONL)"]
        A2["Query\n(Single / Batch JSONL)"]
        A3["Config\nconfig/semantic_folding.yml"]
    end

    subgraph STEP1["🔧 Step 1\nPhrase Extraction"]
        B1["English: spaCy\nen_core_web_sm"]
        B2["Arabic: NLTK\nPOS-based extraction"]
        B3["Unigrams: NOUN, PROPN,\nADJ, VERB, ADV (≥2 chars)"]
        B4["Bigrams/Trigrams:\nNoun chunks + compounds"]
        B5["Output:\nextracted_phrases/"]
    end

    subgraph STEP2["📊 Step 2\nTerm Context"]
        C1["TF-IDF Weighting\n(global corpus statistics)"]
        C2["Co-occurrence Matrix\n(window size=5)"]
        C3["Output:\nterm_context_matrix/"]
    end

    subgraph STEP3["🗺️ Step 3\nSemantic Space"]
        D1["Dim Reduction:\nt-SNE / UMAP / PCA"]
        D2["Grid Mapping:\n64×64 Morton Z-order"]
        D3["Gaussian Smoothing\nσ=1.5"]
        D4["Output:\nsemantic_space/"]
    end

    subgraph STEP4["🧩 Step 4\nPhrase Fingerprints"]
        E1["Binary Grid per Phrase\n(cell=1 if phrase maps there)"]
        E2["Output:\nphrase_fingerprints/"]
    end

    subgraph STEP5["📄 Step 5\nDoc Fingerprints"]
        F1["Aggregate Phrase FPs\n(per document)"]
        F2["Normalization:\n√nnz / L2"]
        F3["Output:\ndoc_fingerprints/"]
    end

    subgraph STEP7["🔍 Step 7\nQuery Processor"]
        G1["Query Encoding\n→ phrase fingerprint"]
        G2["Spreading\n(radius=1, decay=0.5)"]
        G3["Geometric Scoring\n(optional 3×3 kernel)"]
        G4["Similarity:\ncosine vs doc FPs"]
        G5["Top-5% + Top-20 Rerank"]
        G6["Fusion:\nSPLADE (linear/RRF)"]
    end

    subgraph OUTPUT["📈 Output"]
        H1["Ranked Documents"]
        H2["Metrics:\nMRR, AP, P@K, NDCG"]
        H3["Reports:\nqa_evaluation_report.md"]
    end

    A1 --> STEP1 --> STEP2 --> STEP3 --> STEP4 --> STEP5
    A2 --> STEP7
    STEP5 --> STEP7
    A3 --> STEP1
    A3 --> STEP7
    STEP7 --> OUTPUT
```

---

## 2. Extraction Symmetry (English vs Arabic)

```mermaid
flowchart LR
    subgraph COMMON["Shared Design"]
        S["POS-pattern bigram strategy\nfor both languages"]
    end

    subgraph EN["🇬🇧 English — phrase_extractor.py"]
        E1["Tokenizer: spaCy"]
        E2["Unigrams:\npos ∈ {NOUN, PROPN, ADJ,\nVERB, ADV}, not is_stop"]
        E3["Bigrams:\nspaCy noun chunks +\nleft modifiers +\ncompound chains +\nconjunction expansion"]
        E4["Trigrams:\nnoun chunks +\nleft-anchored sub-spans"]
    end

    subgraph AR["🇸🇦 Arabic — lib.py:extract_raw_phrases_ar_fa"]
        A1["Tokenizer: NLTK"]
        A2["Unigrams:\n≥2 chars, not in\n_AR_FUNCTION_WORDS"]
        A3["Bigrams:\nPOS pattern [NN/JJ/VBN]\n+ [N*], both not function\nwords"]
        A4["Trigrams:\nStrict POS filter\n[NN/JJ/VBN] × 3"]
        A5["Clitic Stripping:\nال/و/ف/ب/ل/ک/س/بال/فل/\nول/فب + guard\n(stem≥2 + not function word)"]
    end

    subgraph OUTPUT_P["Output"]
        O["phrases.jsonl\n(unified format)"]
    end

    S --> EN
    S --> AR
    EN --> O
    AR --> O
```

---

## 3. Benchmarking Framework (3-Phase)

```mermaid
flowchart TD
    subgraph P1["Phase 1: Index"]
        I1["Input:\ndataset JSONL"]
        I2["build_combined_corpus()\n• Unique paragraphs\n• Dedup by paragraph_text\n• Assign doc_XXXXXX IDs"]
        I3["Run Steps 1–5\n(once per dataset)"]
        I4["Artifacts:\nruns/run_<ts>/\n├ corpus.txt\n├ config.yml\n├ query_doc_map.json\n├ query_gold.json\n├ extracted_phrases/\n├ term_context_matrix/\n├ semantic_space/\n├ phrase_fingerprints/\n└ doc_fingerprints/"]
        I1 --> I2 --> I3 --> I4
    end

    subgraph P2["Phase 2: Benchmark"]
        B1["Input:\npre-built fingerprints\n+ queries JSONL"]
        B2["query_processor.py\n(Step 7) per query"]
        B3["Post-filter to\n100 candidates\n(top_k=100)"]
        B4["SPLADE Fusion\n(optional linear/RRF)"]
        B5["Compute Metrics:\nMRR, AP, P@K, NDCG"]
        B6["Artifacts:\nbenchmarks/benchmark_<ts>/\n├ queries.txt\n├ results.tsv\n├ summary.json\n└ benchmark_report.md"]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6
    end

    subgraph P3["Phase 3: Report"]
        R1["Auto-generate\nbenchmark_report.md"]
        R2["Cross-dataset\ncomparison tables"]
        R3["Fusion method\ncomparison"]
        R1 --> R2 --> R3
    end

    subgraph OPTIMIZATIONS["🚀 Performance Optimizations"]
        O1["Vectorized Morton encoding\n5-10× speedup"]
        O2["Sparse scatter in\nfingerprint building\n5-10× speedup"]
        O3["Batch doc scoring\n10-50× speedup"]
        O4["Larger LRU caches\n30-50% less recomputation"]
        O5["Precomputed\ndocument norms"]
    end

    P1 --> P2 --> P3
    OPTIMIZATIONS -.-> P2
```

---

## 4. Dataset Adapters

```mermaid
flowchart LR
    subgraph ADAPTERS["Available Dataset Adapters"]
        A1["belebele_adapter.py\n(100 queries, 476 docs)"]
        A2["narrativeqa_adapter.py\n(50 queries, 19,585 docs)"]
        A3["pubmedqa_adapter.py\n(311 queries, 1,475 docs)"]
        A4["popqa_adapter.py\n(1,000 queries, 1,475 docs)"]
        A5["beir_adapter.py\nSciDocs, NFCorpus,\nSciFact, Quora,\nTREC-COVID, DBPedia"]
        A6["bioasq_adapter.py"]
        A7["hotpotqa_adapter.py"]
        A8["nq_rear_adapter.py"]
        A9["twowiki_adapter.py"]
    end

    subgraph COMMON_STRUCT["Common Structure"]
        C1["BaseAdapter\n(abstract)"]
        C2["download()"]
        C3["convert_to_musique_format()"]
        C4["Output:\nconverted/<name>.jsonl"]
    end

    ADAPTERS --> COMMON_STRUCT
```

---

## 5. Fusion Methods (SF + SPLADE)

```mermaid
flowchart TD
    subgraph METHODS["Fusion Methods"]
        M1["Pure SF\n(100% Semantic Folding)"]
        M2["Linear α\nscore = α·SF + (1-α)·SPLADE\nα ∈ [0, 1]"]
        M3["RRF\nscore = Σ 1/(k + rank_i)\nk=60"]
        M4["BM25 Baseline\n(Okapi BM25)"]
    end

    subgraph KEY_CONFIG["Key Config Changes"]
        K1["top_k: 20 → 100\n(more gold docs in pool)"]
        K2["splade: True\n(fusion enabled)"]
        K3["Batch + Vectorized Ops\n(10-50x speedup)"]
    end

    subgraph RESULTS["Final Results (top_k=100)"]
        R1["Belebele: RRF/Linear\nMRR=1.00, AP=1.00\n(+8.7% over Pure SF)"]
        R2["NarrativeQA: RRF\nMRR=1.00, AP=0.30\n(+9.9% over Pure SF)"]
        R3["PubMedQA: RRF\nMRR=1.00, AP=0.946\n(+12.2% over Pure SF)"]
        R4["PopQA: RRF\nMRR=0.990, AP=0.698\n(+17.9% over Pure SF)"]
        R5["SciFact: Linear\nMRR=0.966, AP=0.966\n(wins BM25: 0.947)"]
        R6["SciDocs: Linear\nMRR=0.947, AP=0.644\n(ties BM25: 0.946)"]
        R7["MuSiQue: Pure SF\nMRR=0.507, AP=0.306\n(BM25: 0.622)"]
        R8["nfcorpus: RRF\nMRR=0.647, AP=0.419\n(BM25: 0.686)"]
    end

    subgraph VARS["Key Variables"]
        V1["SPLADE Model:\nnaver/splade-cocondenser-\nensembledistil"]
        V2["Doc Vector Caching:\nsplade_doc_vectors.npy\n(auto-reuse)"]
        V3["Fusion Trigger:\n--splade --fusion-method\nrrf/linear --hybrid-alpha N"]
    end

    subgraph TAKEAWAY["Takeaways"]
        T1["RRF > Linear on most datasets\n(PubmedQA, PopQA, NarrativeQA)"]
        T2["top_k=100 is the KEY improvement\nfactor across all datasets"]
        T3["SF+SPLADE beats BM25 on\nBelebele, NarrativeQA, SciFact"]
    end

    METHODS --> RESULTS
    KEY_CONFIG --> RESULTS
    RESULTS --> VARS
    RESULTS --> TAKEAWAY
```

---

## 6. Project Directory Structure

```mermaid
mindmap
  root((Semantic Folding\nRoot))
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

## 7. Pipeline Configuration Parameters

```mermaid
flowchart LR
    subgraph GRID["Grid & Space"]
        G1["grid_size: 64×64"]
        G2["morton: True\n(Z-order encoding)"]
        G3["method: t-SNE\n(alt: UMAP, PCA)"]
        G4["smoothing_sigma: 1.5"]
    end

    subgraph QUERY["Query Processing"]
        Q1["spreading_steps: 1"]
        Q2["spreading_radius: 1"]
        Q3["spreading_decay: 0.5"]
        Q4["top_percent: 0.10"]
        Q5["top_k: 100\n(was 20 — KEY improvement)"]
        Q6["weighting: idf\n(alt: uniform)"]
        Q7["doc_norm: sqrt(nnz)"]
        Q8["query_norm: l2"]
    end

    subgraph SPLADE["SPLADE Fusion"]
        S1["splade: True\n(enables fusion)"]
        S2["model: naver/\nsplade-cocondenser-\nensembledistil"]
        S3["fusion_method:\nrrf (best) / linear"]
        S4["hybrid_alpha: 0.3\n(for linear)"]
        S5["rrf_k: 60"]
    end

    subgraph BENCH["Benchmark"]
        B1["seed: 42"]
        B2["sim_metric: cosine"]
        B3["score_norm: none"]
        B4["rerank: False"]
    end

    GRID --> QUERY --> SPLADE --> BENCH
```

---

## 8. Evaluation Metrics Pipeline

```mermaid
flowchart TD
    subgraph INPUT_M["Input"]
        I1["Query Q"]
        I2["Ranked docs:\n[d1, d2, ..., d20]"]
        I3["Gold docs:\n[g1, g2, ...]"]
    end

    subgraph METRICS["Metrics Computation"]
        M1["MRR\nMean Reciprocal Rank\n= 1/rank_of_first_gold"]
        M2["AP\nAverage Precision\n= Σ P@k / num_gold"]
        M3["P@K\nPrecision at K\n= relevant_in_top_K / K"]
        M4["R@K\nRecall at K\n= relevant_in_top_K / num_gold"]
        M5["NDCG@K\nNormalized Discounted\nCumulative Gain"]
    end

    subgraph OUTPUT_M["Output"]
        O1["Per-query metrics\nto results.tsv"]
        O2["Aggregate metrics\nto summary.json"]
        O3["Evaluation report\nto benchmark_report.md"]
    end

    INPUT_M --> METRICS --> OUTPUT_M
```

---

## 9. Version History

```
v3.2 ─── Current (Final)
  ├── top_k=100: KEY improvement (20→100)
  ├── SF+SPLADE fusion (RRF > Linear)
  ├── RRF MRR=1.00 on Belebele, NarrativeQA, PubMedQA
  ├── PopQA RRF MRR=0.990 (+17.9% over Pure SF)
  ├── SciDocs, SciFact, nfcorpus, MuSiQue benchmarks
  ├── Vectorized ops (5-50× speedup)
  ├── Belebele α sweep: best α=0.25–0.30
  ├── PopQA failure analysis
  ├── SPLADE doc vector caching
  └── Registry auto-backup on corruption

v3.1
  ├── Generic multi-dataset benchmark
  ├── 3-phase framework (index→benchmark→report)
  ├── BM25 baseline
  └── Quran benchmark (MRR=0.334)

v3.0
  ├── Arabic phrase extraction (lib.py)
  ├── Clitic stripping + function word filtering
  ├── English-Arabic extraction symmetry
  └── Bilingual groups removed
```
