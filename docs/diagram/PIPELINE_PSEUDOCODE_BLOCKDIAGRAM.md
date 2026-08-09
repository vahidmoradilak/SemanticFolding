# Mixed Arabic-English Benchmark — Pipeline Documentation

**Project:** Semantic Folding Pipeline — Cross-Lingual & Bilingual Retrieval  
**Date:** 2026-08-08  
**Corpus:** Belebele (Arabic-English bilingual, 488 unique passages)

---

## 1. Overview

This document provides a detailed description of the mixed Arabic-English benchmark pipeline, including its pseudocode and a block diagram of the overall architecture. The benchmark evaluates Semantic Folding (SF) and BM25 on cross-lingual queries composed of mixed Arabic and English phrases against a bilingual passage corpus.

### Variants Tested

| Variant | Description |
|---------|-------------|
| Pure SF | Semantic Folding retrieval only (cosine similarity on 2D grid fingerprints) |
| SF + SPLADE Linear | Linear fusion: `score = α·SF + (1-α)·SPLADE` with α = 0.3 |
| SF + SPLADE RRF | Rank Reciprocal Fusion with k = 60 |
| BM25 | Baseline lexical scoring (sklearn CountVectorizer + numpy) |

### Key Results

| Variant | MRR | AP | P@1 | R@5 | NDCG@20 |
|---------|:---:|:---:|:---:|:---:|:-------:|
| **SF + SPLADE Linear** | **0.8231** | **0.8231** | **0.7705** | **0.8852** | **0.8470** |
| **Pure SF** | **0.8086** | **0.8086** | **0.7500** | **0.8730** | **0.8407** |
| **BM25** | 0.7854 | 0.7854 | 0.7193 | 0.8689 | 0.8233 |
| SF + SPLADE RRF | 0.6827 | 0.6827 | 0.5861 | 0.8340 | 0.7426 |

---

## 2. Pseudocode

```
================================================================================
  MIXED ARABIC-ENGLISH BENCHMARK PIPELINE
================================================================================

--------------------------------------------------------------------------------
PHASE 0: DATA PREPARATION
--------------------------------------------------------------------------------

INPUT:  corpus_belebele_ar_en_deduped.csv  (488 rows)
        Columns: id, passage, question_1, question_2

FOR each row i IN [0 .. 487]:
    corpus_texts[i]  ← row["passage"]          # "Arabic text | English text"
    gold_answer[i]   ← corpus_texts[i]          # self-retrieval
    doc_ids[i]       ← "doc_" ++ zero_pad(i, 6)

CALL write_corpus(corpus_texts, doc_ids)
    → writes corpus.txt:  "doc_XXXXXX, passage <text>"  per line

--------------------------------------------------------------------------------
PHASE 1: QUERY GENERATION (SF Phrase Extraction)
--------------------------------------------------------------------------------

FOR each row i IN [0 .. 487]:
    question ← row["question_1"]
    [ar_part, en_part] ← SPLIT question ON " | "  (first occurrence only)

    // ── Arabic phrase extraction ──────────────────────────────
    ar_tokens  ← nltk_word_tokenize(ar_part)
    ar_tagged  ← nltk.pos_tag(ar_tokens)
    ar_phrases ← ∅

    // Unigrams
    FOR each token t IN ar_tokens:
        nt ← normalize_arabic(t)       // unicode normalisation
        IF len(nt) ≥ 2 AND nt ∉ _AR_FUNCTION_WORDS:
            ar_phrases.ADD(nt)

    // Bigrams (POS-filtered)
    FOR each pair (t₁, t₂) IN consecutive(ar_tagged):
        IF t₁.tag ∈ {NN,NNS,NNP,NNPS,JJ,JJR,JJS,VBN}
           AND t₂.tag STARTS_WITH 'N':
            n1, n2 ← normalize_arabic(t₁), normalize_arabic(t₂)
            IF n1 ∉ _AR_FUNCTION_WORDS AND n2 ∉ _AR_FUNCTION_WORDS:
                ar_phrases.ADD(n1 ++ " " ++ n2)

    // Trigrams (strict POS filter)
    FOR each triple (t₁, t₂, t₃) IN consecutive(ar_tagged):
        IF all(t.tag ∈ {NN,JJ,VBN} for t in [t₁,t₂,t₃]):
            ... (filter function words, min length 2)
            ar_phrases.ADD(join(" ", [n1, n2, n3]))

    // Clean up: strip residual punctuation, keep only len ≥ 2
    ar_phrases ← [strip_punct(n) FOR n IN ar_phrases IF len(strip_punct(n)) ≥ 2]

    // ── English phrase extraction ─────────────────────────────
    en_clean ← normalize_hyphens(en_part)
    doc      ← nlp(en_clean)                          // spaCy en_core_web_sm
    en_phrases ← extract_raw_phrases_spacy(doc)
        // Returns list of: noun chunks, named entities, left-modifier chains,
        // compound chains, conjunction-expanded phrases, bare head nouns

    // ── Combine ────────────────────────────────────────────────
    mixed_query[i] ← join(" ", ar_phrases ++ en_phrases)

OUTPUT: queries.txt  (one mixed query per line, 488 lines)

--------------------------------------------------------------------------------
PHASE 2: INDEXING — Steps 1 → 2 → 3 → 4 → 5  (runs ONCE)
--------------------------------------------------------------------------------

// ── Step 1: Phrase Extraction ──────────────────────────────────────────
READ corpus.txt  (already written in Phase 0)

FOR each passage IN corpus:
    EXTRACT phrases using the same Arabic + English extraction logic
    // (reads corpus line-by-line, calls extract_raw_phrases_ar_fa + spacy)

BUILD vocabulary:
    min_freq       = 1
    min_word_length = 2
    keep_verbs     = True

OUTPUT:
    extracted_phrases/vocabulary.csv
    extracted_phrases/phrase_to_contexts.json

// ── Step 2: Term-Context Matrix ────────────────────────────────────────
FOR each passage IN corpus:
    EXTRACT phrase occurrences
    COUNT co-occurrences within window

APPLY TF-IDF weighting:
    idf_weights.json      ← log(N / df) per phrase
    term_context_matrix.npz ← sparse TF-IDF matrix

OUTPUT: term_context_matrix/

// ── Step 3: Semantic Space Mapping ─────────────────────────────────────
    Load term_context_matrix (sparse N × V)
    DIMENSIONALITY REDUCTION → 2D (64 × 64 grid)

    Method: UMAP
        n_neighbors  = 15
        min_dist     = 0.1
        metric       = cosine

OUTPUT: semantic_space/context_coordinates.json  (N rows × 2 cols)

// ── Step 4: Phrase Fingerprints ────────────────────────────────────────
FOR each phrase IN vocabulary:
    (row, col) ← context_coordinates[phrase]          // 2D position
    Activate cell (row, col) on 64×64 zero grid
    APPLY Morton Z-order encoding (interleave row/col bits)
    APPLY Gaussian blur with σ = 1.5

OUTPUT: phrase_fingerprints/phrase_fingerprints.npz  (N_phrases × 4096)
        phrase_fingerprints/phrase_fingerprints_meta.json

// ── Step 5: Document Fingerprints ──────────────────────────────────────
FOR each passage IN corpus:
    phrases ← EXTRACT_PHRASES(passage)               // Phase 1 pipeline
    fp      ← SUM( phrase_fingerprint[p] FOR p IN phrases, WEIGHTED BY TF-IDF)
    fp      ← L2_NORMALIZE(fp)
    fp      ← MULTIPLY_BY sqrt(number_of_nonzero_cells)

OUTPUT: doc_fingerprints/doc_fingerprints.npz  (488 × 4096)
        doc_fingerprints/doc_fingerprints_meta.json

ALSO SAVE (for Phase 3):
    query_doc_map.json     ← { "0": ["doc_000000", ..., "doc_000487"], ... }
                               // each query maps to ALL 488 docs
    query_gold.json        ← { "0": ["doc_000000"], "1": ["doc_000001"], ... }
                               // each query's gold = the passage it came from

--------------------------------------------------------------------------------
PHASE 3: QUERY PROCESSING  (Step 6) — Runs per variant
--------------------------------------------------------------------------------

// The following steps are executed ONCE per benchmark variant

FOR each query i IN [0 .. 487]:
    // ── Decode query ─────────────────────────────────────────
    query_text ← queries.txt[i]
    [ar_q, en_q] ← SPLIT query_text ON " | "  // (if present)

    // ── Build query fingerprint ──────────────────────────────
    // (same Steps 1→5 pipeline, but on the QUERY text instead of corpus)
    q_phrases ← extract_phrases(query_text)     // AR + EN extraction
    q_fp      ← build_fingerprint(q_phrases)      // Steps 3-5 on query

    // ── Score ─────────────────────────────────────────────────
    IF variant == "pure_sf" OR variant == "bm25":
        FOR each doc j IN [0 .. 487]:
            score[i][j] ← cosine_similarity(q_fp, doc_fp[j])

    IF variant == "splade_linear":
        // SPLADE encoding
        s_q ← splade_encode(query_text)
        s_d ← splade_encode(corpus_paragraphs)          // pre-computed
        // Linear fusion
        FOR each doc j:
            score[i][j] ← 0.3 * SF_score[i][j] + 0.7 * inner_product(s_q, s_d[j])

    IF variant == "splade_rrf":
        // RRF fusion over rankings
        sf_ranks   ← argsort(score[i][j]  DESCENDING)  // SF ranking
        spade_ranks ← argsort(splaDE_score[i][j] DESCENDING) // SPLADE ranking
        FOR each doc j:
            score[i][j] ← 1/(60 + sf_ranks[j]) + 1/(60 + spade_ranks[j])

    // ── Rank ─────────────────────────────────────────────────
    ranked[i] ← SORT(score[i], DESCENDING)
    → list of (doc_id, score) tuples

OUTPUT: all_results.json
  [ { "query": "...", "results": [("doc_000000", 0.85), ...], "metadata": {...} }, ... ]

--------------------------------------------------------------------------------
PHASE 4: METRICS COMPUTATION  (Per variant)
--------------------------------------------------------------------------------

FOR each query i IN [0 .. 487]:
    relevant ← { gold_doc_id[i] }      // set with exactly 1 element
    ranked   ← top-20 from ranked[i]

    // Reciprocal Rank
    first_rank ← position of first doc in ranked that is IN relevant
    mrr[i]     ← 1.0 / first_rank  (0.0 if not found)

    // Average Precision
    hits  ← 0
    psum  ← 0.0
    FOR rank, doc_id IN enumerate(ranked, start=1):
        IF doc_id IN relevant:
            hits  ← hits + 1
            psum  ← psum + hits / rank
    ap[i]  ← psum / |relevant|

    // Precision / Recall / NDCG at k
    FOR each k IN [1, 2, 3, 5, 20]:
        top_k    ← ranked[0:k]
        n_hits   ← COUNT(doc IN top_k WHERE doc IN relevant)
        p_at_k[i] ← n_hits / k
        r_at_k[i] ← n_hits / |relevant|
        dcg      ← SUM(1.0 / log2(rank+1) FOR rank,doc IN top_k WHERE doc IN relevant)
        ideal    ← SUM(1.0 / log2(rank+1) FOR rank IN [1..min(|relevant|,k)])
        ndcg_at_k[i] ← dcg / ideal  (0.0 if ideal == 0)

AGGREGATE across all 488 queries:
    mean_mrr, mean_ap, mean_p@k, mean_r@k, mean_ndcg@k
    min_*, max_*

OUTPUT: benchmarks/benchmark_*/pure_sf/summary.json  (and similarly for other variants)

--------------------------------------------------------------------------------
PHASE 5: REPORT GENERATION
--------------------------------------------------------------------------------

READ all variant summary.json files  (Phase 4 output)

WRITE comparison_report.md containing:
    1. Results comparison table (all metrics, all variants)
    2. Ranking summary table
    3. Key takeaways / conclusions

```

---

## 3. Block Diagram

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                MIXED ARABIC-ENGLISH BENCHMARK PIPELINE                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  PHASE 0: DATA PREPARATION                                              │
  │                                                                         │
  │  ┌───────────────────────────────────────────────┐                      │
  │  │ corpus_belebele_ar_en_deduped.csv  (488 rows) │                      │
  │  │     id │   passage   │   q1   │   q2          │                      │
  │  └────────┴─────────────┴────────┴───────────────┘                      │
  │            │                         │                                  │
  │            ▼                         ▼                                  │
  │  ┌───────────────────┐   ┌──────────────────────────┐                   │
  │  │ corpus_texts[488] │   │ queries_raw[488]         │                   │
  │  │ "Ar | En" per     │   │ "Ar | En question?"      │                   │
  │  │ bilingual passage │   │                          │                   │
  │  └────────┬──────────┘   └────────────┬─────────────┘                   │
  │           │                           │                                 │
  │           ▼                           ▼                                 │
  │  ┌────────────────────────────────────────────────────────────────┐     │
  │  │  corpus.txt (488 bilingual passages, one per line)             │     │
  │  └────────────────────────────────────────────────────────────────┘     │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  PHASE 1: QUERY GENERATION (runs once)                                 │
  │                                                                        │
  │  ┌──────────────────────────┐     ┌────────────────────────────────┐   │
  │  │  ARABIC PIPELINE         │     │  ENGLISH PIPELINE              │   │
  │  │                          │     │                                │   │
  │  │  nltk_word_tokenize()    │     │  spacy.load("en_core_web_sm")  │   │
  │  │       ↓                  │     │       ↓                        │   │
  │  │  nltk.pos_tag()          │     │  spaCy parser pipeline         │   │
  │  │       ↓                  │     │       ↓                        │   │
  │  │  ┌───────────────────┐   │     │  ┌──────────────────────────┐  │   │
  │  │  │ Unigrams:         │   │     │  │ Noun chunks              │  │   │
  │  │  │  token ≥ 2 chars  │   │     │  │ Named entities           │  │   │
  │  │  │  NOT in stop list │   │     │  │ Left modifiers           │  │   │
  │  │  └───────────────────┘   │     │  │ Compound chains          │  │   │
  │  │                          │     │  │ Conjunction expansion    │  │   │
  │  │  ┌──────────────────┐    │     │  │ Head nouns               │  │   │
  │  │  │ Bigrams:         │    │     │  └──────────────────────────┘  │   │
  │  │  │  [NN*/JJ/VBN]    │    │     │       ↓                        │   │
  │  │  │  + [N*]          │    │     │  deduplicate                   │   │
  │  │  └──────────────────┘    │     │       ↓                        │   │
  │  │                          │     │  en_phrases (list of strings)  │   │
  │  │  normalize_arabic()      │     │                                │   │
  │  │  strip_punctuation()     │     │                                │   │
  │  └───────────┬──────────────┘     └───────────┬────────────────────┘   │
  │              │                                │                        │
  │              ▼                                ▼                        │
  │  ┌───────────────────────────────────────────────────────────────┐     │
  │  │           mixed_query = ar_phrases + " " + en_phrases         │     │
  │  └─────────────────────────────┬─────────────────────────────────┘     │
  │                                │                                       │
  │                                ▼                                       │
  │                   ┌─────────────────────────┐                          │
  │                   │ queries.txt (488 lines) │                          │
  │                   └─────────────────────────┘                          │
  └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │                     
                                    ▼                     
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  PHASE 2: INDEXING (Steps 1 → 5) — runs ONCE, shared by all variants    │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │  Step 1: Phrase Extraction                                      │    │
  │  │  • Read corpus.txt line by line                                 │    │
  │  │  • For each passage: AR extraction + EN extraction              │    │
  │  │  • Build vocabulary (min_freq=1, min_wl=2, keep_verbs=True)     │    │
  │  │  • Output: vocabulary.csv, phrase_to_contexts.json              │    │
  │  └────────────────────────────┬────────────────────────────────────┘    │
  │                               ▼                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │  Step 2: Term-Context Matrix                                    │    │
  │  │  • For each passage: count phrase co-occurrences in window      │    │
  │  │  • Apply TF-IDF weighting                                       │    │
  │  │  • Output: term_context_matrix.npz + idf_weights.json           │    │
  │  └────────────────────────────┬────────────────────────────────────┘    │
  │                               ▼                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │  Step 3: Semantic Space Mapping                                 │    │
  │  │  • Input: sparse term-context matrix (N × V)                    │    │
  │  │  • UMAP: n_neighbors=15, min_dist=0.1, metric=cosine            │    │
  │  │  • Output: context_coordinates.json (N × 2)                     │    │
  │  └────────────────────────────┬────────────────────────────────────┘    │
  │                               ▼                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │  Step 4: Phrase Fingerprints                                    │    │
  │  │  • For each phrase: get (row,col) from coordinates              │    │
  │  │  • Activate cell on 64×64 binary grid                           │    │
  │  │  • Morton Z-order encoding (interleave row/col bits)            │    │
  │  │  • Gaussian blur, σ = 1.5                                       │    │
  │  │  • Output: phrase_fingerprints.npz (N_phrases × 4096)           │    │
  │  └────────────────────────────┬────────────────────────────────────┘    │
  │                               ▼                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │  Step 5: Document Fingerprints                                  │    │
  │  │  • For each passage: gather phrase fingerprints                 │    │
  │  │  • Weighted sum by TF-IDF                                       │    │
  │  │  • L2 normalize + multiply by sqrt(nnz)                         │    │
  │  │  • Output: doc_fingerprints.npz (488 × 4096)                    │    │
  │  │  • Also save: corpus.txt, query_doc_map.json, query_gold.json   │    │
  │  └─────────────────────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ╔═════════════════════════════════════════════════════════════════════════╗
  ║  PHASE 3: QUERY PROCESSING (Step 6) — runs per variant                  ║
  ╚═════════════════════════════════════════════════════════════════════════╝

  ┌────────────────────────────────────────────────────────────────────────┐
  │  For EACH query i (0..487):                                            │
  │                                                                        │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  1. Read query from queries.txt[i]                              │   │
  │  │  2. Extract phrases (same AR + EN pipeline as Phase 1)          │   │
  │  │  3. Build query fingerprint (same Steps 3-5 as Phase 2)         │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                              │                                         │
  │               ┌──────────────┼──────────────────┐                      │
  │               ▼              ▼                  ▼                      │
  │  ┌─────────────────┐ ┌────────────────┐ ┌────────────────────────┐     │
  │  │ COSINE SIM      │ │ SF+SPLADE      │ │ SF+SPLADE              │     │
  │  │ (Pure SF)       │ │ LINEAR (α=0.3) │ │ RRF (k=60)             │     │
  │  │                 │ │                │ │                        │     │
  │  │ score =         │ │ score =        │ │ score =                │     │
  │  │ cos(q_fp, d_fp) │ │ 0.3·SF         │ │ 1/(60+rank_SF)         │     │
  │  │                 │ │     + 0.7·     │ │      +                 │     │
  │  │                 │ │         SPLADE │ │ 1/(60+rank_SPLADE)     │     │
  │  │                 │ │                │ │                        │     │
  │  │ FOR EACH doc j: │ │ pre-compute    │ │ pre-compute            │     │
  │  │   score[q][j] = │ │ SPLADE vectors │ │ SPLADE vectors         │     │
  │  │                 │ │ for ALL corpus │ │ for ALL corpus         │     │
  │  └────────┬────────┘ └────────┬───────┘ └───────────┬────────────┘     │
  │           │                   │                     │                  │
  │           └───────────────────┼─────────────────────┘                  │
  │                               ▼                                        │
  │           ┌────────────────────────────────────────────┐               │
  │           │  SORT all 488 scores DESCENDING            │               │
  │           │  ranked[i] = [(doc_id, score), ...]        │               │
  │           └───────────────────┬────────────────────────┘               │
  └───────────────────────────────┼────────────────────────────────────────┘
                                  │
                                  ▼
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  PHASE 4: METRICS COMPUTATION  (per variant)                             ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FOR each variant (pure_sf, splade_linear, splade_rrf, bm25):           │
  │                                                                         │
  │  FOR each query i IN [0 .. 487]:                                        │
  │    relevant ← {gold_doc_id[i]}         (1 relevant passage)             │
  │    ranked   ← top-20 from score sort                                    │
  │                                                                         │
  │    MRR   ← 1.0 / position_of_first_relevant_in_ranked  (or 0)           │
  │    AP    ← Σ(hits_so_far / rank) / |relevant|                           │
  │                                                                         │
  │    FOR each k IN [1, 2, 3, 5, 20]:                                      │
  │      P@k   ← hits_in_top_k / k                                          │
  │      R@k   ← hits_in_top_k / |relevant|                                 │
  │      NDCG@k ← DCG@k / IDCG@k                                            │
  │                                                                         │
  │    all_metrics[i] ← {mrr, ap, p@1, p@5, r@5, ndcg@20, found_at}         │
  │                                                                         │
  │  AGGREGATE: mean, min, max across 488 queries                           │
  │                                                                         │
  │  OUTPUT: benchmarks/benchmark_*/<variant>/summary.json                  │
  │          (with all aggregated metrics)                                  │
  └─────────────────────────────────────────────────────────────────────────┘


  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  RESULTS COMPARISON TABLE                                                ║
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║                                                                          ║
  ║  ┌────────────┬────────┬────────┬────────┬────────┬────────┬───────┐     ║
  ║  │ Metric     │ Pure SF│ SF+SPLA│ SF+SPLA│ BM25   │ Winner │ Delta │     ║
  ║  │            │        │ Linear │  RRF   │        │        │vs BM25│     ║
  ║  ╞════════════╪════════╪════════╪════════╪════════╪════════╪═══════╡     ║
  ║  │ MRR        │ 0.8086 │ 0.8231 │ 0.6827 │ 0.7854 │ Linear │ +4.8% │     ║
  ║  │ AP         │ 0.8086 │ 0.8231 │ 0.6827 │ 0.7854 │ Linear │ +4.8% │     ║
  ║  │ P@1        │ 0.7500 │ 0.7705 │ 0.5861 │ 0.7193 │ Linear │ +7.1% │     ║
  ║  │ P@5        │ 0.1746 │ 0.1770 │ 0.1668 │ 0.1738 │ Linear │ +1.8% │     ║
  ║  │ R@5        │ 0.8730 │ 0.8852 │ 0.8340 │ 0.8689 │ Linear │ +1.9% │     ║
  ║  │ NDCG@20    │ 0.8407 │ 0.8470 │ 0.7426 │ 0.8233 │ Linear │ +2.9% │     ║
  ║  └────────────┴────────┴────────┴────────┴────────┴────────┴───────┘     ║
  ║                                                                          ║
  ║  Rankings:  1) SF+SPLADE Linear  2) Pure SF  3) BM25  4) SF+SPLADE RRF   ║
  ║                                                                          ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  PHASE 5: REPORT GENERATION                                             │
  │                                                                         │
  │  comparison_report.md  ← tables + rankings + key takeaways              │
  │  summary.json per variant  ← raw metrics per query                      │
  │  results_log.csv  ← per-query detail (spread, elapsed, etc.)            │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Design Decisions

### 4.1 Self-Retrieval Task
Each query's gold passage is the passage from which it was derived. The model must find its own source passage among all 488 candidates. This is the simplest ground truth — a binary relevant/not relevant per query — but it directly measures the retrieval accuracy of the pipeline.

### 4.2 Mixed-Language Query Construction
Queries are constructed by extracting key phrases from both the Arabic and English parts of each bilingual question, then concatenating them (Arabic first, English second). This forces the retrieval system to handle queries that span two languages simultaneously, testing cross-lingual semantic understanding.

### 4.3 Full Corpus Retrieval
All 488 passages are candidates for every query (no distractor filtering). This tests the full retrieval strength of SF and provides a realistic measure of rank quality — the correct passage must compete against all other 487 passages.

### 4.4 Shared Indexing Pipeline
Phases 1 and 2 (query generation and indexing) run ONCE and produce fingerprints that are reused across all four variants. Only Phase 3 (retrieval scoring) differs per variant, ensuring a fair comparison.

### 4.5 SPLADE Fusion Methods
Two fusion strategies are evaluated:
- **Linear** (α=0.3): Weighted average of SF score and SPLADE inner-product score
- **RRF** (k=60): Rank Reciprocal Fusion combining reciprocal ranks from both methods

These are also evaluated on the cross-lingual Arabic→English benchmark from the session's earlier work, providing a multi-dimensional view of SPLADE's effectiveness.

---

## 5. File Locations

### Input
| File | Description |
|------|-------------|
| `data/datasets/belebele/raw/all/corpus_belebele_ar_en_deduped.csv` | Deduplicated bilingual corpus (488 rows) |

### Phase 1 Output
| File | Description |
|------|-------------|
| `outputs/mixed_ar_en_benchmark/runs/run_*/queries.txt` | 488 mixed-language queries |

### Phase 2 Output (Indexing Artifacts)
| Path | Description |
|------|-------------|
| `corpus.txt` | Combined bilingual corpus (488 passages) |
| `extracted_phrases/` | Vocabulary + phrase-context mappings |
| `term_context_matrix/` | TF-IDF weighted term-context matrix |
| `semantic_space/` | UMAP 2D coordinates (N × 2) |
| `phrase_fingerprints/` | Binary grid fingerprints per phrase |
| `doc_fingerprints/` | Combined document fingerprint vectors |
| `query_doc_map.json` | Query → all 488 doc IDs |
| `query_gold.json` | Query → gold doc ID |

### Phase 3/4/5 Output
| Path | Description |
|------|-------------|
| `benchmarks/benchmark_*/pure_sf/` | Pure SF results |
| `benchmarks/benchmark_*/splade_linear/` | SF+SPLADE Linear results |
| `benchmarks/benchmark_*/splade_rrf/` | SF+SPLADE RRF results |
| `benchmarks/benchmark_*/bm25/` | BM25 results |
| `benchmarks/benchmark_*/comparison_report.md` | Comparison table |
| `PIPELINE_PSEUDOCODE_BLOCKDIAGRAM.md` | This document |

---

## 6. Conclusion

Semantic Folding successfully extracts and matches mixed-language concepts from bilingual Arabic-English text. With an MRR of **0.81** on 488-candidate retrieval, the pipeline demonstrates strong cross-lingual semantic understanding through its 2D semantic grid representation. Linear SPLADE fusion provides a consistent +1.8% MRR gain, while BM25 remains a competitive baseline. The RRF fusion method underperforms significantly on this bilingual task compared to linear fusion, suggesting that score-based fusion is more effective than rank-based fusion for mixed-language retrieval.
