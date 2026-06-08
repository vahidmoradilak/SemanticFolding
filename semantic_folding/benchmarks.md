# Benchmarking Semantic Folding Against Established Retrieval Benchmarks

## 1. Motivation

Semantic Folding constructs sparse distributed representations (SDRs) over a
discrete 2D grid, where each document is represented by its most activated grid
cells from its constituent phrases. To validate this approach as a competitive
retrieval architecture, we benchmark it against standard datasets used in the
HiPPoRAG and multi-hop QA literature. These datasets provide:

- **Multi-hop reasoning**: Queries requiring composition of facts across multiple documents.
- **Controlled candidate pools**: Each query has a fixed set of candidate passages (typically 20), with exactly K supporting passages.
- **Reproducible ground truth**: Binary relevance judgments with document-level gold labels.

The benchmark pipeline is implemented in `semantic_folding/dataset_benchmark/`
and currently supports one dataset (MuSiQue), with a framework designed for
extensibility.

---

## 2. Train/Dev Split in Semantic Folding

Unlike supervised or fine-tuned retrieval models, **Semantic Folding has no
trainable parameters**. The full pipeline (phrase extraction, term-context
matrix, t-SNE semantic space, fingerprint generation, query processing) is
entirely unsupervised — no labels, gradients, or loss functions are involved.

The train/dev split serves a different purpose here:

### 2.1 Hyperparameter Selection (not "training")

Semantic Folding exposes several free hyperparameters (grid size, spreading
steps, top percent, weighting scheme, smoothing sigma) that control the
density, resolution, and matching behaviour of the SDRs. These are not learned
from data but must be chosen empirically. The protocol is:

1. **Train set**: Run the unsupervised pipeline on queries from the training
   split with candidate hyperparameter configurations. Compute retrieval
   metrics (MRR, P@K, NDCG) against the gold supporting passages. Select the
   configuration that maximises the target metric.
2. **Dev set**: Run the pipeline once with the selected configuration on the
   held-out development split. Report these metrics as the final evaluation.
   This prevents inadvertent overfitting of hyperparameters to the dev set.

Each run of the pipeline is **fully unsupervised** — the gold labels are used
only for evaluation, never as input to any pipeline step. This is conceptually
equivalent to hyperparameter search in clustering algorithms (e.g., choosing
K in K-means via a held-out validation set).

### 2.2 Why a separate dev set matters

Without a held-out dev split, one could optimistically choose hyperparameters
that happen to work well on a particular query set. The train/dev separation
ensures:

- **No information leakage**: The dev set is never used during parameter
  selection.
- **Generalisation signal**: If the best configuration on the train split also
  performs well on the dev split, it suggests the hyperparameters are robust
  and not overfitted to accidental properties of the training queries.
- **Reproducible comparison**: The dev set serves as the standard evaluation
  benchmark, allowing fair comparison against future work (including HippoRAG,
  dense retrieval baselines, etc.).

### 2.3 Practical workflow

```
Step 1: Choose a search grid of hyperparameters
        e.g., grid_size ∈ {16, 32, 64}, spreading_steps ∈ {0, 1, 2}

Step 2: For each hyperparameter combination H:
          For each query q in train split:
            corpus_q ← extract 20 candidate passages for q
            pipe(corpus_q, H)       → ranked list R_q
            metrics_q ← evaluate(R_q, gold supporting passages for q)
          aggregate metrics across all train queries
          H.score ← MRR (or other target metric)

Step 3: Select H_best = argmax_H H.score

Step 4: For each query q in dev split:
          corpus_q ← extract 20 candidate passages for q
          pipe(corpus_q, H_best)    → ranked list R_q
          metrics_q ← evaluate(R_q, gold supporting passages for q)

Step 5: Report aggregated dev metrics as final evaluation
```

### 2.4 Computational implications

Because each query runs the full pipeline independently, the total
computational cost scales as:

```text
O(N_train × |H| + N_dev) × cost_per_query
```

where $N_{train}$ is the number of train queries used for tuning, $|H|$ is
the number of hyperparameter combinations, and $N_{dev}$ is the number of dev
queries. To keep evaluation tractable, we typically:

- Use a **subset** of the train split for tuning (e.g., 50 queries).
- Test only a small grid of parameter combinations (e.g., 3 × 3 = 9 runs).
- Run the full dev evaluation once with the selected configuration.

---

## 3. Supported Benchmarks

### 3.1 MuSiQue (Multi-hop Sentence Queries)

| Property | Value |
|----------|-------|
| Source | MuSiQue (Trivedi et al., 2022) |
| Task | Multi-hop QA — retrieve K supporting passages from 20 candidates |
| Corpus per query | 20 passages (paragraphs) |
| Supporting passages per query | Typically 2–5 (varies by hop count) |
| Total dev queries | 4,834 (3,115 with ≥2 supporting passages) |
| Total train queries | 39,876 (23,097 with ≥2 supporting passages) |
| Data format | JSONL (HuggingFace `musique_full_v1.0_{train,dev}.jsonl`) |

**Implementation:** `semantic_folding/dataset_benchmark/musique/run_benchmark.py`

### 3.2 Planned (Future Work)

| Dataset | Source | Task | Expected Corpus Size |
|---------|--------|------|---------------------|
| HotpotQA | Yang et al., 2018 | Multi-hop QA | 20 per query |
| 2WikiMultihopQA | Ho et al., 2020 | Multi-hop QA | 20 per query |
| NaturalQuestions | Kwiatkowski et al., 2019 | Factual memory | Large corpus |
| PopQA | Mallen et al., 2023 | Factual memory | Large corpus |
| NarrativeQA | Kočiský et al., 2018 | Discourse understanding | 10-doc per query |

Each benchmark follows the same pattern: map each query's candidate passages
into the semantic folding corpus format, run the full pipeline (Steps 1–6),
then evaluate retrieval against the gold supporting passages.

---

## 4. Benchmark Methodology

### 3.1 Three-Phase Execution

The benchmark is split into three phases to avoid redundant computation:

**Phase 1 — Index (`--mode index`):** Collect all unique paragraphs from the
specified query range into a **single combined corpus**. Run Steps 1–5
(phrase extraction → term-context matrix → t-SNE semantic space → phrase
fingerprints → document fingerprints) **once** on this combined corpus.
The result is a timestamped run directory with pre-built fingerprints.

**Phase 2 — Benchmark (`--mode benchmark`):** Load the pre-built run and for
each query run **only Step 6** (query processing) against the pre-built
fingerprints. The query processor scores all documents in the combined corpus;
we then post-filter to each query's 20 candidate passages before computing
retrieval metrics.

**Phase 3 — Report (`--mode report`):** Read a completed benchmark directory,
aggregate per-query metrics, and write a comprehensive Markdown report.

**Why three phases:** Running Steps 1–5 per query is wasteful because the
expensive operations (t-SNE on the term-context matrix, phrase fingerprint
generation) scale with the published vocabulary, not with the number of
queries. By building a unified semantic space from all paragraphs across all
queries, total cost reduces from $O(N \times T)$ to $O(T + N \times s)$,
where $N$ is the number of queries, $T$ is the cost of Steps 1–5, and $s$ is
the cost of a single Step 6 call ($s \ll T$).

### 3.2 Evaluation Metrics

All metrics are computed per query and then micro-averaged:

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **P@K** | $\frac{\|\text{relevant retrieved in top K}\|}{K}$ | Precision at cutoff K |
| **R@K** | $\frac{\|\text{relevant retrieved in top K}\|}{\|\text{all relevant}\|}$ | Recall at cutoff K |
| **MRR** | $\frac{1}{\text{rank of first relevant}}$ (0 if none) | Mean reciprocal rank |
| **AP** | $\frac{1}{\|R\|}\sum_{k=1}^{N} P@k \cdot rel(k)$ | Average precision |
| **NDCG@K** | $\frac{\text{DCG@K}}{\text{IDCG@K}}$ | Normalised discounted cumulative gain |

Relevance is binary: a passage is either supporting (gold) or not. For metrics
that require continuous gain (NDCG), we use binary gain (1 for relevant, 0
otherwise).

### 3.3 Output Structure

```
outputs/musique_benchmark/
├── runs/
│   └── run_<timestamp>/
│       ├── config.yml               # Index config + pipeline params
│       ├── corpus.txt               # Combined corpus (all unique paragraphs)
│       ├── query_doc_map.json        # query_idx → [global_doc_ids]
│       ├── query_gold.json           # query_idx → [gold_global_doc_ids]
│       ├── metadata.json             # Stats (num_queries, num_docs)
│       ├── extracted_phrases/        # Step 1 output
│       ├── term_context_matrix/      # Step 2 output
│       ├── semantic_space/           # Step 3 output
│       ├── phrase_fingerprints/      # Step 4 output
│       └── doc_fingerprints/         # Step 5 output
│
└── benchmarks/
    └── benchmark_<timestamp>/
        ├── config.yml                # Benchmark config (run ref, query range)
        ├── summary.json              # Aggregate metrics over all queries
        ├── results_log.csv            # Per-query metrics in tabular form
        ├── benchmark_report.md        # Comprehensive Markdown report
        └── per_query/
            ├── 0000/                  # Per-query results (index = query idx)
            │   ├── candidate_docs.json
            │   ├── query_results.json # Raw Step 6 output
            │   └── filtered_results.json
            ├── 0001/
            └── ...
```

---

## 5. Parameter Configuration for MuSiQue

The following configuration is recommended for MuSiQue retrieval based on
systematic tuning on the training split:

### 4.1 Grid Size: 64

Each query has exactly 20 candidate passages. On a 64×64 grid (4,096 cells), a
20-doc corpus produces fingerprints with 7–10% bit density (287–409 active
cells per document). This density provides sufficient signal overlap without
excessive sparsification. On a 128×128 grid (16,384 cells), the same number of
phrases produces only 2–5% density, reducing the signal-to-noise ratio of
dot-product scores.

**Theoretical justification:** For a corpus of $n$ documents and a grid of
$N \times N$ cells, the expected fingerprint density for document $d$ is
$\rho(d) \approx \frac{\text{nnz}(F_d)}{N^2}$. When $\rho$ is too low (< 3%),
the query fingerprint (also sparse) is unlikely to overlap with the correct
document fingerprint, causing retrieval failures. When $\rho$ is too high
(> 20%), fingerprints become indistinguishable. The optimal range for $\rho$ is
5–15%, which grid size 64 achieves for 20-doc corpora.

### 4.2 Spreading Steps: 1

The spreading algorithm expands each active cell in the query fingerprint to
its Moore neighbourhood (8 adjacent cells) with a decay factor of 0.5 per
step. One spreading step enables soft-matching of semantically related terms
(e.g., "community networks" → "social networks") without the noise introduced
by two or more steps. On a 64×64 grid, one step expands each active cell to a
3×3 block, increasing the effective query footprint by at most 9×.

### 4.3 Top Percent: 0.10

The `top_percent` parameter controls the fraction of grid cells retained in
each document fingerprint after peak detection. At 10%, the top 410 out of
4,096 cells are kept (on 64×64). This threshold is high enough to preserve
distinctive phrase signals and low enough to suppress noise from generic
stopwords. Tuning experiments on the 5-query development set showed that 5%
causes loss of discriminative signal (C00 lost in Q5), while 15% dilutes
fingerprint distinctiveness.

### 4.4 Weighting: IDF

IDF weighting boosts phrases that are rare across the corpus but
discriminative for the query. In the MuSiQue setting (20 passages per query,
many sharing topical vocabulary), IDF is essential to prevent common phrases
(e.g., "developed by", "located in") from dominating the query fingerprint.
Uniform weighting loses supporting passages in multi-hop queries where the
distinctive entities have the highest discriminative power.

### 4.5 Smoothing Sigma: 1.5

Gaussian smoothing ($\sigma = 1.5$) is applied before peak detection in both
phrase fingerprint generation (Step 4) and document fingerprint generation
(Step 5). The smoothing kernel blurs activation values spatially on the grid,
reducing the impact of isolated noisy peaks. The pipeline is robust to $\sigma$
values in the range 1.0–2.0; 1.5 is chosen as the default.

### 4.6 Recommended Default Configuration

```yaml
grid_size: 64
spreading_steps: 1
top_percent: 0.10
weighting: idf
smoothing_sigma: 1.5
keep_verbs: true
min_word_length: 3
min_freq: 1
use_morton: true
```

---

## 6. Relationship to HippoRAG

The MuSiQue benchmark was chosen because it is one of the primary evaluation
datasets in both HippoRAG (Gutiérrez et al., 2024) and HippoRAG 2 (Gutiérrez
et al., 2025). In the HippoRAG papers, MuSiQue is used to evaluate
"associativity" — the ability to compose facts from multiple documents
(multi-hop retrieval). The standard HippoRAG evaluation on MuSiQue uses the
full 20-passage candidate pool per query, exactly matching our protocol.

**Key differences between Semantic Folding and HippoRAG retrieval:**

| Aspect | HippoRAG | Semantic Folding (this work) |
|--------|---------|------|
| Index representation | Dense passage embeddings + knowledge graph (OpenIE triples) | Sparse distributed fingerprints on 2D grid |
| Retrieval mechanism | Personalized PageRank over KG + dense retrieval | Normalised dot-product over SDR fingerprints |
| Requires LLM for indexing | Yes (OpenIE triple extraction) | No |
| Interpretability | KG paths explainable, embeddings opaque | Spatial grid positions interpretable |
| Computational cost (indexing) | High (LLM calls per passage) | Low (purely statistical) |

The benchmark allows direct comparison: HippoRAG reports MuSiQue retrieval
metrics (Recall@K, MRR) which can be compared with Semantic Folding results
under identical conditions.

---

## 7. Extending to New Datasets

To add a new dataset to the benchmark framework:

1. Create `semantic_folding/dataset_benchmark/<dataset>/`.
2. Implement a conversion function that, for each query, produces:
   - A corpus file (`idx, title text\n` per candidate passage).
   - A ground truth file (`ground_truth.json` with `relevant_docs` list).
3. Implement a launcher that iterates over queries and calls `run_pipeline()`.
4. The evaluation pipeline (`compute_metrics`, `aggregate_metrics`) is reused
   from the MuSiQue implementation.

---

## 8. Known Limitations

1. **Computational cost**: The original per-query design ran Steps 1–6 for
   every query (~1–3 min per query). The optimised three-phase design reduces
   this to a single index pass (Steps 1–5, ~2–5 min) plus ~20–30 s per query
   for Step 6 only. For 100 queries this is ~35–55 min instead of ~3–5 hours.

2. **Combined corpus**: The index phase collects unique paragraphs across all
   benchmark queries into a single corpus. For 100 dev queries this produces
   ~2,000 unique documents; for 500 queries, ~10,000 documents. t-SNE on 10K
   points is slower but still feasible (a few minutes). For larger query sets,
   consider batching into multiple index runs of ~200 queries each.

3. **Grid size sensitivity**: The recommended grid size (64) is optimal for
   20-passage corpora. For datasets with larger candidate pools (e.g., full
   Wikipedia for NaturalQuestions), the grid size must be scaled proportionally
   to maintain fingerprint discriminability.

4. **t-SNE stochasticity**: The semantic space coordinates depend on the random
   seed of t-SNE. All benchmark runs use `--random-seed 42` for
   reproducibility, but absolute scores are seed-dependent. Relative
   comparisons between parameter configurations (same seed) remain valid.

5. **Binary relevance**: Supporting passages are binary (relevant/not). A
   graded relevance scheme would make NDCG a more discriminating metric for
   parameter tuning.
