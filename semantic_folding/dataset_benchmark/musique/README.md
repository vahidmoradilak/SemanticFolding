# MuSiQue Benchmark for Semantic Folding

## Overview

This module evaluates the Semantic Folding pipeline against the **MuSiQue**
(Multi-hop Sentence Queries) dataset. MuSiQue is a multi-hop QA dataset where
each query requires composing facts from 2–5 supporting passages drawn from a
candidate pool of 20 passages. The retrieval task is: given a query, rank the
20 passages such that the supporting (gold) passages appear at the top.

**Source:** Trivedi et al., "MuSiQue: Multi-hop Sentence Queries", ACL 2022.
**Original data:** `data/HippoRAG2/dataset/musique/` (HuggingFace format).

---

## Train/Dev Split in Semantic Folding

Semantic Folding is a fully **unsupervised** pipeline — no labels, gradients,
or loss functions are involved at any step (phrase extraction, term-context
matrix, t-SNE, fingerprinting, query processing). The train/dev split is used
exclusively for **hyperparameter selection**, not for model training.

### How it works

| Step | Split | Purpose |
|------|-------|---------|
| Tuning | Train (e.g., 50 queries) | Run the pipeline with candidate hyperparameters (grid_size, spreading_steps, etc.), evaluate against gold labels, select the best configuration. |
| Evaluation | Dev (all queries) | Run the pipeline once with the selected configuration. Report these metrics as the final result. |

The gold labels are used **only for evaluation** — never as input to any
pipeline step. This is analogous to choosing K in K-means clustering via a
validation set: the algorithm itself is unsupervised, but a held-out set
prevents overfitting of hyperparameters to the evaluation queries.

### Why it matters

Without a held-out dev split, one could optimistically choose hyperparameters
that happen to work well on a particular query set. The train/dev separation
ensures:

1. **No information leakage** — the dev queries are never used during parameter
   selection.
2. **Generalisation signal** — if the best config on train also performs well on
   dev, the hyperparameters are robust.
3. **Fair comparison** — the dev set serves as the standard benchmark for
   comparing against future work (HippoRAG, dense retrieval baselines, etc.).

### Practical workflow

```
For each hyperparameter combination H:
  For each query q in train subset:
    run pipeline(corpus_q, H) → ranked list R_q
    metrics_q ← evaluate(R_q, gold for q)
  aggregate → H.score

Select H_best = argmax MRR (or other target metric)

For each query q in dev set:
  run pipeline(corpus_q, H_best) → ranked list R_q
  metrics_q ← evaluate(R_q, gold for q)

Report aggregated dev metrics
```

---

## Files

| File | Purpose |
|------|---------|
| `run_benchmark.py` | Main entry point — interactive TUI or CLI (`--mode`). Three phases: index, benchmark, report. |
| `benchmark_analyzer.py` | Deep-dive analysis of completed benchmark results (distributions, failures, top performers). |
| `runs/registry.yml` | Run registry for resume capability — tracks all index & benchmark runs with status and params. |
| `__init__.py` | Package marker |

---

## Interactive TUI (Default)

Run the script **without arguments** to enter the interactive TUI:

```powershell
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py
```

The TUI presents a colored menu with these options:

1. **Phase 1: Index Corpus** — Guides you through parameter entry with auto-generated defaults. Prompts for split, query count, grid size, spreading steps, etc. You can accept defaults or override any value.
2. **Phase 2: Benchmark** — Lets you select a pre-built index run from the registry, then runs Step 6 per query. Parameters are auto-loaded from the run's config.yml.
3. **Phase 3: Generate Report** — Select a completed benchmark and regenerate the Markdown report.
4. **Analyze Last Results** — Runs `benchmark_analyzer.py` on the most recent benchmark for deep-dive metrics.
5. **Resume / Re-run** — Shows interrupted runs (failed at a specific step) and completed benchmarks for re-report generation.
6. **Exit**

After completing Phase 1, you're asked if you want to proceed immediately to Phase 2. Similarly after Phase 2 you're offered Phase 3.

All parameters are auto-generated from `PIPELINE_DEFAULTS` (matching the established tuning: grid=64, spread=1, top% = 0.10, weighting=idf, etc.) but every value can be overridden.

---

## CLI Mode (Non-Interactive)

For automation and scripting, use `--mode`:

## Dataset Structure

The downloaded MuSiQue dataset consists of four JSONL files:

| File | Split | Entries | With ≥2 supporting |
|------|-------|---------|--------------------|
| `musique_full_v1.0_train.jsonl` | Train | 39,876 | 23,097 |
| `musique_full_v1.0_dev.jsonl` | Dev | 4,834 | 3,115 |
| `musique_ans_v1.0_train.jsonl` | Train (answerable only) | ~20K | Subset |
| `musique_ans_v1.0_dev.jsonl` | Dev (answerable only) | 2,417 | Subset |

Each JSONL line has this structure:

```
{
  "id": "2hop__42543_20093",
  "question": "What year did the writer of Crazy Little Thing Called Love die?",
  "answer": "1991",
  "paragraphs": [
    {
      "idx": 0,
      "title": "All Things in Time",
      "paragraph_text": "album by American R&B singer Lou Rawls...",
      "is_supporting": false
    },
    ...
  ],
  "question_decomposition": [
    {"id": 42543, "question": "who wrote crazy little thing called love", "answer": "Freddie Mercury"},
    {"id": 20093, "question": "In what year did #1 die?", "answer": "1991"}
  ]
}
```

**Key observations:**
- Each entry has exactly 20 paragraphs (indices 0–19).
- Exactly 2–5 paragraphs have `is_supporting: true` (depending on the number of hops).
- The supporting paragraphs are the gold standard for retrieval.
- ~20% of entries have zero supporting paragraphs (unanswerable questions);
  these are filtered out by the benchmark.

---

## Three-Phase Design

The benchmark is split into three phases to avoid redundant computation:

### Phase 1: Index (`--mode index`)

Collects all unique paragraphs from the specified query range into a single
combined corpus, then runs Steps 1–5 of the Semantic Folding pipeline **once**.
The result is a timestamped run directory under `outputs/musique_benchmark/runs/`.

```
Steps 1–5 (once)
    ↓
phrase fingerprints + doc fingerprints + IDF weights
    ↓
saved to runs/run_<timestamp>/
```

**Why:** The expensive operations (t-SNE on the term-context matrix, phrase
fingerprint generation) scale with the published vocabulary, not with the
number of queries. By building a unified semantic space from all paragraphs
across all queries in the benchmark, we avoid re-running these steps per
query.

### Phase 2: Benchmark (`--mode benchmark`)

Loads a pre-built run (fingerprints, IDF weights) and for each query runs
**only Step 6** (query_processing.py). The query processor scores all
documents in the combined corpus; we then post-filter to each query's 20
candidate passages before computing retrieval metrics.

```
run_<timestamp>/  (pre-built fingerprints)
    ↓
for each query q:
    step_6(q) → scores for ALL docs in corpus
    filter to q's 20 candidates
    evaluate against gold
    ↓
saved to benchmarks/benchmark_<timestamp>/per_query/<query_idx>/
```

This reduces total computation from $O(N \times T)$ to $O(T + N \times s)$,
where $N$ is the number of queries, $T$ is the cost of Steps 1–5, and $s$ is
the cost of a single Step 6 run ($s \ll T$).

### Phase 3: Report (`--mode report`)

Reads a completed benchmark directory, aggregates all per-query metrics, and
writes a comprehensive Markdown report (`benchmark_report.md`) with:

- Configuration snapshot (pipeline parameters, query range)
- Aggregate metrics (mean/min/max MRR, AP, P@K, R@K, NDCG@K)
- Per-query results table
- Distribution analysis (found-at-rank counts)

---

## Semantic Folding Pipeline Mapping

Each MuSiQue entry is converted to the Semantic Folding corpus format as
follows:

```
corpus.txt format (1 line per candidate passage):
  0, Title of passage 0 text of passage 0...
  1, Title of passage 1 text of passage 1...
  ...
 19, Title of passage 19 text of passage 19...
```

The ground truth is stored separately:

```json
{"query_id": "2hop__42543_20093", "relevant_docs": [5, 17]}
```

The full 6-step pipeline is then run on this 20-document corpus, and the
retrieved ranked list is compared against `relevant_docs`.

---

## Parameter Details

Recommended parameters (established through systematic tuning):

### `--grid-size` (default: 64)

The side length of the $N \times N$ semantic grid.

- **64×64 (4,096 cells)** — optimal for 20-doc MuSiQue passages. Produces
  fingerprints with ~7–10% bit density, balancing signal overlap and
  discriminability.
- **16×16 (256 cells)** — fast prototyping but low resolution causes semantic
  collisions (~40% density).
- **32×32 (1,024 cells)** — intermediate quality (~15% density).
- **128×128 (16,384 cells)** — excessive sparsification (~2–3% density) reduces
  signal overlap for small corpora.

**Theoretical basis:** The expected fingerprint density for $n$ documents on an
$N \times N$ grid is $\rho \approx \frac{\text{nnz}(F_d)}{N^2}$. The optimal
range for $\rho$ is 5–15%, which grid 64 achieves for 20-doc corpora.

### `--spreading-steps` (default: 1)

Number of Moore-neighbourhood expansion steps on the query fingerprint.

- **0** — exact cell matching only. Loses soft semantic matches (e.g.,
  "community networks" cannot reach "social networks").
- **1** — expands each active cell to its 8 neighbours with decay 0.5.
  Enables soft matching without excessive noise. **Recommended.**
- **2** — expands further. Does not improve retrieval for 20-doc corpora;
  adds noise from distant neighbours.

### `--top-percent` (default: 0.10)

Fraction of grid cells retained in each document fingerprint.

- **0.05** — too sparse; risks losing discriminative phrase signals.
- **0.10** — balanced precision–recall. Retains ~410 out of 4,096 cells on
  grid 64. **Recommended.**
- **0.15** — too dense; fingerprint overlap diluted by irrelevant cells.

### `--weighting` (default: idf)

Phrase weighting strategy for query fingerprint aggregation.

- **`idf`** — boosts phrases with high inverse document frequency. Essential
  for MuSiQue where multiple passages share topical vocabulary. **Recommended.**
- **`uniform`** — all phrases weighted equally. Loses discriminative signal in
  multi-hop queries.
- **`frequency`** — weights by term frequency. Not evaluated on MuSiQue.

### `--smoothing-sigma` (default: 1.5)

Gaussian blur sigma applied before peak detection.

- Pipeline is robust to 1.0–2.0. Value 1.5 is a safe default.

### Other parameters held constant

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--keep-verbs` | True | Verbs carry semantic signal in multi-hop queries |
| `--min-word-length` | 3 | Filters trivial tokens ("it", "to", "is") |
| `--min-freq` | 1 | All phrases kept (20-doc corpus, low frequency counts) |
| Morton encoding | True | Preserves spatial locality in fingerprint indexing |
| t-SNE metric | cosine | Standard for high-dimensional semantic spaces |
| t-SNE perplexity | 30 | Balanced local/global structure for 20 points |
| t-SNE iterations | 1000 | Sufficient for 20-point embedding convergence |

---

## Usage

### Prerequisites

- Python virtual environment activated (`.venv\scripts\activate`)
- All dependencies installed (`numpy`, `scipy`, `spacy`, `scikit-learn`, `pyyaml`)
- spaCy model downloaded (`python -m spacy download en_core_web_sm`)
- MuSiQue dataset present in `data/HippoRAG2/dataset/musique/`

### Phase 1: Index the Corpus

Build a combined corpus from the first N queries and run Steps 1–5 once:

```powershell
# Index first 100 dev queries (recommended params)
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split dev --max-queries 100 --grid-size 64 --spreading-steps 1 `
    --top-percent 0.10 --weighting idf --smoothing-sigma 1.5
```

Creates: `outputs/musique_benchmark/runs/run_<timestamp>/`

If you want to index a different subset, use `--query-start` and `--query-end`:

```powershell
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split dev --query-start 100 --query-end 200 --grid-size 64
```

### Phase 2: Benchmark Queries

Run Step 6 only for queries 0–49 against a pre-built run:

```powershell
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode benchmark --split dev `
    --run-dir outputs/musique_benchmark/runs/run_<timestamp>/ `
    --query-start 0 --query-end 50
```

Creates: `outputs/musique_benchmark/benchmarks/benchmark_<timestamp>/`
with per-query results in `per_query/<query_idx>/`.

### One-Command Index + Benchmark

Run Phase 1 and Phase 2 together:

```powershell
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split dev --max-queries 100 --grid-size 64 --spreading-steps 1 `
    --top-percent 0.10 --weighting idf --smoothing-sigma 1.5 --benchmark
```

This creates the index, runs the benchmark on the same query range, and
generates the report — all in one command.

### Phase 3: Generate Report

Re-generate a comprehensive Markdown report from a completed benchmark:

```powershell
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode report --benchmark-dir outputs/musique_benchmark/benchmarks/benchmark_<timestamp>/
```

### Parameter Tuning Workflow

```powershell
# Step 1: Index a training subset (50 queries) with different grid sizes
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split train --max-queries 50 --grid-size 16 --benchmark
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split train --max-queries 50 --grid-size 32 --benchmark
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split train --max-queries 50 --grid-size 64 --benchmark

# Step 2: Compare spreading steps on the best grid size
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split train --max-queries 50 --grid-size 64 --spreading-steps 0 --benchmark
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split train --max-queries 50 --grid-size 64 --spreading-steps 2 --benchmark

# Step 3: Evaluate best config on the full dev set
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split dev --max-queries 100 --grid-size 64 --spreading-steps 1 --benchmark
```

### Benchmark Analyzer

For a deep-dive analysis of any completed benchmark:

```powershell
# Interactive — picks the most recent benchmark
.venv\scripts\python semantic_folding\dataset_benchmark\musique\benchmark_analyzer.py

# Specific benchmark
.venv\scripts\python semantic_folding\dataset_benchmark\musique\benchmark_analyzer.py `
    --benchmark-dir outputs/musique_benchmark/benchmarks/benchmark_<timestamp>/
```

Produces:
- Per-metric histograms (MRR, AP, P@1, P@2) with mean, median, std, zero/perfect counts
- Found-at-rank distribution (visual bar chart)
- Failure analysis (queries where no gold passage was found)
- Top performers (queries with MRR=1.0)
- JSON output saved to `analysis.json` in the benchmark directory

---

### Run Registry

The benchmark maintains a registry at `semantic_folding/dataset_benchmark/musique/runs/registry.yml` that tracks all index and benchmark runs. This enables:

- **Resume detection**: When you run the TUI, it shows interrupted runs (failed at Step N) so you can investigate.
- **Re-reporting**: Select a completed benchmark to regenerate its report.
- **Status tracking**: Each run records type (index/benchmark), status (completed/failed_stepN/running), params, and path.

The actual data artifacts remain in `outputs/musique_benchmark/` — the registry is a lightweight index for the TUI.

---

### Batch Processing for Large Sets

```powershell
# Index first 500 dev queries (once)
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode index --split dev --max-queries 500 --grid-size 64 --spreading-steps 1

# Benchmark in batches of 100
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode benchmark --split dev --run-dir outputs/musique_benchmark/runs/run_<ts>/ `
    --query-start 0 --query-end 100
.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py `
    --mode benchmark --split dev --run-dir outputs/musique_benchmark/runs/run_<ts>/ `
    --query-start 100 --query-end 200
# ...
```

---

## Output Format

### `benchmark_summary.json`

```json
{
  "mean_mrr": 0.8500,
  "mean_ap": 0.6125,
  "mean_p@1": 0.7500,
  "mean_p@2": 0.6250,
  "mean_r@2": 0.4375,
  "mean_ndcg@2": 0.5625,
  "num_queries": 100,
  "skipped": 0,
  "failed": 2,
  "grid_size": 64,
  "spreading_steps": 1,
  "top_percent": 0.10,
  "weighting": "idf",
  "smoothing_sigma": 1.5
}
```

### `analysis.json` (generated by `benchmark_analyzer.py`)

```json
{
  "benchmark": "benchmark_20260520_130000",
  "metrics_distribution": {
    "mrr": {"mean": 0.85, "median": 1.0, "min": 0.0, "max": 1.0, "std": 0.28, "num_zero": 5, "num_perfect": 60},
    "ap":    {"mean": 0.61, "median": 0.67, "min": 0.0, "max": 1.0, "std": 0.31, "num_zero": 5, "num_perfect": 10}
  },
  "found_at_distribution": {"0": 5, "1": 60, "2": 15, "3": 10, "4": 5, "5": 5},
  "failures": [{"query_idx": 12, "query": "What year...", "num_gold": 2, "num_candidates": 20}],
  "top_performers": [{"query_idx": 3, "query": "Who...", "mrr": 1.0, "ap": 1.0}]
}
```

### `results_log.csv`

| query_id | query | mrr | ap | p@1 | p@2 | p@3 | p@5 | r@2 | ndcg@2 | found_at | elapsed_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2hop__... | What year did... | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 0.6667 | 0.4000 | 0.6667 | 0.6667 | 1 | 46.0 |

---

## Output Structure

After running the full benchmark (index + benchmark + report), the output directory
tree looks like this:

```
outputs/musique_benchmark/
├── runs/
│   └── run_<timestamp>/                # Phase 1: Index run
│       ├── config.yml                  # Index config + pipeline params
│       ├── corpus.txt                  # Combined corpus (doc_000000, title text...)
│       ├── metadata.json               # Stats: num_queries, num_docs, query range
│       ├── query_doc_map.json          # query_idx -> [doc_000000, doc_000001, ...]
│       ├── query_gold.json             # query_idx -> [gold doc IDs]
│       ├── extracted_phrases/          # Step 1 output
│       │   ├── vocabulary.csv          # Phrase -> frequency
│       │   └── phrase_to_contexts.json # Phrase -> context vector
│       ├── term_context_matrix/        # Step 2 output
│       │   ├── term_context_matrix.npz # Dense term-context matrix
│       │   ├── term_context_matrix.json# Metadata (contexts, phrases, shape)
│       │   └── idf_weights.json        # IDF weights for each context
│       ├── semantic_space/             # Step 3 output
│       │   ├── context_coordinates.json # t-SNE projected coordinates
│       │   ├── context_coordinates.csv # Coordinates in CSV format
│       │   ├── context_coordinates_continuous.csv
│       │   ├── coordinate_statistics.json
│       │   └── grid_visualization.*    # Optional grid visualization
│       ├── phrase_fingerprints/        # Step 4 output
│       │   ├── phrase_fingerprints.npz # Dense fingerprint matrix
│       │   ├── phrase_fingerprints_meta.json # Phrase -> row index mapping
│       │   └── phrase_fingerprints_stats.json
│       └── doc_fingerprints/           # Step 5 output
│           ├── doc_fingerprints.npz    # Dense fingerprint matrix
│           ├── doc_fingerprints_meta.json # Doc -> row + flags
│           └── doc_fingerprints_stats.json
│
└── benchmarks/
    └── benchmark_<timestamp>/           # Phase 2+3: Benchmark + Report
        ├── config.yml                  # Benchmark config (run ref, query range)
        ├── summary.json                # Aggregate metrics over all queries
        ├── results_log.csv              # Per-query metrics in tabular form
        ├── benchmark_report.md          # Comprehensive Markdown report
        ├── analysis.json               # Deep-dive analysis (from benchmark_analyzer.py)
        └── per_query/
            ├── 0000/                    # Per-query folder (query index)
            │   ├── candidate_docs.json  # This query's 20 candidate doc IDs
            │   ├── query_results.json   # Raw Step 6 output (all combined corpus docs)
            │   └── filtered_results.json# Filtered to 20 candidates + metrics
            ├── 0001/
            ├── 0002/
            └── ...
```

Additionally, a lightweight run registry is maintained next to the runner script:

```
semantic_folding/dataset_benchmark/musique/
├── run_benchmark.py
├── runs/
│   └── registry.yml                    # Tracks all runs for resume/report
```

---

## Performance Notes (Three-Phase Design)

The three-phase design drastically reduces runtime vs per-query pipeline execution:

| Setup | Phase 1 (Index 100 queries) | Phase 2 (100 queries) | Total |
|-------|----------------------------|----------------------|-------|
| Grid 16, 100 queries | ~2 min | ~5–10 min | ~7–12 min |
| Grid 64, 100 queries | ~15–30 min | ~20–40 min | ~35–70 min |

- Phase 1 (Steps 1-5) scales with **document count**: ~2,000 unique docs for
  100 queries vs ~10,000 for 500 queries. t-SNE is the bottleneck (~O(n²) in
  document count).
- Phase 2 (Step 6 per query) is **embarrassingly parallel** — each query is
  independent. A parallel launcher can be added by partitioning queries.
- GPU is not used; all computation is CPU-based (t-SNE via scikit-learn, no
  neural models).
- Old per-query design (not used): 100 queries on grid 64 would take ~2–3 hours.

---

## Comparison to HippoRAG

MuSiQue is a primary evaluation dataset in both HippoRAG papers. The standard
HippoRAG protocol on MuSiQue uses the same 20-passage candidate pool per query
with gold supporting passage labels, making results directly comparable.

| Aspect | HippoRAG | Semantic Folding |
|--------|----------|------------------|
| Indexing cost | LLM-based OpenIE per passage | Statistical phrase extraction only |
| Index size | Dense embeddings + KG triples | Sparse fingerprints (4K bits per doc) |
| Retrieval | PPR + dense retrieval | Normalised dot-product |
| Interpretability | Yes (KG path traversal) | Yes (grid positions) |

---

## Citation

When using this benchmark, please cite:

```
@inproceedings{trivedi2022musique,
  title={MuSiQue: Multi-hop Sentence Queries},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  booktitle={ACL},
  year={2022}
}
```
