# Semantic Folding Pipeline — Project Rules

## Project Structure

- Pipeline scripts: `semantic_folding/*.py`
- Entry point: `semantic_folding/semantic_folder.py` (interactive TUI)
- Individual steps can be run directly:
  - `semantic_folding/phrase_extractor.py` (Step 1)
  - `semantic_folding/term_context.py` (Step 2)
  - `semantic_folding/semantic_space.py` (Step 3)
  - `semantic_folding/phrase_fingerprints.py` (Step 4)
  - `semantic_folding/doc_fingerprints.py` (Step 5)
  - `semantic_folding/query_processor.py` (Step 6)
- Utilities: `semantic_folding/lib.py`
- Parameter tuning study: `semantic_folding/parameters_tuning.md`
- Visualizers: `semantic_folding/phrase_visualizer.py`, `semantic_folding/doc_visualizer.py`
- Notebooks: `semantic_folding/notebooks/`
- Outputs: `outputs/run_<timestamp>/`
- Config: `config/semantic_folding.yml`, `config/exec_state.yml`
- Test queries: `data/qa-sample.md`
- Evaluation reports: `outputs/run_<timestamp>/query_metrics/qa_evaluation_report.md`
- Benchmarking framework: `semantic_folding/dataset_benchmark/`
  - MuSiQue benchmark: `semantic_folding/dataset_benchmark/musique/run_benchmark.py`
  - Analysis: `semantic_folding/dataset_benchmark/musique/benchmark_analyzer.py`
- Benchmark overview: `semantic_folding/benchmarks.md`

## Python Environment

- Virtual env: `.venv\scripts\python` (Windows)
- spaCy model: `en_core_web_sm`
- Key deps: `numpy`, `scipy`, `spacy`, `plotly`, `scikit-learn`, `pyyaml`

## Pipeline Parameters

- Grid size: 64x64 (optimal for 20-doc corpus; set in config; must match across Steps 3–6)
  - Sweep results: 64×64 beats 128×128 (MRR 1.000 vs 0.900, AP 0.869 vs 0.836)
- Encoding: Morton Z-order (`use_morton: true`)
- Smoothing: Gaussian blur, sigma=1.5 (`no_smooth: false` to enable)
- TF-IDF: applied in Step 2
- Dim reduction: t-SNE (default; also supports UMAP, PCA)
- Spreading: radius=1, decay=0.5 (in query processor) — spread=0 loses recall on Q4/C09, spread=2 doesn't improve
- top_percent: 0.10 (0.05 loses C00 in Q5, 0.15 dilutes signal)
- Query weighting: IDF (best; uniform drops C17 ranking and loses C00)
- Normalization: L2 for query, `sqrt(nnz)` for document fingerprints
- Geometric scoring: optional `--geometric` flag (Step 6) applies a 3×3 spatial adjacency kernel before scoring, rewarding nearby (not just exact) cell overlap on the 2D grid. See `semantic_folding/parameters_tuning.md` for evaluation.

## Benchmarking (MuSiQue)

- **Script**: `semantic_folding/dataset_benchmark/musique/run_benchmark.py`
- **Three-phase design**:
  - Phase 1 (index): Collect unique paragraphs across query range, run Steps 1-5 once
  - Phase 2 (benchmark): Run Step 6 per query against pre-built fingerprints, post-filter to 20 candidates
  - Phase 3 (report): Auto-generate `benchmark_report.md`
- **Interactive TUI** (default, no args): Colorama-colored menu with parameter auto-generation & user override
- **CLI mode** (`--mode index|benchmark|report`): Non-interactive for automation; flags same as before
- **Run registry**: `semantic_folding/dataset_benchmark/musique/runs/registry.yml` tracks all index & benchmark runs for resume
- **Analysis**: `semantic_folding/dataset_benchmark/musique/benchmark_analyzer.py` — deep-dive into last benchmark results (distributions, failures, top performers)
- **Key command (interactive)**: `.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py` (no args = TUI with colored menu + param auto-generation & user override)
- **Key command (CLI)**: `.venv\scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py --mode index --split dev --max-queries 100 --grid-size 64 --spreading-steps 1 --top-percent 0.10 --weighting idf --benchmark`
- **Params read from run's config.yml during benchmark** (must match index phase)
- **Metrics**: MRR, AP, P@K, R@K, NDCG@K
- **Output layout**:
  ```
  outputs/musique_benchmark/
    runs/run_<ts>/          # Phase 1: combined corpus + Steps 1-5 artifacts
    benchmarks/benchmark_<ts>/  # Phase 2: per-query results + report
  ```

## Ground Truth Conventions

- `data/qa-sample.md` defines 5 test queries with 3 relevant documents each
- Document IDs match corpus line numbers (C00–C19)
- Relevance: binary (relevant/not) — update for graded when needed

## Evaluation

- Metrics script: run `.venv\scripts\python tools\compute_ir_metrics.py`
- Report output: `outputs/run_<timestamp>/query_metrics/qa_evaluation_report.md`
- Key metrics: P@K, R@K, MRR, NDCG@K, AP
- Normalization diagnostic: check doc_nnz uniqueness

## Naming Conventions

- Python: snake_case for functions/variables
- CLI flags: kebab-case (e.g., `--grid-size`, `--no-smooth`)
- Config keys: snake_case (e.g., `grid_size`, `no_smooth`)
- Output dirs: `snake_case` (e.g., `phrase_fingerprints`, `query_results`)

## Git Conventions

- Tags follow `v<major>.<minor>` pattern (current: v3.2)
- Commit messages: lowercase, descriptive
