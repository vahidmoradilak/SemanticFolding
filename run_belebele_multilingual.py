"""
Belebele Multilingual Benchmark — SF vs BM25 across 4 languages.
Runs index + benchmark for each language, then generates comparison report.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_folding.dataset_benchmark.adapters import get_adapter
from semantic_folding.dataset_benchmark.generic_benchmark import GenericBenchmarkRunner
from semantic_folding.dataset_benchmark.bm25_benchmark import run_bm25_benchmark

RAW_BASE = Path("data/datasets/belebele/raw")
OUTPUT_BASE = Path("outputs/belebele_multilingual")

# Languages to benchmark
LANGUAGES = [
    ("eng_Latn", "English"),
    ("fra_Latn", "French"),
    ("arb_Arab", "Arabic"),
    ("pes_Arab", "Persian"),
]

PIPELINE_PARAMS = {
    "grid_size": 64,
    "spreading_steps": 1,
    "top_k": 100,
    "weighting": "idf",
    "top_percent": 0.10,
    "smoothing_sigma": 1.5,
    "keep_verbs": True,
    "min_word_length": 3,
    "min_freq": 1,
    "morton": True,
    "method": "umap",
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.0,
    "random_seed": 42,
    "doc_norm": "l2",
}

NUM_QUERIES = 100  # start with 100 for fast comparison

results = {}

for lang_code, lang_name in LANGUAGES:
    print(f"\n{'='*60}")
    print(f"  Belebele — {lang_name} ({lang_code})")
    print(f"{'='*60}")

    adapter = get_adapter("belebele", language=lang_code)

    raw_path = adapter.download(RAW_BASE)
    jsonl_path = adapter.convert_to_musique_format(
        raw_path, RAW_BASE.parent / "converted", max_queries=NUM_QUERIES
    )
    print(f"  JSONL: {jsonl_path} ({jsonl_path.stat().st_size:,} bytes)")

    runner = GenericBenchmarkRunner(adapter, params=dict(PIPELINE_PARAMS))

    # Phase 1: Index
    print(f"\n  [Phase 1] Indexing {NUM_QUERIES} queries...")
    t0 = time.time()
    run_dir = runner.phase1_index(jsonl_path, max_queries=NUM_QUERIES)
    if not run_dir:
        print(f"  [ERROR] Index phase failed for {lang_name}")
        results[lang_code] = None
        continue
    t1 = time.time()
    print(f"  Index done in {t1-t0:.0f}s -> {run_dir.name}")

    # Phase 2: SF Benchmark
    print(f"\n  [Phase 2] SF Benchmarking {NUM_QUERIES} queries...")
    t0 = time.time()
    bench_dir = runner.phase2_benchmark(run_dir, jsonl_path, query_start=0, query_end=NUM_QUERIES)
    if not bench_dir:
        print(f"  [ERROR] SF Benchmark phase failed for {lang_name}")
        results[lang_code] = None
        continue
    t1 = time.time()
    print(f"  SF Benchmark done in {t1-t0:.0f}s -> {bench_dir.name}")

    # Read SF aggregate results
    agg_path = bench_dir / "summary.json"
    sf_mrr = 0
    sf_ap = 0
    if agg_path.exists():
        with open(agg_path) as f:
            agg = json.load(f)
        sf_mrr = agg.get("mean_mrr", 0)
        sf_ap = agg.get("mean_ap", 0)
        print(f"  SF MRR={sf_mrr:.4f}  AP={sf_ap:.4f}")
    else:
        print(f"  [WARN] No summary.json found at {agg_path}")

    runner.phase3_report(bench_dir)

    # Phase 2b: BM25 Benchmark
    print(f"\n  [Phase 2b] BM25 Benchmarking {NUM_QUERIES} queries...")
    t0 = time.time()
    bm25_bench_dir = run_bm25_benchmark(
        dataset="belebele",
        jsonl_path=jsonl_path,
        run_dir=run_dir,
        query_start=0,
        query_end=NUM_QUERIES,
        top_k=100,
    )
    if not bm25_bench_dir:
        print(f"  [ERROR] BM25 Benchmark failed for {lang_name}")
        results[lang_code] = None
        continue
    t1 = time.time()
    print(f"  BM25 Benchmark done in {t1-t0:.0f}s -> {bm25_bench_dir.name}")

    # Read BM25 aggregate results
    bm25_agg_path = bm25_bench_dir / "summary.json"
    bm25_mrr = 0
    bm25_ap = 0
    if bm25_agg_path.exists():
        with open(bm25_agg_path) as f:
            bm25_agg = json.load(f)
        bm25_mrr = bm25_agg.get("mean_mrr", 0)
        bm25_ap = bm25_agg.get("mean_ap", 0)
        print(f"  BM25 MRR={bm25_mrr:.4f}  AP={bm25_ap:.4f}")
    else:
        print(f"  [WARN] No summary.json found at {bm25_agg_path}")

    results[lang_code] = {
        "name": lang_name,
        "sf_mrr": sf_mrr,
        "sf_ap": sf_ap,
        "bm25_mrr": bm25_mrr,
        "bm25_ap": bm25_ap,
        "num_queries": agg.get("num_queries", NUM_QUERIES),
    }
    print(f"  SF={sf_mrr:.4f} vs BM25={bm25_mrr:.4f}  Δ={sf_mrr - bm25_mrr:+.4f}")


# =======================================================================
# Generate combined multilingual comparison report
# =======================================================================
print(f"\n\n{'='*60}")
print("  Belebele Multilingual Comparison — SF vs BM25")
print(f"{'='*60}")

report_lines = []
report_lines.append("# Belebele Multilingual Benchmark Results")
report_lines.append("")
report_lines.append(f"**Queries per language:** {NUM_QUERIES}")
report_lines.append("")
report_lines.append("## Aggregate Results")
report_lines.append("")
report_lines.append("| Language | SF MRR | SF AP | BM25 MRR | BM25 AP | Δ MRR | Winner |")
report_lines.append("|----------|:-----:|:-----:|:--------:|:--------:|:-----:|:------:|")

for lang_code, lang_name in LANGUAGES:
    r = results.get(lang_code)
    if r is None:
        report_lines.append(f"| {lang_name} | FAILED | FAILED | FAILED | FAILED | — | — |")
        continue
    delta = r["sf_mrr"] - r["bm25_mrr"]
    winner = "SF" if delta > 0 else ("BM25" if delta < 0 else "Tie")
    report_lines.append(
        f"| {lang_name} | {r['sf_mrr']:.4f} | {r['sf_ap']:.4f} | {r['bm25_mrr']:.4f} | {r['bm25_ap']:.4f} | {delta:+.4f} | {winner} |"
    )
    print(f"  {lang_name:10s} | SF={r['sf_mrr']:.4f} | BM25={r['bm25_mrr']:.4f} | Δ={delta:+.4f} | {winner}")

report_lines.append("")
report_lines.append("## Per-Language Details")
for lang_code, lang_name in LANGUAGES:
    r = results.get(lang_code)
    if r is None:
        continue
    report_lines.append(f"")
    report_lines.append(f"### {lang_name} ({lang_code})")
    report_lines.append(f"- Queries: {r['num_queries']}")
    report_lines.append(f"- SF MRR: {r['sf_mrr']:.4f}")
    report_lines.append(f"- SF AP: {r['sf_ap']:.4f}")
    report_lines.append(f"- BM25 MRR: {r['bm25_mrr']:.4f}")
    report_lines.append(f"- BM25 AP: {r['bm25_ap']:.4f}")
    report_lines.append(f"- Δ MRR: {r['sf_mrr'] - r['bm25_mrr']:+.4f}")

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
report_path = OUTPUT_BASE / "multilingual_comparison.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"\nReport saved to {report_path}")

# Save raw results as JSON
with open(OUTPUT_BASE / "multilingual_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nDone!")
