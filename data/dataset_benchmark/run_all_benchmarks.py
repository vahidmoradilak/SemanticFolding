"""
Run all dataset benchmarks sequentially.

For each dataset:
  1. Download raw data (via adapter.download)
  2. Convert to MuSiQue-like JSONL (via adapter.convert_to_musique_format)
  3. Phase 1: Index corpus (Steps 1-5)
  4. Phase 2: Semantic folding benchmark (Step 6 per query)
  5. Phase 2: BM25 baseline benchmark
  6. Phase 3: Generate reports

Usage:
    .venv\\Scripts\\python semantic_folding\\dataset_benchmark\\run_all_benchmarks.py
    .venv\\Scripts\\python semantic_folding\\dataset_benchmark\\run_all_benchmarks.py --datasets belebele bioasq
    .venv\\Scripts\\python semantic_folding\\dataset_benchmark\\run_all_benchmarks.py --max-queries 100
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_folding.dataset_benchmark.adapters import get_adapter, ADAPTER_REGISTRY
from semantic_folding.dataset_benchmark.generic_benchmark import GenericBenchmarkRunner, load_entries, load_dataset_registry
from semantic_folding.dataset_benchmark.bm25_benchmark import run_bm25_benchmark
from lib import get_logger

logger = get_logger("run_all")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DATASETS_TO_RUN = ["belebele", "bioasq", "popqa", "nq_rear"]


def run_single_dataset(dataset_name: str, max_queries: int = None,
                       skip_index: bool = False, skip_bm25: bool = False,
                       only_bm25: bool = False, run_dir_override: Path = None,
                       registry_path: Path = None):
    """Run full benchmark pipeline for one dataset."""

    separator = f"{'=' * 60}"
    logger.info(f"\n{separator}")
    logger.info(f"  DATASET: {dataset_name}")
    logger.info(f"{separator}")

    adapter = get_adapter(dataset_name)
    raw_dir = DATA_DIR / dataset_name / "raw"
    converted_dir = DATA_DIR / dataset_name / "converted"

    t_start = time.time()

    # Step 1: Download
    logger.info(f"[1/5] Downloading {adapter.display_name}...")
    try:
        adapter.download(raw_dir)
    except FileNotFoundError as e:
        logger.error(f"Download failed: {e}")
        return False

    # Step 2: Convert
    logger.info(f"[2/5] Converting to MuSiQue format...")
    jsonl_path = converted_dir / f"{dataset_name}.jsonl"
    if not jsonl_path.exists():
        try:
            jsonl_path = adapter.convert_to_musique_format(raw_dir, converted_dir, max_queries=max_queries or 999999)
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
    else:
        logger.info(f"  JSONL already exists: {jsonl_path}")

    # Verify JSONL
    entries = load_entries(jsonl_path)
    actual_queries = len(entries)
    if max_queries:
        actual_queries = min(actual_queries, max_queries)
    logger.info(f"  Total queries in JSONL: {len(entries)}, will benchmark: {actual_queries}")

    if actual_queries == 0:
        logger.error("No queries to benchmark")
        return False

    runner = GenericBenchmarkRunner(adapter)

    # Apply registry params (per-dataset recommended settings)
    if registry_path:
        registry_params = load_dataset_registry(registry_path, dataset_name)
    else:
        registry_params = load_dataset_registry(dataset=dataset_name)
    if registry_params:
        runner.params.update(registry_params)
        logger.info(f"  [REGISTRY] Applied {len(registry_params)} params for {dataset_name}")

    run_dir = run_dir_override

    if not only_bm25:
        # Step 3: Phase 1 - Index
        if not skip_index and run_dir is None:
            logger.info(f"[3/5] Phase 1: Indexing corpus (Steps 1-5)...")
            run_dir = runner.phase1_index(jsonl_path, max_queries=max_queries)
            if run_dir is None:
                logger.error("Indexing failed")
                return False
        elif run_dir is not None:
            logger.info(f"[3/5] Using existing run directory: {run_dir}")
        else:
            logger.info(f"[3/5] Skipping indexing (existing run)")
            # Find latest completed run
            runs_dir = runner.runs_dir
            completed = sorted(runs_dir.glob("run_*"))
            for r in reversed(completed):
                status_file = r / "metadata.json"
                if status_file.exists():
                    run_dir = r
                    break
            if run_dir is None:
                logger.error("No existing run found. Cannot skip indexing.")
                return False

        # Step 4: Phase 2 - Semantic Folding Benchmark
        logger.info(f"[4/5] Phase 2: Semantic folding benchmark...")
        bench_dir = runner.phase2_benchmark(run_dir, jsonl_path, query_end=max_queries)
        if bench_dir:
            runner.phase3_report(bench_dir)
            runner.analyze(bench_dir)
            logger.info(f"  Semantic folding benchmark complete: {bench_dir}")
        else:
            logger.error("Semantic folding benchmark failed")

        # Step 5: BM25 Baseline
        if not skip_bm25:
            logger.info(f"[5/5] BM25 baseline benchmark...")
            try:
                bm25_dir = run_bm25_benchmark(
                    dataset=dataset_name,
                    jsonl_path=jsonl_path,
                    run_dir=run_dir,
                    query_end=max_queries,
                )
                if bm25_dir:
                    logger.info(f"  BM25 benchmark complete: {bm25_dir}")
                else:
                    logger.warning("BM25 benchmark returned no results")
            except Exception as e:
                logger.error(f"BM25 benchmark failed: {e}")
    else:
        # Only BM25
        if run_dir is None:
            runs_dir = runner.runs_dir
            completed = sorted(runs_dir.glob("run_*"))
            for r in reversed(completed):
                if (r / "corpus.txt").exists():
                    run_dir = r
                    break
        if run_dir is None:
            logger.error("No existing run directory found for BM25-only mode")
            return False
        logger.info(f"[BM25 only] Using run directory: {run_dir}")
        bm25_dir = run_bm25_benchmark(
            dataset=dataset_name, jsonl_path=jsonl_path,
            run_dir=run_dir, query_end=max_queries,
        )

    elapsed = time.time() - t_start
    logger.info(f"\n{dataset_name} completed in {elapsed / 60:.1f} minutes")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all dataset benchmarks")
    parser.add_argument("--datasets", nargs="+", default=DATASETS_TO_RUN,
                        help=f"Datasets to benchmark (default: {DATASETS_TO_RUN})")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Max queries per dataset (None = all)")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip Phase 1 indexing (use existing run)")
    parser.add_argument("--skip-bm25", action="store_true",
                        help="Skip BM25 baseline")
    parser.add_argument("--only-bm25", action="store_true",
                        help="Only run BM25 baseline (requires existing index)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Override run directory for all datasets")
    parser.add_argument("--registry", type=Path, default=None,
                        help="Path to dataset_registry.yml for per-dataset parameter overrides")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    results = {}

    for ds in args.datasets:
        if ds not in ADAPTER_REGISTRY:
            logger.warning(f"Unknown dataset '{ds}', skipping. Available: {list(ADAPTER_REGISTRY.keys())}")
            continue
        try:
            ok = run_single_dataset(
                ds, max_queries=args.max_queries,
                skip_index=args.skip_index, skip_bm25=args.skip_bm25,
                only_bm25=args.only_bm25, run_dir_override=run_dir,
                registry_path=args.registry,
            )
            results[ds] = "OK" if ok else "FAILED"
        except Exception as e:
            logger.error(f"ERROR on {ds}: {e}")
            results[ds] = f"ERROR: {e}"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ds, status in results.items():
        print(f"  {ds:20s} {status}")
    print()


if __name__ == "__main__":
    main()
