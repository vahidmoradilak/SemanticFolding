#!/usr/bin/env python3
"""
Run benchmarks with SPLADE fusion enabled.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "semantic_folding"))

from semantic_folding.dataset_benchmark.adapters import get_adapter
from semantic_folding.dataset_benchmark.generic_benchmark import GenericBenchmarkRunner, load_entries
from semantic_folding.dataset_benchmark.bm25_benchmark import run_bm25_benchmark
from lib import get_logger

logger = get_logger("run_splade")


def run_benchmark_with_splade(dataset_name: str, alpha: float = 0.3, max_queries: int = None):
    """Run benchmark with SPLADE fusion enabled."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  DATASET: {dataset_name} (SPLADE alpha={alpha})")
    logger.info(f"{'=' * 60}")

    adapter = get_adapter(dataset_name)
    datasets_dir = Path("data/datasets")
    converted_dir = datasets_dir / dataset_name / "converted"
    jsonl_path = converted_dir / f"{dataset_name}.jsonl"

    if not jsonl_path.exists():
        logger.error(f"JSONL not found: {jsonl_path}")
        return False

    entries = load_entries(jsonl_path)
    actual_queries = len(entries)
    if max_queries:
        actual_queries = min(actual_queries, max_queries)
    logger.info(f"  Total queries: {len(entries)}, will benchmark: {actual_queries}")

    runner = GenericBenchmarkRunner(adapter)

    # Enable SPLADE fusion
    runner.params["splade"] = True
    runner.params["hybrid_alpha"] = alpha
    runner.params["fusion_method"] = "linear"
    runner.params["splade_model"] = "naver/splade-cocondenser-ensembledistil"
    # SPLADE requires corpus path
    runner.params["corpus_path"] = str(run_dir / "corpus.txt")

    logger.info(f"  SPLADE enabled: alpha={alpha}, model={runner.params['splade_model']}")

    # Phase 1: Index
    logger.info("[Phase 1] Indexing corpus...")
    run_dir = runner.phase1_index(jsonl_path, max_queries=max_queries)
    if run_dir is None:
        logger.error("Indexing failed")
        return False

    # Phase 2: Benchmark
    logger.info("[Phase 2] Running SPLADE benchmark...")
    benchmark_dir = runner.phase2_benchmark(run_dir, jsonl_path)

    # BM25 baseline
    logger.info("[Phase 3] Running BM25 baseline...")
    bm25_dir = run_bm25_benchmark(
        dataset=dataset_name, jsonl_path=jsonl_path,
        run_dir=run_dir, query_end=max_queries,
    )

    logger.info(f"\n{dataset_name} with SPLADE completed")
    return True


if __name__ == "__main__":
    datasets = ["belebele", "narrativeqa", "pubmedqa"]
    alpha = 0.3

    for ds in datasets:
        try:
            run_benchmark_with_splade(ds, alpha=alpha, max_queries=100)
        except Exception as e:
            logger.error(f"ERROR on {ds}: {e}")
