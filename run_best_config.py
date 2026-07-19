#!/usr/bin/env python3
"""
Run best configuration on all four datasets.
Best config: splade=True, hybrid_alpha=0.3, fusion_method=linear
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
sys.path.insert(0, str(Path('semantic_folding').resolve()))

from semantic_folding.dataset_benchmark.adapters import get_adapter
from semantic_folding.dataset_benchmark.generic_benchmark import GenericBenchmarkRunner, load_entries
from semantic_folding.dataset_benchmark.bm25_benchmark import run_bm25_benchmark
from lib import get_logger

logger = get_logger('best_config')


def run_dataset(dataset_name: str, jsonl_filename: str, max_queries: int):
    """Run best config on a single dataset."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  DATASET: {dataset_name}")
    logger.info(f"{'=' * 60}")

    adapter = get_adapter(dataset_name)
    jsonl_path = Path(f'data/datasets/{dataset_name}/converted/{jsonl_filename}')

    if not jsonl_path.exists():
        logger.error(f"JSONL not found: {jsonl_path}")
        return None

    entries = load_entries(jsonl_path)
    actual_queries = min(len(entries), max_queries)
    logger.info(f"  Total queries: {len(entries)}, will benchmark: {actual_queries}")

    runner = GenericBenchmarkRunner(adapter)

    # Best configuration
    runner.params['splade'] = True
    runner.params['hybrid_alpha'] = 0.3
    runner.params['fusion_method'] = 'linear'
    runner.params['splade_model'] = 'naver/splade-cocondenser-ensembledistil'

    # Phase 1: Index
    logger.info("[Phase 1] Indexing corpus...")
    run_dir = runner.phase1_index(jsonl_path, max_queries=max_queries)
    if run_dir is None:
        logger.error("Indexing failed")
        return None

    # Set corpus path for SPLADE
    runner.params['corpus_path'] = str(run_dir / 'corpus.txt')

    # Phase 2: Benchmark
    logger.info("[Phase 2] Running benchmark...")
    benchmark_dir = runner.phase2_benchmark(run_dir, jsonl_path)

    # BM25 baseline
    logger.info("[Phase 3] Running BM25 baseline...")
    bm25_dir = run_bm25_benchmark(
        dataset=dataset_name, jsonl_path=jsonl_path,
        run_dir=run_dir, query_end=max_queries,
    )

    return {
        'dataset': dataset_name,
        'run_dir': str(run_dir),
        'benchmark_dir': str(benchmark_dir),
    }


if __name__ == "__main__":
    datasets = [
        ('belebele', 'belebele.jsonl', 100),
        ('narrativeqa', 'narrativeqa.jsonl', 50),
        ('pubmedqa', 'pubmedqa_pqa_labeled.jsonl', 311),
        ('popqa', 'popqa.jsonl', 1000),
    ]

    results = []
    for name, jsonl, max_q in datasets:
        try:
            result = run_dataset(name, jsonl, max_q)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"ERROR on {name}: {e}")

    print("\n" + "=" * 60)
    print("COMPLETED DATASETS")
    print("=" * 60)
    for r in results:
        print(f"  {r['dataset']:20s} {r['benchmark_dir']}")
