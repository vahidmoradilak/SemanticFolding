"""
BM25 baseline benchmark for Semantic Folding pipeline.

Runs the same retrieval task as the main benchmark but uses BM25 lexical
scoring instead of semantic folding fingerprints. Output format is identical
(per_query, summary.json, results_log.csv) so results can be compared directly.

Usage:
    python -m semantic_folding.dataset_benchmark.bm25_benchmark
        --dataset pubmedqa --jsonl data/pubmedqa/pubmedqa_pqa_labeled.jsonl
        --run-dir outputs/pubmedqa_benchmark/runs/run_20260606_161956
        --query-start 0 --query-end 200
"""
import argparse, csv, json, math, re, sys, time, yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import get_logger
from .generic_benchmark import (
    load_entries, compute_metrics, filter_results_to_candidates,
    register_run, update_run_status, OUTPUTS_DIR,
)

logger = get_logger("bm25_bench")


def parse_corpus(corpus_path: Path) -> Tuple[List[str], List[str]]:
    """Parse corpus.txt into (doc_ids, texts). Handles multi-line documents."""
    doc_ids, texts = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: doc_000000, Title text...
            comma_idx = line.find(",")
            if comma_idx == -1 or not line[:comma_idx].startswith("doc_"):
                # Continuation line — append to last text
                if texts:
                    texts[-1] += " " + line
                continue
            gid = line[:comma_idx].strip()
            text = line[comma_idx + 1:].strip()
            doc_ids.append(gid)
            texts.append(text)
    return doc_ids, texts


class BM25Scorer:
    """BM25 scorer using sklearn CountVectorizer + numpy."""

    def __init__(self, corpus_texts: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(analyzer="word", lowercase=True)
        self.tf_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.doc_lengths = np.array(self.tf_matrix.sum(axis=1)).flatten()
        self.avg_doc_len = float(self.doc_lengths.mean())
        self.n_docs = len(corpus_texts)
        self._compute_idf()

    def _compute_idf(self):
        """Compute BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)"""
        df = np.array((self.tf_matrix > 0).sum(axis=0)).flatten()
        self.idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> List[Tuple[str, float]]:
        """Score all corpus documents against query. Returns [(doc_id, score), ...]"""
        q_vec = self.vectorizer.transform([query])
        q_terms = q_vec.indices

        if len(q_terms) == 0:
            return []

        # term frequencies in each doc for matching query terms
        tf = self.tf_matrix[:, q_terms]  # sparse (n_docs, n_terms)

        # BM25: idf * (tf * (k1+1)) / (tf + k1 * (1 - b + b * |D| / avgdl))
        doc_lens = self.doc_lengths[:, np.newaxis]
        idf_vals = self.idf[q_terms]  # (n_terms,)

        # Convert to dense for safe arithmetic (small: n_docs*~n_terms)
        tf_dense = np.array(tf.toarray(), dtype=np.float64)
        denom = tf_dense + self.k1 * (1.0 - self.b + self.b * doc_lens / self.avg_doc_len)
        scores = idf_vals * tf_dense * (self.k1 + 1.0) / denom
        doc_scores = scores.sum(axis=1).flatten()

        # Return all docs with non-zero scores, sorted descending
        nonzero = np.where(doc_scores > 0)[0]
        if len(nonzero) == 0:
            return []
        sorted_idx = nonzero[np.argsort(doc_scores[nonzero])[::-1]]
        return [(f"doc_{i:06d}", float(doc_scores[i])) for i in sorted_idx]


def run_bm25_benchmark(
    dataset: str,
    jsonl_path: Path,
    run_dir: Path,
    query_start: int = 0,
    query_end: int = None,
    top_k: int = 5,
) -> Optional[Path]:
    """Run BM25 benchmark, return benchmark dir path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_base = OUTPUTS_DIR / f"{dataset}_benchmark" / "benchmarks"
    bench_base.mkdir(parents=True, exist_ok=True)
    bench_dir = bench_base / f"benchmark_{ts}"
    per_query_dir = bench_dir / "per_query"
    bench_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir.mkdir(exist_ok=True)

    # Load data
    with open(run_dir / "query_doc_map.json", encoding="utf-8") as f:
        query_doc_map = json.load(f)
    with open(run_dir / "query_gold.json", encoding="utf-8") as f:
        query_gold = json.load(f)

    # Parse corpus and build BM25 index
    corpus_path = run_dir / "corpus.txt"
    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        return None
    doc_ids, texts = parse_corpus(corpus_path)
    doc_id_to_idx = {gid: i for i, gid in enumerate(doc_ids)}
    logger.info(f"Loaded corpus: {len(doc_ids)} documents")

    bm25 = BM25Scorer(texts)
    logger.info("BM25 index built")

    # Save config
    params = {
        "grid_size": 0,  # N/A for BM25
        "spreading_steps": 0,
        "top_k": top_k,
        "weighting": "bm25",
        "smoothing_sigma": 0,
        "keep_verbs": False,
        "min_word_length": 0,
        "min_freq": 1,
    }
    bench_config = {
        "phase2": {
            "mode": "bm25_benchmark",
            "dataset": dataset,
            "timestamp": ts,
            "run_dir": str(run_dir),
            "query_start": query_start,
            "query_end": query_end,
        },
        "pipeline": params,
    }
    with open(bench_dir / "config.yml", "w") as f:
        yaml.dump(bench_config, f, default_flow_style=False)
    register_run(bench_dir, dataset, "bm25_benchmark", params, "running")

    # Load entries
    entries = load_entries(jsonl_path)
    if query_end is None:
        query_end = len(entries)
    else:
        query_end = min(query_end, len(entries))

    all_metrics = []
    results_log = bench_dir / "results_log.csv"
    failed = 0
    total = query_end - query_start

    logger.info(f"BM25 benchmark: {bench_dir.name} - queries {query_start}-{query_end}")

    for i, q_idx in enumerate(range(query_start, query_end)):
        q_idx_str = str(q_idx)
        entry = entries[q_idx]
        query_text = entry["question"]
        candidate_ids = query_doc_map.get(q_idx_str, [])
        gold_ids = query_gold.get(q_idx_str, [])

        if not gold_ids:
            logger.debug(f"  [{q_idx}] no gold passages, skipping")
            continue

        n_words = len(query_text.split())
        query_out_dir = per_query_dir / f"{q_idx:04d}"
        query_out_dir.mkdir(exist_ok=True)

        # Save candidate info
        with open(query_out_dir / "candidate_docs.json", "w") as f:
            json.dump({"candidate_ids": candidate_ids, "gold_ids": gold_ids}, f, indent=2)

        # Score with BM25
        t0 = time.time()
        all_results = bm25.score(query_text)
        elapsed = time.time() - t0

        if len(all_results) == 0:
            logger.warning(f"  [{q_idx}] BM25 returned no results")
            failed += 1
            continue

        # Filter to candidates
        candidate_results = filter_results_to_candidates(all_results, candidate_ids)

        # Save full and filtered results
        with open(query_out_dir / "query_results.json", "w") as f:
            json.dump([{"query": query_text, "results": all_results}], f, indent=2)
        with open(query_out_dir / "filtered_results.json", "w") as f:
            json.dump({
                "query_idx": q_idx,
                "query": query_text,
                "query_word_count": n_words,
                "spreading_steps_used": 0,
                "spreading_reason": "bm25",
                "gold": gold_ids,
                "candidates": candidate_ids,
                "filtered_ranked": [(doc_id, float(score)) for doc_id, score in candidate_results],
                "full_top10": [(doc_id, float(score)) for doc_id, score in all_results[:10]],
                "elapsed_s": round(elapsed, 3),
            }, f, indent=2)

        metrics = compute_metrics(candidate_results, gold_ids,
                                  top_k_list=[1, 2, 3, 5, top_k])
        metrics["spreading_steps"] = 0
        all_metrics.append(metrics)

        if (i + 1) % 10 == 0 or i == 0 or i == total - 1:
            logger.info(f"  [{q_idx:04d}/{query_end - 1}] MRR={metrics['mrr']:.3f} "
                        f"AP={metrics['ap']:.3f} P@2={metrics['p@2']:.3f} "
                        f"[{elapsed:.2f}s]  ({i+1}/{total})")

        # Write CSV row
        with open(results_log, "a", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            if i == 0:
                header = ["query_idx", "query", "n_words", "spread", "spread_reason",
                          "mrr", "ap", "p@1", "p@2", "p@3", "p@5", "r@2", "ndcg@2",
                          "found_at", "elapsed_s"]
                writer.writerow(header)
            writer.writerow([
                q_idx, query_text, n_words, 0, "bm25",
                f"{metrics['mrr']:.4f}", f"{metrics['ap']:.4f}",
                f"{metrics['p@1']:.4f}", f"{metrics['p@2']:.4f}",
                f"{metrics['p@3']:.4f}", f"{metrics['p@5']:.4f}",
                f"{metrics['r@2']:.4f}", f"{metrics['ndcg@2']:.4f}",
                metrics["found_at"], round(elapsed, 1),
            ])

    # Summary
    n = len(all_metrics)
    if n == 0:
        logger.error("No queries completed")
        update_run_status(bench_dir, dataset, "failed")
        return None

    summary = {
        "dataset": dataset,
        "display_name": dataset,
        "num_queries": n,
        "failed": failed,
        "mean_mrr": float(np.mean([m["mrr"] for m in all_metrics])),
        "mean_ap": float(np.mean([m["ap"] for m in all_metrics])),
        "mean_p@1": float(np.mean([m["p@1"] for m in all_metrics])),
        "mean_p@2": float(np.mean([m["p@2"] for m in all_metrics])),
    }
    with open(bench_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(bench_dir / "params.json", "w") as f:
        json.dump({
            "dataset": dataset,
            "display_name": dataset,
            "run_dir": str(run_dir),
            "num_queries": n,
            "failed": failed,
            "pipeline": params,
            "generated": datetime.now().isoformat(),
        }, f, indent=2)

    update_run_status(bench_dir, dataset, "completed")
    logger.success(f"BM25 benchmark complete: {bench_dir}")
    logger.info(f"  MRR={summary['mean_mrr']:.4f}  AP={summary['mean_ap']:.4f}  "
                f"P@1={summary['mean_p@1']:.4f}  P@2={summary['mean_p@2']:.4f}")

    return bench_dir


def cli_main():
    parser = argparse.ArgumentParser(
        description="BM25 baseline benchmark for Semantic Folding pipeline",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Existing Phase 1 run directory (needs corpus.txt)")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-end", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    bench_dir = run_bm25_benchmark(
        dataset=args.dataset,
        jsonl_path=args.jsonl,
        run_dir=args.run_dir,
        query_start=args.query_start,
        query_end=args.query_end,
        top_k=args.top_k,
    )
    if bench_dir is None:
        sys.exit(1)
    print(f"\nBM25_OK:{bench_dir}")


if __name__ == "__main__":
    cli_main()
