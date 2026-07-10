#!/usr/bin/env python3
"""
BM25 Baseline for MuSiQue Benchmark — evaluate BM25 on the same data as Semantic Folding.

Usage:
    .venv/scripts/python semantic_folding/dataset_benchmark/musique/b25_baseline.py

Output:
    outputs/musique_benchmark/baselines/bm25_<timestamp>/
        - results_log.csv
        - summary.json
        - bm25_report.md
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import get_logger
logger = get_logger("bm25_baseline")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "HippoRAG2" / "dataset" / "musique"
BASELINE_BASE = PROJECT_ROOT / "outputs" / "musique_benchmark" / "baselines"

BM25_PARAM_CACHE = {}

def load_musique_entries(split: str = "dev") -> List[dict]:
    fname = f"musique_full_v1.0_{split}.jsonl"
    path = DATASET_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"MuSiQue dataset not found: {path}")
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info(f"Loaded {len(entries)} entries from {fname}")
    return entries


def build_combined_corpus(entries: List[dict], start: int, end: int):
    seen = {}
    corpus_lines = []
    query_doc_map = {}
    query_gold = {}
    next_id = 0
    for q_idx in range(start, end):
        entry = entries[q_idx]
        doc_ids = []
        gold_ids = []
        for p in entry["paragraphs"]:
            key = (p["title"], p["paragraph_text"])
            if key not in seen:
                gid = f"doc_{next_id:06d}"
                seen[key] = gid
                corpus_lines.append(f"{gid}, {p['title']} {p['paragraph_text']}")
                next_id += 1
            else:
                gid = seen[key]
            doc_ids.append(gid)
            if p.get("is_supporting", False):
                gold_ids.append(gid)
        query_doc_map[str(q_idx)] = doc_ids
        if gold_ids:
            query_gold[str(q_idx)] = gold_ids
    logger.info(f"Combined corpus: {len(corpus_lines)} unique paragraphs across {end - start} queries")
    return corpus_lines, query_doc_map, query_gold


def tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def compute_metrics(retrieved: List[Tuple[str, float]], relevant: List[str],
                    top_k_list: List[int] = None) -> dict:
    if top_k_list is None:
        top_k_list = [1, 2, 3, 5]
    retrieved_ids = [doc_id for doc_id, _ in retrieved]
    rel_set = set(relevant)

    found_at = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            found_at = rank
            break
    mrr = 1.0 / found_at if found_at > 0 else 0.0

    ap = 0.0
    hits = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            hits += 1
            ap += hits / rank
    ap /= len(relevant) if relevant else 1

    metrics = {"mrr": mrr, "ap": ap, "found_at": found_at}
    for k in top_k_list:
        retrieved_k = retrieved_ids[:k]
        rel_k = sum(1 for d in retrieved_k if d in rel_set)
        metrics[f"p@{k}"] = rel_k / k
        metrics[f"r@{k}"] = rel_k / len(relevant) if relevant else 0.0

    for k in top_k_list:
        dcg_k = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in rel_set:
                dcg_k += 1.0 / (rank + 1).bit_length()
        num_rel = min(len(relevant), k)
        idcg_k = sum(1.0 / (i + 1).bit_length() for i in range(num_rel))
        metrics[f"ndcg@{k}"] = dcg_k / idcg_k if idcg_k > 0 else 0.0
    return metrics


def prepare_bm25_index(corpus_lines: List[str], bm25_params: dict = None) -> Tuple:
    from rank_bm25 import BM25Okapi
    bm25_params = bm25_params or {}
    k1 = bm25_params.get("k1", 1.5)
    b = bm25_params.get("b", 0.75)

    doc_ids = []
    tokenized_docs = []
    for line in corpus_lines:
        doc_id, text = line.split(",", 1)
        doc_ids.append(doc_id.strip())
        tokenized_docs.append(tokenize(text))

    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    logger.info(f"BM25 index built: {len(doc_ids)} docs (k1={k1}, b={b})")
    return bm25, doc_ids


def run_bm25_benchmark(entries: List[dict], query_start: int, query_end: int,
                       bm25_params: dict = None) -> Path:
    if bm25_params is None:
        bm25_params = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASELINE_BASE / f"bm25_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir = out_dir / "per_query"
    per_query_dir.mkdir(exist_ok=True)

    corpus_lines, query_doc_map, query_gold = build_combined_corpus(
        entries, query_start, query_end
    )

    with open(out_dir / "corpus.txt", "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")
    with open(out_dir / "query_doc_map.json", "w") as f:
        json.dump(query_doc_map, f, indent=2)
    with open(out_dir / "query_gold.json", "w") as f:
        json.dump(query_gold, f, indent=2)

    bm25, doc_ids = prepare_bm25_index(corpus_lines, bm25_params)
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    all_metrics = []
    results_log = out_dir / "results_log.csv"
    failed = 0

    for i, q_idx in enumerate(range(query_start, query_end)):
        q_idx_str = str(q_idx)
        entry = entries[q_idx]
        query_text = entry["question"]
        candidate_ids = query_doc_map.get(q_idx_str, [])
        gold_ids = query_gold.get(q_idx_str, [])

        if not gold_ids:
            logger.debug(f"  [{q_idx}] no gold passages, skipping")
            continue

        query_tokens = tokenize(query_text)
        full_scores = bm25.get_scores(query_tokens)

        candidate_results = []
        for cid in candidate_ids:
            idx = doc_id_to_idx.get(cid)
            if idx is not None:
                candidate_results.append((cid, float(full_scores[idx])))
        candidate_results.sort(key=lambda x: x[1], reverse=True)

        query_out_dir = per_query_dir / f"{q_idx:04d}"
        query_out_dir.mkdir(exist_ok=True)
        with open(query_out_dir / "filtered_results.json", "w") as f:
            json.dump({
                "query_idx": q_idx,
                "query": query_text,
                "gold": gold_ids,
                "candidates": candidate_ids,
                "filtered_ranked": candidate_results,
                "elapsed_s": 0,
            }, f, indent=2)

        metrics = compute_metrics(candidate_results, gold_ids,
                                  top_k_list=[1, 2, 3, 5, 10, 20])
        all_metrics.append(metrics)
        logger.info(f"  [{q_idx:04d}] MRR={metrics['mrr']:.3f} AP={metrics['ap']:.3f} "
                    f"P@2={metrics['p@2']:.3f}")

        with open(results_log, "a", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            if i == 0:
                writer.writerow(["query_idx", "query", "mrr", "ap", "p@1", "p@2",
                                 "p@3", "p@5", "r@2", "ndcg@2", "found_at", "elapsed_s"])
            writer.writerow([
                q_idx, query_text[:60],
                f"{metrics['mrr']:.4f}", f"{metrics['ap']:.4f}",
                f"{metrics['p@1']:.4f}", f"{metrics['p@2']:.4f}",
                f"{metrics['p@3']:.4f}", f"{metrics['p@5']:.4f}",
                f"{metrics['r@2']:.4f}", f"{metrics['ndcg@2']:.4f}",
                metrics.get("found_at", "none"), "0.0",
            ])

    if all_metrics:
        agg = defaultdict(list)
        for m in all_metrics:
            for k, v in m.items():
                agg[k].append(v)
        summary = {"num_queries": len(all_metrics), "failed": failed}
        for k, vals in agg.items():
            summary[f"mean_{k}"] = sum(vals) / len(vals)
            summary[f"min_{k}"] = min(vals)
            summary[f"max_{k}"] = max(vals)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.success(f"BM25 benchmark complete - {len(all_metrics)} queries, "
                       f"mean MRR={summary['mean_mrr']:.4f}, AP={summary['mean_ap']:.4f}")
    else:
        logger.warning("No metrics collected")

    generate_report(out_dir)
    return out_dir


def generate_report(out_dir: Path):
    report_path = out_dir / "bm25_report.md"
    with open(out_dir / "summary.json") as f:
        summary = json.load(f)
    per_query = sorted(out_dir.glob("per_query/[0-9]*"))

    report_lines = [
        "# BM25 Baseline — MuSiQue Benchmark Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Output:** `{out_dir.name}`\n",
        "---\n",
        "## BM25 Parameters\n",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| `k1` | 1.5 |",
        "| `b` | 0.75 |",
        "| Tokenization | lowercase + `[a-zA-Z0-9]+` |\n",
        "---\n",
        "## Aggregate Results\n",
        "| Metric | Mean | Min | Max |",
        "|--------|------|-----|-----|",
    ]
    for metric in ["mrr", "ap", "p@1", "p@2", "p@3", "p@5", "p@10", "p@20",
                    "r@2", "r@5", "ndcg@2", "ndcg@5"]:
        mean_k = f"mean_{metric}"; min_k = f"min_{metric}"; max_k = f"max_{metric}"
        if mean_k in summary:
            report_lines.append(
                f"| **{metric.upper()}** | {summary[mean_k]:.4f} | "
                f"{summary[min_k]:.4f} | {summary[max_k]:.4f} |"
            )
    report_lines += [
        f"\n**Queries evaluated:** {summary.get('num_queries', '?')}",
        f"\n**Failed:** {summary.get('failed', 0)}\n",
        "---\n",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.success(f"Report saved -> {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BM25 Baseline for MuSiQue Benchmark")
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    entries = load_musique_entries(args.split)
    end = args.start + args.max_queries
    end = min(end, len(entries))

    bm25_params = {"k1": args.k1, "b": args.b}
    out_dir = run_bm25_benchmark(entries, args.start, end, bm25_params)
    print(f"\nDone. Results: {out_dir}")


if __name__ == "__main__":
    main()
