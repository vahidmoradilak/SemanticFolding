#!/usr/bin/env python3
"""
Compare BM25 vs Semantic Folding on MuSiQue benchmark results.

Usage:
    .venv/scripts/python semantic_folding/dataset_benchmark/musique/compare_baselines.py

Output:
    outputs/musique_benchmark/comparison_<timestamp>/comparison_report.md
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import get_logger
logger = get_logger("comparison")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
BENCHMARKS_DIR = PROJECT_ROOT / "outputs" / "musique_benchmark"
BASELINES_DIR = BENCHMARKS_DIR / "baselines"
BENCHMARKS_RESULTS_DIR = BENCHMARKS_DIR / "benchmarks"
COMPARISON_DIR = BENCHMARKS_DIR / "comparisons"

def find_latest(pattern_dir):
    if not pattern_dir.exists():
        return None
    dirs = sorted(pattern_dir.iterdir(), reverse=True)
    return dirs[0] if dirs else None

def load_summary(path):
    with open(path / "summary.json") as f:
        return json.load(f)

def load_per_query(path, method="sf"):
    per_query = sorted(path.glob("per_query/[0-9]*"))
    results = {}
    for qd in per_query:
        fpath = qd / "filtered_results.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            idx = data["query_idx"]
            ranked = data.get("filtered_ranked", [])
            gold = data.get("gold", [])
            rel_set = set(gold)
            # Compute per-query metrics inline
            mrr = ap = 0.0
            found_at = 0
            hits = 0
            for rank, (doc_id, _) in enumerate(ranked, 1):
                if doc_id in rel_set:
                    if found_at == 0:
                        found_at = rank
                        mrr = 1.0 / rank
                    hits += 1
                    ap += hits / rank
            if gold:
                ap /= len(gold)
            results[idx] = {
                "mrr": mrr, "ap": ap, "found_at": found_at,
                "gold": gold, "ranked_ids": [d for d, _ in ranked],
                "method": method,
            }
    return results

def compare():
    sf_dir = find_latest(BENCHMARKS_RESULTS_DIR)
    bm25_dir = find_latest(BASELINES_DIR)

    if not sf_dir or not bm25_dir:
        logger.error(f"Missing results: SF={sf_dir}, BM25={bm25_dir}")
        return

    sf_summary = load_summary(sf_dir)
    bm25_summary = load_summary(bm25_dir)

    sf_per_query = load_per_query(sf_dir, "sf")
    bm25_per_query = load_per_query(bm25_dir, "bm25")

    # Find common queries
    common = sorted(set(sf_per_query.keys()) & set(bm25_per_query.keys()))
    logger.info(f"SF: {len(sf_per_query)} queries, BM25: {len(bm25_per_query)} queries, Common: {len(common)}")

    # Aggregate comparison over common queries
    sf_wins_mrr = bm25_wins_mrr = tie_mrr = 0
    sf_wins_ap = bm25_wins_ap = tie_ap = 0
    both_fail = 0

    per_query_compare = []
    for q_idx in common:
        s = sf_per_query[q_idx]
        b = bm25_per_query[q_idx]
        delta_mrr = s["mrr"] - b["mrr"]
        delta_ap = s["ap"] - b["ap"]
        if delta_mrr > 0: sf_wins_mrr += 1
        elif delta_mrr < 0: bm25_wins_mrr += 1
        else: tie_mrr += 1
        if delta_ap > 0: sf_wins_ap += 1
        elif delta_ap < 0: bm25_wins_ap += 1
        else: tie_ap += 1
        if s["found_at"] == 0 and b["found_at"] == 0:
            both_fail += 1
        per_query_compare.append({
            "idx": q_idx,
            "sf_mrr": s["mrr"], "bm25_mrr": b["mrr"],
            "sf_ap": s["ap"], "bm25_ap": b["ap"],
            "sf_found": s["found_at"], "bm25_found": b["found_at"],
        })

    # Generate report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = COMPARISON_DIR / f"comparison_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline Comparison: Semantic Folding vs BM25\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**SF Benchmark:** `{sf_dir.name}`",
        f"\n**BM25 Benchmark:** `{bm25_dir.name}`",
        f"**Common queries compared:** {len(common)}\n",
        "---\n",
        "## Aggregate Results (Common Queries)\n",
        "| Metric | Semantic Folding | BM25 | Δ | Winner |",
        "|--------|:-:|:-:|:-:|:-:|",
    ]
    metrics = [("mrr", "MRR"), ("ap", "AP"), ("p@1", "P@1"),
               ("p@2", "P@2"), ("p@5", "P@5"), ("r@2", "R@2"), ("r@5", "R@5")]
    for mkey, mlabel in metrics:
        sf_k = f"mean_{mkey}"
        bm25_k = f"mean_{mkey}"
        sf_v = sf_summary.get(sf_k, 0)
        bm25_v = bm25_summary.get(bm25_k, 0)
        delta = sf_v - bm25_v
        delta_s = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        winner = "SF" if delta > 0 else ("BM25" if delta < 0 else "Tie")
        lines.append(f"| **{mlabel}** | {sf_v:.4f} | {bm25_v:.4f} | {delta_s} | {winner} |")

    lines += [
        f"\n### Win Counts (MRR)\n",
        f"- **SF wins:** {sf_wins_mrr}/{len(common)} ({sf_wins_mrr/len(common)*100:.1f}%)",
        f"- **BM25 wins:** {bm25_wins_mrr}/{len(common)} ({bm25_wins_mrr/len(common)*100:.1f}%)",
        f"- **Ties:** {tie_mrr}/{len(common)}",
        f"\n**Queries where both failed:** {both_fail}/{len(common)}\n",
        "---\n",
        "## Per-Query Comparison\n",
        "| # | SF MRR | BM25 MRR | SF AP | BM25 AP | Winner |",
        "|---|:-:|:-:|:-:|:-:|:-:|",
    ]
    for pq in sorted(per_query_compare, key=lambda x: x["idx"]):
        delta_mrr = pq["sf_mrr"] - pq["bm25_mrr"]
        winner = "SF" if delta_mrr > 0 else ("BM25" if delta_mrr < 0 else "Tie")
        lines.append(
            f"| {pq['idx']:04d} | {pq['sf_mrr']:.3f} | {pq['bm25_mrr']:.3f} | "
            f"{pq['sf_ap']:.3f} | {pq['bm25_ap']:.3f} | {winner} |"
        )

    report = "\n".join(lines)
    report_path = out_dir / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.success(f"Comparison report -> {report_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPARISON: SEMANTIC FOLDING vs BM25")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'SF':<12} {'BM25':<12} {'Delta':<12} {'Winner':<8}")
    print(f"{'-'*60}")
    for mkey, mlabel in metrics:
        sf_v = sf_summary.get(f"mean_{mkey}", 0)
        bm25_v = bm25_summary.get(f"mean_{mkey}", 0)
        delta = sf_v - bm25_v
        d = f"{delta:+.4f}"
        w = "SF" if delta > 0 else ("BM25" if delta < 0 else "Tie")
        print(f"{mlabel:<20} {sf_v:<12.4f} {bm25_v:<12.4f} {d:<12} {w:<8}")
    print(f"{'='*60}")
    print(f"SF wins MRR: {sf_wins_mrr}/{len(common)} ({sf_wins_mrr/len(common)*100:.1f}%)")
    print(f"BM25 wins MRR: {bm25_wins_mrr}/{len(common)} ({bm25_wins_mrr/len(common)*100:.1f}%)")
    print(f"Both fail: {both_fail}/{len(common)}")

if __name__ == "__main__":
    compare()
