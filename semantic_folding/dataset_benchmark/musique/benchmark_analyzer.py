#!/usr/bin/env python3
"""
benchmark_analyzer.py — Deep analysis of completed MuSiQue benchmark results.

Usage:
    python semantic_folding/dataset_benchmark/musique/benchmark_analyzer.py
    (interactive -- picks the most recent benchmark)

    python semantic_folding/dataset_benchmark/musique/benchmark_analyzer.py --benchmark-dir <path>
    (specific benchmark directory)

Produces:
    - Per-metric histograms (JSON)
    - Failure analysis (queries where no gold found)
    - Metric breakdown by number of hops (if available)
    - Comparison with last benchmark in the same series
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import get_logger
logger = get_logger("bench_analyzer")

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "musique_benchmark" / "benchmarks"
RUNS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "musique_benchmark" / "runs"


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def get_last_benchmark_dir() -> Path:
    """Return the most recent benchmark directory."""
    if not BENCHMARKS_DIR.exists():
        return None
    dirs = sorted(BENCHMARKS_DIR.iterdir(), reverse=True)
    return dirs[0] if dirs else None


def load_results_csv(bench_dir: Path) -> List[dict]:
    csv_path = bench_dir / "results_log.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_summary(bench_dir: Path) -> dict:
    summary_path = bench_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        return json.load(f)


def load_per_query_metrics(bench_dir: Path) -> List[dict]:
    per_query = sorted(bench_dir.glob("per_query/[0-9]*"))
    results = []
    for qd in per_query:
        fpath = qd / "filtered_results.json"
        if fpath.exists():
            with open(fpath) as f:
                results.append(json.load(f))
    return results


def analyze_benchmark(bench_dir: Path) -> dict:
    logger.info(f"Analyzing: {bench_dir.name}")

    results = load_per_query_metrics(bench_dir)
    csv_rows = load_results_csv(bench_dir)
    summary = load_summary(bench_dir)

    if not results:
        logger.error("No per-query results found")
        return None

    analysis = {
        "benchmark": bench_dir.name,
        "generated_at": datetime.now().isoformat(),
        "num_queries": len(results),
        "summary": summary,
        "metrics_distribution": {},
        "found_at_distribution": defaultdict(int),
        "failures": [],
        "top_performers": [],
        "by_elapsed": {},
    }

    # Found-at distribution
    mrr_values = []
    ap_values = []
    p1_values = []
    p2_values = []

    for qd in results:
        q_idx = qd["query_idx"]
        gold = qd.get("gold", [])
        ranked = qd.get("filtered_ranked", [])

        retrieved_ids = [doc_id for doc_id, _ in ranked]
        rel_set = set(gold)

        found_at = 0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in rel_set:
                found_at = rank
                break
        analysis["found_at_distribution"][found_at] += 1

        # MRR
        mrr = 1.0 / found_at if found_at > 0 else 0.0
        mrr_values.append(mrr)

        # AP
        ap = 0.0
        hits = 0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in rel_set:
                hits += 1
                ap += hits / rank
        ap /= len(gold) if gold else 1
        ap_values.append(ap)

        # P@1, P@2
        p1 = 1.0 if any(doc_id in rel_set for doc_id in retrieved_ids[:1]) else 0.0
        p2 = sum(1 for d in retrieved_ids[:2] if d in rel_set) / 2
        p1_values.append(p1)
        p2_values.append(p2)

        if found_at == 0:
            analysis["failures"].append({
                "query_idx": q_idx,
                "query": qd.get("query", "")[:80],
                "num_gold": len(gold),
                "num_candidates": len(qd.get("candidates", [])),
            })

        if mrr >= 0.9:
            analysis["top_performers"].append({
                "query_idx": q_idx,
                "query": qd.get("query", "")[:60],
                "mrr": mrr,
                "ap": ap,
            })

    # Distribution stats
    for name, vals in [("mrr", mrr_values), ("ap", ap_values),
                       ("p@1", p1_values), ("p@2", p2_values)]:
        if vals:
            analysis["metrics_distribution"][name] = {
                "mean": sum(vals) / len(vals),
                "median": sorted(vals)[len(vals) // 2],
                "min": min(vals),
                "max": max(vals),
                "std": (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals))**0.5,
                "num_zero": sum(1 for v in vals if v == 0),
                "num_perfect": sum(1 for v in vals if v >= 1.0),
            }

    return analysis


def print_analysis(analysis: dict):
    if not analysis:
        return

    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}  MuSiQue Benchmark Analysis{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {analysis['benchmark']}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")

    dist = analysis.get("metrics_distribution", {})
    for metric in ["mrr", "ap", "p@1", "p@2"]:
        if metric not in dist:
            continue
        d = dist[metric]
        bar = _bar(d["mean"], 20)
        print(f"  {Colors.CYAN}{metric.upper():>6}{Colors.ENDC}  "
              f"{bar}  {d['mean']:.4f}  "
              f"(med={d['median']:.3f}, min={d['min']:.3f}, max={d['max']:.3f}, "
              f"std={d['std']:.4f})")
        print(f"         zero={d['num_zero']}, perfect={d['num_perfect']}")

    print(f"\n{Colors.BOLD}Found-at Rank Distribution:{Colors.ENDC}")
    fa = analysis.get("found_at_distribution", {})
    total = sum(fa.values()) or 1
    for rank in sorted(fa.keys()):
        pct = fa[rank] / total * 100
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        label = "not found" if rank == 0 else f"rank {rank}"
        print(f"  {Colors.YELLOW}{label:>10}{Colors.ENDC}  "
              f"{bar}  {fa[rank]:>3}/{total} ({pct:.0f}%)")

    failures = analysis.get("failures", [])
    print(f"\n{Colors.RED}{Colors.BOLD}Failures (no gold found): {len(failures)}{Colors.ENDC}")
    for f in failures[:10]:
        print(f"  [{f['query_idx']:04d}] {f['query']} ({f['num_gold']} gold docs)")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")

    top = analysis.get("top_performers", [])
    print(f"\n{Colors.GREEN}{Colors.BOLD}Top Performers (MRR=1.0): {len(top)}{Colors.ENDC}")
    for t in top[:5]:
        print(f"  [{t['query_idx']:04d}] {t['query']} (AP={t['ap']:.4f})")
    if len(top) > 5:
        print(f"  ... and {len(top) - 5} more")

    print()


def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "#" * filled + "." * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Analyze MuSiQue benchmark results")
    parser.add_argument("--benchmark-dir", type=Path, default=None,
                        help="Benchmark directory to analyze (default: most recent)")
    args = parser.parse_args()

    bench_dir = args.benchmark_dir or get_last_benchmark_dir()
    if not bench_dir:
        print(f"{Colors.RED}No benchmarks found in {BENCHMARKS_DIR}{Colors.ENDC}")
        sys.exit(1)

    analysis = analyze_benchmark(bench_dir)
    if analysis:
        print_analysis(analysis)

        # Save analysis JSON
        out_path = bench_dir / "analysis.json"
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.success(f"Analysis saved → {out_path}")


if __name__ == "__main__":
    main()
