#!/usr/bin/env python3
"""
MuSiQue Benchmark for Semantic Folding — Interactive TUI

Three-Phase Design:
  Phase 1 (index)   — Build combined corpus from unique paragraphs, run Steps 1-5 once
  Phase 2 (benchmark) — Run Step 6 per query against pre-built fingerprints
  Phase 3 (report)   — Generate markdown report from benchmark results

Usage:
    # Interactive TUI (default)
    python semantic_folding/dataset_benchmark/musique/run_benchmark.py

    # CLI mode (for automation)
    python semantic_folding/dataset_benchmark/musique/run_benchmark.py
        --mode index --split dev --max-queries 100 --grid-size 64 --benchmark

See README.md for full documentation.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import get_logger
logger = get_logger("musique_bench")

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # go up: musique -> dataset_benchmark -> semantic_folding -> knowledge-graph-builder
DATASET_DIR = PROJECT_ROOT / "data" / "HippoRAG2" / "dataset" / "musique"
BENCHMARK_BASE = PROJECT_ROOT / "outputs" / "musique_benchmark"
RUNS_DIR = BENCHMARK_BASE / "runs"
BENCHMARKS_DIR = BENCHMARK_BASE / "benchmarks"
SEMANTIC_FOLDING = PROJECT_ROOT / "semantic_folding"
RUNS_REGISTRY = SCRIPT_DIR / "runs" / "registry.yml"

STEP_SCRIPTS = {
    1: SEMANTIC_FOLDING / "phrase_extractor.py",
    2: SEMANTIC_FOLDING / "term_context.py",
    3: SEMANTIC_FOLDING / "semantic_space.py",
    4: SEMANTIC_FOLDING / "phrase_fingerprints.py",
    5: SEMANTIC_FOLDING / "doc_fingerprints.py",
    6: SEMANTIC_FOLDING / "query_processor.py",
}

PIPELINE_DEFAULTS = {
    "grid_size": 64,
    "spreading_steps": 1,
    "top_k": 5,
    "weighting": "idf",
    "top_percent": 0.10,
    "smoothing_sigma": 1.5,
    "keep_verbs": True,
    "min_word_length": 3,
    "min_freq": 1,
    "morton": True,
    "tsne_perplexity": 30,
    "tsne_iter": 1000,
}

# ============================================================================
# Terminal Colors
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# ============================================================================
# Registry Management
# ============================================================================

def load_registry() -> dict:
    RUNS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if not RUNS_REGISTRY.exists():
        empty = {"runs": {}}
        with open(RUNS_REGISTRY, "w") as f:
            yaml.dump(empty, f, default_flow_style=False)
        return empty
    with open(RUNS_REGISTRY, "r") as f:
        return yaml.safe_load(f) or {"runs": {}}


def save_registry(registry: dict):
    RUNS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_REGISTRY, "w") as f:
        yaml.dump(registry, f, default_flow_style=False)


def register_run(run_dir: Path, run_type: str, params: dict, status: str = "created"):
    registry = load_registry()
    run_id = run_dir.name
    registry["runs"][run_id] = {
        "type": run_type,  # "index" or "benchmark"
        "path": str(run_dir.resolve()),
        "created_at": datetime.now().isoformat(),
        "status": status,
        "params": {k: str(v) for k, v in params.items()},
    }
    save_registry(registry)


def update_run_status(run_dir: Path, status: str):
    registry = load_registry()
    run_id = run_dir.name
    if run_id in registry["runs"]:
        registry["runs"][run_id]["status"] = status
        save_registry(registry)


def get_registered_runs(run_type: str = None) -> List[Tuple[str, dict]]:
    registry = load_registry()
    runs = []
    for run_id, data in registry["runs"].items():
        if run_type is None or data.get("type") == run_type:
            runs.append((run_id, data))
    return sorted(runs, reverse=True)


# ============================================================================
# Data loading
# ============================================================================

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


# ============================================================================
# Run step helper
# ============================================================================

def run_step(script: Path, args: List[str], workdir: Path, step_name: str,
             timeout: int = 600) -> bool:
    args = [a for a in args if a]
    cmd = [sys.executable, str(script)] + args
    logger.info(f"  [{step_name}] starting...")
    try:
        result = subprocess.run(
            cmd, cwd=str(workdir), capture_output=True, text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr[-800:].replace("\n", " | ")
            logger.error(f"  [{step_name}] FAILED (rc={result.returncode}): {stderr_tail}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"  [{step_name}] TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"  [{step_name}] ERROR: {e}")
        return False


# ============================================================================
# Phase 1 — Index
# ============================================================================

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


def phase1_index(entries: List[dict], start: int, end: int, params: dict) -> Optional[Path]:
    """Run Phase 1: build combined corpus, run steps 1-5, return run directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build combined corpus
    corpus_lines, query_doc_map, query_gold = build_combined_corpus(entries, start, end)
    corpus_path = run_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    # Save mappings
    with open(run_dir / "query_doc_map.json", "w") as f:
        json.dump(query_doc_map, f, indent=2)
    with open(run_dir / "query_gold.json", "w") as f:
        json.dump(query_gold, f, indent=2)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump({
            "num_queries": end - start,
            "query_start": start,
            "query_end": end,
            "num_docs": len(corpus_lines),
            "created_at": ts,
        }, f, indent=2)

    # Save config YAML
    run_config = {"phase1": {
        "mode": "index",
        "timestamp": ts,
        "query_start": start,
        "query_end": end,
        "num_queries": end - start,
        "num_docs": len(corpus_lines),
    }, "pipeline": {k: v for k, v in params.items()}}
    with open(run_dir / "config.yml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    register_run(run_dir, "index", params, "indexing")

    # Run steps 1-5
    logger.info(f"Index run: {run_dir.name} ({len(corpus_lines)} docs, {end - start} queries)")

    # Step 1
    out = run_dir / "extracted_phrases"
    ok = run_step(STEP_SCRIPTS[1], [
        "--corpus", str(corpus_path), "--output", str(out),
        "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
        "--min-freq", str(params["min_freq"]),
    ], PROJECT_ROOT, "Step 1 phrase_extractor")
    if not ok:
        update_run_status(run_dir, "failed_step1")
        return None

    # Step 2
    out = run_dir / "term_context_matrix"
    ok = run_step(STEP_SCRIPTS[2], [
        "--vocab", str(run_dir / "extracted_phrases" / "vocabulary.csv"),
        "--mapping", str(run_dir / "extracted_phrases" / "phrase_to_contexts.json"),
        "--corpus", str(corpus_path), "--output", str(out),
    ], PROJECT_ROOT, "Step 2 term_context")
    if not ok:
        update_run_status(run_dir, "failed_step2")
        return None

    # Step 3 — t-SNE
    out = run_dir / "semantic_space"
    ok = run_step(STEP_SCRIPTS[3], [
        "--matrix", str(run_dir / "term_context_matrix" / "term_context_matrix.npz"),
        "--metadata", str(run_dir / "term_context_matrix" / "term_context_matrix.json"),
        "--output", str(out),
        "--grid-size", str(params["grid_size"]),
        "--perplexity", str(params["tsne_perplexity"]),
        "--tsne-iter", str(params["tsne_iter"]),
    ], PROJECT_ROOT, "Step 3 semantic_space", timeout=900)
    if not ok:
        update_run_status(run_dir, "failed_step3")
        return None

    # Step 4
    out = run_dir / "phrase_fingerprints"
    morton_flag = "--morton" if params["morton"] else "--no-morton"
    ok = run_step(STEP_SCRIPTS[4], [
        "--coordinates", str(run_dir / "semantic_space" / "context_coordinates.json"),
        "--metadata", str(run_dir / "term_context_matrix" / "term_context_matrix.json"),
        "--output", str(out),
        "--grid-size", str(params["grid_size"]),
        "--smoothing-sigma", str(params["smoothing_sigma"]),
        morton_flag,
    ], PROJECT_ROOT, "Step 4 phrase_fingerprints")
    if not ok:
        update_run_status(run_dir, "failed_step4")
        return None

    # Step 5
    out = run_dir / "doc_fingerprints"
    ok = run_step(STEP_SCRIPTS[5], [
        "--corpus", str(corpus_path),
        "--fingerprints", str(run_dir / "phrase_fingerprints"),
        "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
        "--output", str(out),
        "--grid-size", str(params["grid_size"]),
        "--top-percent", str(params["top_percent"]),
        "--normalize-method", "l2",
        "--min-word-length", str(params["min_word_length"]),
        "--smoothing-sigma", str(params["smoothing_sigma"]),
        "--min-peak-distance", "2",
        morton_flag,
    ], PROJECT_ROOT, "Step 5 doc_fingerprints")
    if not ok:
        update_run_status(run_dir, "failed_step5")
        return None

    update_run_status(run_dir, "completed")
    logger.success(f"Index phase complete -> {run_dir}")
    return run_dir


# ============================================================================
# Phase 2 — Benchmark
# ============================================================================

def filter_results_to_candidates(full_results: List[list], candidate_ids: List[str]) -> List[Tuple[str, float]]:
    cand_set = set(candidate_ids)
    filtered = [(doc_id, score) for doc_id, score in full_results if doc_id in cand_set]
    return filtered


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


def load_query_results(result_path: Path) -> List[dict]:
    with open(result_path, "r") as f:
        return json.load(f)


def phase2_benchmark(run_dir: Path, entries: List[dict], query_start: int,
                     query_end: int, params: dict) -> Optional[Path]:
    """Run Phase 2: process queries against pre-built fingerprints."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = BENCHMARKS_DIR / f"benchmark_{ts}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir = bench_dir / "per_query"
    per_query_dir.mkdir(exist_ok=True)

    # Load mappings
    with open(run_dir / "query_doc_map.json") as f:
        query_doc_map = json.load(f)
    with open(run_dir / "query_gold.json") as f:
        query_gold = json.load(f)

    # Save benchmark config
    bench_config = {
        "phase2": {
            "mode": "benchmark",
            "timestamp": ts,
            "run_dir": str(run_dir),
            "query_start": query_start,
            "query_end": query_end,
        },
        "pipeline": {k: v for k, v in params.items()},
    }
    with open(bench_dir / "config.yml", "w") as f:
        yaml.dump(bench_config, f, default_flow_style=False)

    register_run(bench_dir, "benchmark", params, "running")
    logger.info(f"Benchmark: {bench_dir.name} - queries {query_start}-{query_end - 1} against run {run_dir.name}")

    all_metrics = []
    results_log = bench_dir / "results_log.csv"
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

        query_out_dir = per_query_dir / f"{q_idx:04d}"
        query_out_dir.mkdir(exist_ok=True)

        cand_path = query_out_dir / "candidate_docs.json"
        with open(cand_path, "w") as f:
            json.dump({"candidate_ids": candidate_ids, "gold_ids": gold_ids}, f, indent=2)

        result_json = query_out_dir / "query_results.json"
        t0 = time.time()
        ok = run_step(STEP_SCRIPTS[6], [
            "--query", query_text,
            "--fingerprints", str(run_dir / "phrase_fingerprints"),
            "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
            "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
            "--grid-size", str(params["grid_size"]),
            "--top-k", str(params["top_k"]),
            "--weighting", params["weighting"],
            "--spreading-steps", str(params["spreading_steps"]),
            "--output", str(result_json),
            "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
        ], PROJECT_ROOT, "Step 6 query_processor", timeout=120)
        elapsed = time.time() - t0

        if not ok:
            logger.error(f"  [{q_idx}] query processor FAILED ({elapsed:.0f}s)")
            failed += 1
            continue

        raw_results = load_query_results(result_json)
        full_results = raw_results[0]["results"] if raw_results else []
        candidate_results = filter_results_to_candidates(full_results, candidate_ids)

        with open(query_out_dir / "filtered_results.json", "w") as f:
            json.dump({
                "query_idx": q_idx,
                "query": query_text,
                "gold": gold_ids,
                "candidates": candidate_ids,
                "filtered_ranked": [(doc_id, float(score)) for doc_id, score in candidate_results],
                "full_top10": [(doc_id, float(score)) for doc_id, score in full_results[:10]],
                "elapsed_s": round(elapsed, 1),
            }, f, indent=2)

        metrics = compute_metrics(candidate_results, gold_ids,
                                  top_k_list=[1, 2, 3, 5, params["top_k"]])
        all_metrics.append(metrics)

        logger.info(f"  [{q_idx:04d}] MRR={metrics['mrr']:.3f} AP={metrics['ap']:.3f} "
                    f"P@2={metrics['p@2']:.3f} [{elapsed:.0f}s]")

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
                metrics.get("found_at", "none"), f"{elapsed:.1f}",
            ])

    if all_metrics:
        agg = defaultdict(list)
        for m in all_metrics:
            for k, v in m.items():
                agg[k].append(v)

        summary = {
            "num_queries": len(all_metrics),
            "failed": failed,
        }
        for k, vals in agg.items():
            summary[f"mean_{k}"] = sum(vals) / len(vals)
            summary[f"min_{k}"] = min(vals)
            summary[f"max_{k}"] = max(vals)

        with open(bench_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        update_run_status(bench_dir, "completed")
        logger.success(f"Benchmark complete - {len(all_metrics)} queries, "
                       f"mean MRR={summary['mean_mrr']:.4f}, AP={summary['mean_ap']:.4f}")
        return bench_dir
    else:
        update_run_status(bench_dir, "failed")
        logger.warning("No metrics collected")
        return None


# ============================================================================
# Phase 3 — Report
# ============================================================================

def phase3_report(bench_dir: Path) -> None:
    report_path = bench_dir / "benchmark_report.md"

    with open(bench_dir / "config.yml") as f:
        config = yaml.safe_load(f)
    with open(bench_dir / "summary.json") as f:
        summary = json.load(f)

    run_dir = Path(config["phase2"]["run_dir"])
    run_config_path = run_dir / "config.yml"
    run_config = {}
    if run_config_path.exists():
        with open(run_config_path) as f:
            run_config = yaml.safe_load(f)

    per_query = sorted(bench_dir.glob("per_query/[0-9]*"))
    queries_data = []
    for qd in per_query:
        fpath = qd / "filtered_results.json"
        if fpath.exists():
            with open(fpath) as f:
                queries_data.append(json.load(f))

    pipe = config.get("pipeline", {})
    report_lines = [
        f"# MuSiQue Benchmark Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Benchmark:** `{bench_dir.name}`",
        f"\n**Run:** `{run_dir.name}`\n",
        f"---\n",
        f"## Configuration\n",
        f"| Parameter | Value |",
        f"|-----------|-------|",
    ]
    for k, v in pipe.items():
        report_lines.append(f"| `{k}` | {v} |")
    report_lines += [
        f"\n| Query range | {config['phase2']['query_start']}-{config['phase2']['query_end'] - 1} |",
        f"| Run docs    | {run_config.get('phase1', {}).get('num_docs', '?')} |\n",
        f"---\n",
        f"## Aggregate Results\n",
        f"| Metric | Mean | Min | Max |",
        f"|--------|------|-----|-----|",
    ]
    for metric in ["mrr", "ap", "p@1", "p@2", "p@3", "p@5", "r@2", "r@5", "ndcg@2", "ndcg@5"]:
        mean_k = f"mean_{metric}"; min_k = f"min_{metric}"; max_k = f"max_{metric}"
        if mean_k in summary:
            report_lines.append(
                f"| **{metric.upper()}** | {summary[mean_k]:.4f} | "
                f"{summary[min_k]:.4f} | {summary[max_k]:.4f} |"
            )
    report_lines += [
        f"\n**Queries evaluated:** {summary.get('num_queries', '?')}",
        f"\n**Failed:** {summary.get('failed', 0)}\n",
        f"---\n",
        f"## Per-Query Results\n",
        f"| # | Query | MRR | AP | P@1 | P@2 | R@2 | NDCG@2 | Time |",
        f"|---|-------|-----|-----|-----|-----|-----|--------|------|",
    ]

    not_found = found_r1 = found_r2 = 0
    for qd in queries_data:
        q_idx = qd["query_idx"]
        query_short = qd["query"][:50]
        gold = qd["gold"]
        ranked = qd.get("filtered_ranked", [])
        m = compute_metrics(ranked, gold, [1, 2, 3, 5])
        report_lines.append(
            f"| {q_idx:04d} | {query_short}... | "
            f"{m['mrr']:.3f} | {m['ap']:.3f} | {m['p@1']:.3f} | "
            f"{m['p@2']:.3f} | {m['r@2']:.3f} | {m['ndcg@2']:.3f} | "
            f"{qd.get('elapsed_s', '?'):>5}s |"
        )
        fa = m.get("found_at", 0)
        if fa == 0:
            not_found += 1
        elif fa <= 2:
            found_r2 += 1
        if fa == 1:
            found_r1 += 1

    report_lines += [
        f"\n### Distribution\n",
        f"\n**Found at rank 1:** {found_r1}/{len(queries_data)}",
        f"\n**Found at rank <= 2:** {found_r1 + found_r2}/{len(queries_data)}",
        f"\n**Not found:** {not_found}/{len(queries_data)}\n",
        f"---\n",
        f"*Report generated by `run_benchmark.py --mode report`*",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.success(f"Report saved -> {report_path}")


# ============================================================================
# Interactive TUI — BenchmarkRunner
# ============================================================================

class BenchmarkRunner:
    """Interactive TUI for MuSiQue Benchmark."""

    def __init__(self):
        self.entries_cache = {}
        self.last_run_dir = None
        self.last_bench_dir = None
        registry = load_registry()
        # Find latest completed runs
        for run_id, data in sorted(registry["runs"].items(), reverse=True):
            if data.get("type") == "index" and data.get("status") == "completed":
                if self.last_run_dir is None:
                    self.last_run_dir = Path(data["path"])
            if data.get("type") == "benchmark" and data.get("status") == "completed":
                if self.last_bench_dir is None:
                    self.last_bench_dir = Path(data["path"])

    # ------------------------------------------------------------------
    # UI Helpers
    # ------------------------------------------------------------------
    def print_header(self, text: str):
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")

    def print_success(self, text: str):
        print(f"{Colors.GREEN}  {text}{Colors.ENDC}")

    def print_error(self, text: str):
        print(f"{Colors.RED}  {text}{Colors.ENDC}")

    def print_warning(self, text: str):
        print(f"{Colors.YELLOW}  {text}{Colors.ENDC}")

    def get_input(self, prompt: str, default: Any = None) -> str:
        if default is not None:
            full_prompt = f"{prompt} [{Colors.YELLOW}{default}{Colors.ENDC}]: "
        else:
            full_prompt = f"{prompt}: "
        value = input(full_prompt).strip()
        return value if value else (str(default) if default is not None else "")

    def get_choice(self, prompt: str, options: List[str]) -> int:
        print(f"\n{Colors.BOLD}{prompt}{Colors.ENDC}")
        for i, option in enumerate(options, 1):
            print(f"  {Colors.CYAN}{i}.{Colors.ENDC} {option}")
        while True:
            try:
                choice = int(input(f"\n{Colors.BOLD}Enter choice (1-{len(options)}): {Colors.ENDC}"))
                if 1 <= choice <= len(options):
                    return choice
                self.print_error(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                self.print_error("Please enter a valid number")

    def confirm(self, prompt: str, default: str = "y") -> bool:
        val = self.get_input(f"{prompt} (y/n)", default)
        return val.lower() == "y"

    # ------------------------------------------------------------------
    # Params collection
    # ------------------------------------------------------------------
    def collect_index_params(self) -> dict:
        params = dict(PIPELINE_DEFAULTS)
        print(f"\n{Colors.BOLD}Index Phase Parameters (Enter to accept default):{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-' * 60}{Colors.ENDC}")

        params["split"] = self.get_input("  Split (train/dev)", "dev")
        params["max_queries"] = int(self.get_input("  Number of queries", "100"))
        params["query_start"] = int(self.get_input("  Query start index", "0"))

        print(f"\n{Colors.CYAN}Pipeline Parameters:{Colors.ENDC}")
        params["grid_size"] = int(self.get_input("  Grid size", str(params["grid_size"])))
        params["spreading_steps"] = int(self.get_input("  Spreading steps", str(params["spreading_steps"])))
        params["top_percent"] = float(self.get_input("  Top percent", str(params["top_percent"])))
        params["weighting"] = self.get_input("  Weighting (uniform/idf/frequency)", params["weighting"])
        params["smoothing_sigma"] = float(self.get_input("  Smoothing sigma", str(params["smoothing_sigma"])))
        morton_str = self.get_input("  Morton encoding (true/false)", "true")
        params["morton"] = morton_str.lower() == "true"
        params["top_k"] = int(self.get_input("  Top K", str(params["top_k"])))
        params["min_word_length"] = int(self.get_input("  Min word length", str(params["min_word_length"])))
        params["min_freq"] = int(self.get_input("  Min phrase frequency", str(params["min_freq"])))
        params["tsne_perplexity"] = int(self.get_input("  t-SNE perplexity", str(params["tsne_perplexity"])))
        params["tsne_iter"] = int(self.get_input("  t-SNE iterations", str(params["tsne_iter"])))

        return params

    def collect_benchmark_params(self, run_dir: Path = None) -> dict:
        params = dict(PIPELINE_DEFAULTS)
        print(f"\n{Colors.BOLD}Benchmark Phase Parameters:{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-' * 60}{Colors.ENDC}")

        params["split"] = self.get_input("  Split (train/dev)", "dev")
        params["query_start"] = int(self.get_input("  Query start index", "0"))
        params["query_end"] = int(self.get_input("  Query end index (exclusive)", "100"))

        if run_dir:
            # Load pipeline params from run config
            run_config_path = run_dir / "config.yml"
            if run_config_path.exists():
                with open(run_config_path) as f:
                    run_cfg = yaml.safe_load(f)
                run_pipe = run_cfg.get("pipeline", {})
                for k in PIPELINE_DEFAULTS:
                    if k in run_pipe:
                        params[k] = run_pipe[k]
                params["morton"] = run_pipe.get("morton", True)
                self.print_success(f"Loaded pipeline params from run config (grid={params['grid_size']})")
        else:
            # Manual entry
            params["grid_size"] = int(self.get_input("  Grid size", str(params["grid_size"])))
            params["spreading_steps"] = int(self.get_input("  Spreading steps", str(params["spreading_steps"])))
            params["top_percent"] = float(self.get_input("  Top percent", str(params["top_percent"])))
            params["weighting"] = self.get_input("  Weighting", params["weighting"])
            params["smoothing_sigma"] = float(self.get_input("  Smoothing sigma", str(params["smoothing_sigma"])))
            morton_str = self.get_input("  Morton encoding", "true")
            params["morton"] = morton_str.lower() == "true"
            params["top_k"] = int(self.get_input("  Top K", str(params["top_k"])))

        return params

    def print_params(self, params: dict, phase: str):
        print(f"\n{Colors.BOLD}{phase} Parameters:{Colors.ENDC}")
        for k, v in params.items():
            print(f"  {Colors.CYAN}{k}:{Colors.ENDC} {v}")

    # ------------------------------------------------------------------
    # Phase runners (with TUI wrappers)
    # ------------------------------------------------------------------
    def run_index_phase(self):
        self.print_header("Phase 1: Index Corpus")

        params = self.collect_index_params()
        self.print_params(params, "Index")

        if not self.confirm(f"\n{Colors.YELLOW}Proceed with index?{Colors.ENDC}"):
            self.print_warning("Index cancelled")
            return None

        split = params.pop("split")
        max_q = params.pop("max_queries")
        q_start = params.pop("query_start")

        entries = self._get_entries(split)
        if entries is None:
            return None

        run_dir = phase1_index(entries, q_start, q_start + max_q, params)
        if run_dir:
            self.last_run_dir = run_dir
            self.print_success(f"Index run complete: {run_dir.name}")

            if self.confirm(f"\n{Colors.YELLOW}Run benchmark now?{Colors.ENDC}"):
                # Restore params
                params["split"] = split
                params["query_start"] = q_start
                params["query_end"] = q_start + max_q
                params["max_queries"] = max_q
                self.run_benchmark_phase(run_dir, params, entries)

        return run_dir

    def run_benchmark_phase(self, run_dir: Path = None, params: dict = None, entries: List[dict] = None):
        self.print_header("Phase 2: Benchmark")

        # If no run_dir provided, let user pick from registry
        if run_dir is None:
            index_runs = get_registered_runs("index")
            completed = [(rid, d) for rid, d in index_runs if d.get("status") == "completed"]
            if not completed:
                self.print_error("No completed index runs found. Run Phase 1 first.")
                return None

            print(f"\n{Colors.BOLD}Available index runs:{Colors.ENDC}")
            for i, (rid, data) in enumerate(completed[:10], 1):
                print(f"  {i}. {rid} ({data['params'].get('grid_size', '?')} grid, {data['params'].get('max_queries', '?')} queries)")
            choice = self.get_choice("Select run:", [f"{rid}" for rid, _ in completed[:10]])
            run_dir = Path(completed[choice - 1][1]["path"])

        if params is None:
            params = self.collect_benchmark_params(run_dir)

        split = params.get("split", "dev")
        q_start = params.get("query_start", 0)
        q_end = params.get("query_end", 100)

        self.print_params(params, "Benchmark")
        if not self.confirm(f"\n{Colors.YELLOW}Proceed with benchmark?{Colors.ENDC}"):
            self.print_warning("Benchmark cancelled")
            return None

        if entries is None:
            entries = self._get_entries(split)
            if entries is None:
                return None

        bench_dir = phase2_benchmark(run_dir, entries, q_start, q_end, params)
        if bench_dir:
            self.last_bench_dir = bench_dir
            self.print_success(f"Benchmark complete: {bench_dir.name}")

            if self.confirm(f"\n{Colors.YELLOW}Generate report now?{Colors.ENDC}"):
                phase3_report(bench_dir)
                self.print_success(f"Report generated in {bench_dir.name}")
        return bench_dir

    def run_report_phase(self, bench_dir: Path = None):
        self.print_header("Phase 3: Generate Report")

        if bench_dir is None:
            bench_runs = get_registered_runs("benchmark")
            completed = [(rid, d) for rid, d in bench_runs if d.get("status") == "completed"]
            if not completed:
                self.print_error("No completed benchmarks found.")
                return

            print(f"\n{Colors.BOLD}Available benchmarks:{Colors.ENDC}")
            for i, (rid, data) in enumerate(completed[:10], 1):
                p = data["params"]
                print(f"  {i}. {rid} (grid={p.get('grid_size', '?')}, queries={p.get('query_end', 0)})")
            choice = self.get_choice("Select benchmark:", [f"{rid}" for rid, _ in completed[:10]])
            bench_dir = Path(completed[choice - 1][1]["path"])

        phase3_report(bench_dir)
        self.last_bench_dir = bench_dir
        self.print_success(f"Report -> {bench_dir / 'benchmark_report.md'}")

    def run_analysis_phase(self):
        self.print_header("Analyze Benchmark Results")

        bench_dir = self.last_bench_dir
        if bench_dir is None:
            bench_runs = get_registered_runs("benchmark")
            completed = [(rid, d) for rid, d in bench_runs if d.get("status") == "completed"]
            if not completed:
                self.print_error("No completed benchmarks found.")
                return
            bench_dir = Path(completed[0][1]["path"])

        # Import and run analyzer
        sys.path.insert(0, str(SCRIPT_DIR))
        from benchmark_analyzer import analyze_benchmark, print_analysis
        analysis = analyze_benchmark(bench_dir)
        if analysis:
            print_analysis(analysis)

    # ------------------------------------------------------------------
    # Entry helpers
    # ------------------------------------------------------------------
    def _get_entries(self, split: str) -> Optional[List[dict]]:
        if split not in self.entries_cache:
            try:
                self.entries_cache[split] = load_musique_entries(split)
            except FileNotFoundError as e:
                self.print_error(str(e))
                return None
        return self.entries_cache[split]

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def show_resume_options(self):
        registry = load_registry()
        runs = registry.get("runs", {})

        # Find interrupted runs
        interrupted = [(rid, d) for rid, d in runs.items()
                       if d.get("status", "").startswith("failed_")]
        completed_benchmarks = [(rid, d) for rid, d in runs.items()
                                if d.get("type") == "benchmark" and d.get("status") == "completed"]

        if not interrupted and not completed_benchmarks:
            self.print_warning("No runs to resume")
            return

        options = []
        if interrupted:
            for rid, data in interrupted[:5]:
                status = data.get("status", "interrupted")
                p = data.get("params", {})
                options.append(f"Resume {rid} ({status}, grid={p.get('grid_size', '?')})")

        if completed_benchmarks:
            for rid, data in completed_benchmarks[:5]:
                p = data.get("params", {})
                options.append(f"Re-report {rid} (grid={p.get('grid_size', '?')})")

        options.append("Back")

        choice = self.get_choice("Resume options:", options)
        if options[choice - 1] == "Back":
            return

        selected = options[choice - 1]
        if selected.startswith("Resume"):
            run_id = selected.split(" ")[1]
            data = runs[run_id]
            run_dir = Path(data["path"])
            status = data["status"]
            logger.info(f"Resuming interrupted run: {run_dir.name} (status={status})")

            # Determine what step failed
            failed_step = int(status.replace("failed_step", ""))
            self.print_warning(f"Run failed at Step {failed_step}. "

                              f"Manual inspection needed before resume.")

        elif selected.startswith("Re-report"):
            run_id = selected.split(" ")[1]
            data = runs[run_id]
            bench_dir = Path(data["path"])
            if bench_dir.exists():
                phase3_report(bench_dir)
                self.print_success(f"Report regenerated for {run_id}")

    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------
    def show_main_menu(self):
        self.print_header("MuSiQue Benchmark Runner")

        # Status bar
        idx = get_registered_runs("index")
        bm = get_registered_runs("benchmark")
        print(f"{Colors.CYAN}  Index runs:     {len([r for r in idx if r[1].get('status') == 'completed'])} completed / {len(idx)} total{Colors.ENDC}")
        print(f"{Colors.CYAN}  Benchmarks:     {len([r for r in bm if r[1].get('status') == 'completed'])} completed / {len(bm)} total{Colors.ENDC}\n")

        options = [
            "Phase 1: Index Corpus (Steps 1-5)",
            "Phase 2: Benchmark (Step 6 per query)",
            "Phase 3: Generate Report",
            "Analyze Last Benchmark Results",
            "Resume / Re-run",
            "Exit",
        ]
        choice = self.get_choice("Main Menu:", options)

        if choice == 1:
            self.run_index_phase()
        elif choice == 2:
            self.run_benchmark_phase()
        elif choice == 3:
            self.run_report_phase()
        elif choice == 4:
            self.run_analysis_phase()
        elif choice == 5:
            self.show_resume_options()
        elif choice == 6:
            print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
            sys.exit(0)

    def run(self):
        try:
            while True:
                self.show_main_menu()
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            self.print_error(f"Unexpected error: {e}")
            sys.exit(1)


# ============================================================================
# CLI entry point (non-interactive)
# ============================================================================

def cli_main():
    parser = argparse.ArgumentParser(
        description="MuSiQue Benchmark - 3-phase design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["index", "benchmark", "report"],
                        help="Operation mode (omit for interactive TUI)")
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-end", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Shorthand: sets --query-end = max_queries (assumes start=0)")
    parser.add_argument("--grid-size", type=int, default=PIPELINE_DEFAULTS["grid_size"])
    parser.add_argument("--spreading-steps", type=int, default=PIPELINE_DEFAULTS["spreading_steps"])
    parser.add_argument("--top-k", type=int, default=PIPELINE_DEFAULTS["top_k"])
    parser.add_argument("--weighting", choices=["uniform", "frequency", "idf"],
                        default=PIPELINE_DEFAULTS["weighting"])
    parser.add_argument("--top-percent", type=float, default=PIPELINE_DEFAULTS["top_percent"])
    parser.add_argument("--smoothing-sigma", type=float, default=PIPELINE_DEFAULTS["smoothing_sigma"])
    parser.add_argument("--no-morton", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Pre-built run directory (for --mode benchmark)")
    parser.add_argument("--benchmark-dir", type=Path, default=None,
                        help="Benchmark directory (for --mode report)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark immediately after indexing")

    args = parser.parse_args()

    # No mode -> interactive TUI
    if args.mode is None:
        runner = BenchmarkRunner()
        runner.run()
        return

    params = {k: getattr(args, k, v) for k, v in PIPELINE_DEFAULTS.items()}
    params["morton"] = not args.no_morton

    if args.max_queries is not None:
        query_end = args.max_queries
    elif args.query_end is not None:
        query_end = args.query_end
    else:
        query_end = None

    # Report mode
    if args.mode == "report":
        if not args.benchmark_dir:
            logger.error("--benchmark-dir required for report mode")
            sys.exit(1)
        phase3_report(args.benchmark_dir)
        return

    entries = load_musique_entries(args.split)

    # Index mode
    if args.mode == "index":
        if query_end is None:
            logger.error("Specify --max-queries or --query-end for index mode")
            sys.exit(1)
        run_dir = phase1_index(entries, args.query_start, query_end, params)
        if run_dir is None:
            logger.error("Index phase failed")
            sys.exit(1)
        logger.success(f"Index run ready: {run_dir}")

        if args.benchmark:
            logger.info("Running benchmark immediately after indexing...")
            bench_dir = phase2_benchmark(run_dir, entries, args.query_start, query_end, params)
            if bench_dir:
                phase3_report(bench_dir)
        return

    # Benchmark mode
    if args.mode == "benchmark":
        if not args.run_dir:
            logger.error("--run-dir required for benchmark mode")
            sys.exit(1)
        if query_end is None:
            query_end = len(entries)

        # Load pipeline params from run's config.yml
        run_config_path = args.run_dir / "config.yml"
        if run_config_path.exists():
            with open(run_config_path) as f:
                run_cfg = yaml.safe_load(f)
            run_pipe = run_cfg.get("pipeline", {})
            for k in PIPELINE_DEFAULTS:
                if k in run_pipe:
                    params[k] = run_pipe[k]
            params["morton"] = run_pipe.get("morton", True)
            logger.info(f"Loaded pipeline params from run config: grid={params['grid_size']}, "
                        f"morton={params['morton']}")

        bench_dir = phase2_benchmark(args.run_dir, entries, args.query_start, query_end, params)
        if bench_dir:
            phase3_report(bench_dir)
        return


if __name__ == "__main__":
    cli_main()
