#!/usr/bin/env python3
"""
Quran Benchmark for Semantic Folding

Three-Phase Design:
  Phase 1 (index)   — Run Steps 1-5 on the full Quran corpus (6236 ayahs)
  Phase 2 (evaluate) — Run Step 7 for each QA pair, compute metrics
  Phase 3 (report)   — Generate markdown report with SF vs BM25 comparison

Usage:
    # Interactive (default)
    python semantic_folding/dataset_benchmark/quran/run_benchmark.py

    # CLI — index + evaluate in one shot
    python semantic_folding/dataset_benchmark/quran/run_benchmark.py --mode all

    # CLI — individual phases
    python semantic_folding/dataset_benchmark/quran/run_benchmark.py --mode index
    python semantic_folding/dataset_benchmark/quran/run_benchmark.py --mode evaluate --run-dir <path>
    python semantic_folding/dataset_benchmark/quran/run_benchmark.py --mode report --eval-dir <path>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import get_logger
logger = get_logger("quran_bench")

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # up to project root
CORPUS_PATH = PROJECT_ROOT / "data" / "quran" / "quran_ayahs_clean.txt"
QA_PATH = PROJECT_ROOT / "data" / "quran" / "quran_qa.jsonl"
BENCHMARK_BASE = PROJECT_ROOT / "outputs" / "quran_benchmark"
RUNS_DIR = BENCHMARK_BASE / "runs"
EVALS_DIR = BENCHMARK_BASE / "evaluations"
SEMANTIC_FOLDING = PROJECT_ROOT / "semantic_folding"

# Output layout:
#   outputs/quran_benchmark/
#     runs/run_<ts>/           # Phase 1: Steps 1-5 artifacts (fingerprints, etc.)
#     evaluations/eval_<ts>/   # Phase 2: per-query results + report

STEP_SCRIPTS = {
    1: SEMANTIC_FOLDING / "phrase_extractor.py",
    2: SEMANTIC_FOLDING / "term_context.py",
    3: SEMANTIC_FOLDING / "semantic_space.py",
    4: SEMANTIC_FOLDING / "phrase_fingerprints.py",
    5: SEMANTIC_FOLDING / "doc_fingerprints.py",
    7: SEMANTIC_FOLDING / "query_processor.py",
}

PIPELINE_DEFAULTS = {
    "grid_size": 64,
    "spreading_steps": 0,
    "top_k": 5,
    "weighting": "idf",
    "top_percent": 0.05,
    "smoothing_sigma": 1.5,
    "keep_verbs": True,
    "min_word_length": 3,
    "min_freq": 1,
    "morton": True,
    "method": "umap",
    "tsne_perplexity": 30,
    "tsne_iter": 1000,
}


# ============================================================================
# Utilities
# ============================================================================

def run_step(script_path: Path, args: list, cwd: Path, label: str,
             timeout: int = 600) -> bool:
    """Run a pipeline step via subprocess, return success."""
    cmd = [sys.executable, str(script_path)] + args
    logger.info(f"  [{label}] starting...")
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, timeout=timeout,
                       capture_output=True, text=True, encoding="utf8", errors="replace")
        elapsed = time.time() - t0
        logger.info(f"  [{label}] done in {elapsed:.0f}s")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  [{label}] FAILED (exit={e.returncode})")
        logger.error(e.stderr[-500:] if e.stderr else "")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"  [{label}] TIMEOUT after {timeout}s")
        return False


def load_qa_pairs(qa_path: Path) -> List[dict]:
    """Load QA pairs from jsonl."""
    pairs = []
    with open(qa_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def compute_metrics(retrieved: List[Tuple[str, float]], relevant: List[str],
                    top_k_list: List[int] = None) -> dict:
    """Compute MRR, AP, P@K, R@K, NDCG@K."""
    if top_k_list is None:
        top_k_list = [1, 2, 3, 5, 10]
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


def bm25_retrieve(query: str, corpus_lines: List[str], corpus_ids: List[str],
                  k1: float = 1.5, b: float = 0.75, top_k: int = 6236) -> List[Tuple[str, float]]:
    """Simple BM25 retrieval for the Quran corpus."""
    import re
    import math
    from collections import Counter

    # Tokenize
    def tokenize(text):
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    # Corpus stats
    N = len(corpus_lines)
    avgdl = 0
    doc_tokens = []
    doc_len = []
    for line in corpus_lines:
        toks = tokenize(line)
        doc_tokens.append(toks)
        doc_len.append(len(toks))
        avgdl += len(toks)
    avgdl /= max(N, 1)

    # IDF
    df = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            df[t] += 1
    idf = {}
    for t, n in df.items():
        idf[t] = math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    # Query tokens
    q_tokens = tokenize(query)

    # Score
    scores = []
    for i in range(N):
        score = 0.0
        tf = Counter(doc_tokens[i])
        for qt in q_tokens:
            if qt in idf:
                freq = tf.get(qt, 0)
                numer = freq * (k1 + 1)
                denom = freq + k1 * (1 - b + b * doc_len[i] / avgdl)
                score += idf[qt] * numer / denom
        scores.append((corpus_ids[i], score))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


# ============================================================================
# Phase 1 — Index
# ============================================================================

def phase1_index(params: dict) -> Optional[Path]:
    """Run Steps 1-5 on the full Quran corpus."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy corpus
    corpus_path = run_dir / "corpus.txt"
    shutil.copy2(str(CORPUS_PATH), str(corpus_path))

    # Metadata
    with open(CORPUS_PATH, "r", encoding="utf8") as f:
        num_docs = sum(1 for _ in f)

    with open(run_dir / "metadata.json", "w") as f:
        json.dump({
            "num_docs": num_docs,
            "corpus": str(CORPUS_PATH),
            "created_at": ts,
            "pipeline_params": {k: v for k, v in params.items()},
        }, f, indent=2)

    # Config
    run_config = {
        "phase1": {
            "mode": "index",
            "timestamp": ts,
            "num_docs": num_docs,
        },
        "pipeline": {k: v for k, v in params.items()},
    }
    with open(run_dir / "config.yml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    logger.info(f"Index run: {run_dir.name} ({num_docs} ayahs)")

    # Step 1 — Phrase extraction
    out = run_dir / "extracted_phrases"
    ok = run_step(STEP_SCRIPTS[1], [
        "--corpus", str(corpus_path), "--output", str(out),
        "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
        "--min-freq", str(params["min_freq"]),
    ], PROJECT_ROOT, "Step 1 phrase_extractor")
    if not ok:
        return None

    # Step 2 — Term-context matrix
    out = run_dir / "term_context_matrix"
    ok = run_step(STEP_SCRIPTS[2], [
        "--vocab", str(run_dir / "extracted_phrases" / "vocabulary.csv"),
        "--mapping", str(run_dir / "extracted_phrases" / "phrase_to_contexts.json"),
        "--corpus", str(corpus_path), "--output", str(out),
    ], PROJECT_ROOT, "Step 2 term_context")
    if not ok:
        return None

    # Step 3 — Dimensionality reduction
    out = run_dir / "semantic_space"
    step3_args = [
        "--matrix", str(run_dir / "term_context_matrix" / "term_context_matrix.npz"),
        "--metadata", str(run_dir / "term_context_matrix" / "term_context_matrix.json"),
        "--output", str(out),
        "--grid-size", str(params["grid_size"]),
        "--method", params["method"],
        "--perplexity", str(params["tsne_perplexity"]),
        "--tsne-iter", str(params["tsne_iter"]),
    ]
    ok = run_step(STEP_SCRIPTS[3], step3_args, PROJECT_ROOT, "Step 3 semantic_space", timeout=900)
    if not ok:
        return None

    # Step 4 — Phrase fingerprints
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
        return None

    # Step 5 — Document fingerprints
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
        return None

    logger.success(f"Index phase complete -> {run_dir}")
    return run_dir


# ============================================================================
# Phase 2 — Evaluate
# ============================================================================

def phase2_evaluate(run_dir: Path, params: dict) -> Optional[Path]:
    """Run Step 7 for each QA pair, compute SF + BM25 metrics."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = EVALS_DIR / f"eval_{ts}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    qa_pairs = load_qa_pairs(QA_PATH)
    corpus_path = run_dir / "corpus.txt"

    # Load corpus for BM25
    with open(corpus_path, "r", encoding="utf8") as f:
        corpus_lines = [line.strip() for line in f]
    corpus_ids = [str(i + 1) for i in range(len(corpus_lines))]

    logger.info(f"Evaluating {len(qa_pairs)} QA pairs...")

    # Prepare per-query results
    sf_results = []
    bm25_results = []
    all_metrics = []

    for qa in qa_pairs:
        qid = qa["id"]
        question = qa["question"]
        relevant = qa["relevant"]  # list of line number strings
        logger.info(f"  [{qid}] {question[:60]}...")

        # SF retrieval via Step 7
        sf_retrieved = _run_step7_query(run_dir, question, params)
        sf_metrics = compute_metrics(sf_retrieved, relevant)
        sf_results.append({"id": qid, "question": question, "retrieved": sf_retrieved, "metrics": sf_metrics})
        logger.info(f"    SF: MRR={sf_metrics['mrr']:.4f} AP={sf_metrics['ap']:.4f}")

        # BM25 retrieval
        bm25_retrieved = bm25_retrieve(question, corpus_lines, corpus_ids)
        bm25_metrics = compute_metrics(bm25_retrieved, relevant)
        bm25_results.append({"id": qid, "question": question, "retrieved": bm25_retrieved, "metrics": bm25_metrics})
        logger.info(f"    BM25: MRR={bm25_metrics['mrr']:.4f} AP={bm25_metrics['ap']:.4f}")

        all_metrics.append({
            "id": qid,
            "category": qa.get("category", ""),
            "num_relevant": len(relevant),
            "sf_mrr": sf_metrics["mrr"],
            "sf_ap": sf_metrics["ap"],
            "sf_p@5": sf_metrics["p@5"],
            "sf_r@5": sf_metrics["r@5"],
            "sf_ndcg@10": sf_metrics["ndcg@10"],
            "bm25_mrr": bm25_metrics["mrr"],
            "bm25_ap": bm25_metrics["ap"],
            "bm25_p@5": bm25_metrics["p@5"],
            "bm25_r@5": bm25_metrics["r@5"],
            "bm25_ndcg@10": bm25_metrics["ndcg@10"],
        })

    # Save per-query results
    with open(eval_dir / "sf_results.json", "w") as f:
        json.dump(sf_results, f, indent=2)
    with open(eval_dir / "bm25_results.json", "w") as f:
        json.dump(bm25_results, f, indent=2)
    with open(eval_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Compute aggregate
    agg = _compute_aggregate(all_metrics)
    with open(eval_dir / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Save eval config
    eval_config = {
        "phase2": {
            "timestamp": ts,
            "run_dir": str(run_dir),
            "num_queries": len(qa_pairs),
            "qa_path": str(QA_PATH),
        },
        "pipeline": {k: v for k, v in params.items()},
    }
    with open(eval_dir / "config.yml", "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False)

    # Print summary
    _print_summary(agg, len(qa_pairs))

    # Auto-generate report
    report_path = eval_dir / "quran_benchmark_report.md"
    _generate_report(agg, all_metrics, eval_dir, params, report_path)
    logger.success(f"Report saved -> {report_path}")

    return eval_dir


def _extract_key_query_terms(question: str, run_dir: Path) -> str:
    """Simplify query to key content terms, removing noise words.

    Keeps only vocabulary-matched nouns/proper nouns, discarding common
    verbs and high-frequency generic terms like 'quran' and 'prophet'.
    Falls back to the original query if no key terms survive filtering.
    """
    import re

    _VERB_BLOCKLIST = {
        "say", "describe", "teach", "command", "instruct", "promise",
        "narrate", "tell", "mention", "speak", "talk", "ask", "give",
        "make", "take", "do", "does", "know", "think", "believe",
        "practice", "importance",
    }
    _GENERIC_BLOCKLIST = {
        "quran", "prophet", "surah", "people", "story", "chapter",
        "message", "opening", "great",
    }

    meta_path = run_dir / "phrase_fingerprints" / "phrase_fingerprints_meta.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        vocab = set(meta.get("phrase_to_row", {}).keys())
    except Exception:
        return question

    tokens = re.findall(r"[a-zA-Z]+", question.lower())
    key_terms = [
        t for t in tokens
        if t in vocab and len(t) >= 3
        and t not in _VERB_BLOCKLIST and t not in _GENERIC_BLOCKLIST
    ]
    seen, uniq = set(), []
    for t in key_terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    return " ".join(uniq) if uniq else question


def _run_step7_query(run_dir: Path, question: str, params: dict) -> List[Tuple[str, float]]:
    """Run a single query through Step 7 (query_processor.py)."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf8") as f:
        result_json = Path(f.name)

    try:
        # Simplify query to key terms for better fingerprint discrimination
        simplified = _extract_key_query_terms(question, run_dir)

        cmd = [
            sys.executable, str(STEP_SCRIPTS[7]),
            "--query", simplified,
            "--fingerprints", str(run_dir / "phrase_fingerprints"),
            "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
            "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
            "--output", str(result_json),
            "--grid-size", str(params["grid_size"]),
            "--top-k", str(params["top_k"]),
            "--spreading-steps", str(params["spreading_steps"]),
            "--weighting", params["weighting"],
            "--keep-verbs", "--min-word-length", str(params.get("min_word_length", 2)),
            "--simple-query",
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True,
                       capture_output=False, timeout=300)

        # Read results from JSON output
        if result_json.exists():
            with open(result_json, "r", encoding="utf8") as f:
                data = json.load(f)
            # data is a list of [{query, results, metadata}]
            if isinstance(data, list) and len(data) > 0:
                retrieved = [(item[0], item[1]) for item in data[0].get("results", [])]
            else:
                retrieved = []
        else:
            retrieved = []

    except Exception as e:
        logger.error(f"Step 7 query failed: {e}")
        retrieved = []
    finally:
        if result_json.exists():
            os.unlink(result_json)

    return retrieved


def _compute_aggregate(all_metrics: List[dict]) -> dict:
    """Compute aggregate metrics across all queries."""
    n = len(all_metrics)
    if n == 0:
        return {}

    agg = {"num_queries": n}
    for prefix in ["sf_", "bm25_"]:
        for metric in ["mrr", "ap", "p@5", "r@5", "ndcg@10"]:
            key = prefix + metric
            values = [qm[key] for qm in all_metrics]
            agg[key + "_mean"] = sum(values) / n
            agg[key + "_median"] = sorted(values)[n // 2]
            agg[key + "_min"] = min(values)
            agg[key + "_max"] = max(values)
    return agg


def _print_summary(agg: dict, n: int):
    """Print a summary of results."""
    print(f"\n{'='*60}")
    print(f"  Quran Benchmark Results ({n} queries)")
    print(f"{'='*60}")
    for label, prefix in [("Semantic Folding", "sf_"), ("BM25", "bm25_")]:
        print(f"\n  {label}:")
        print(f"    MRR      = {agg[prefix + 'mrr_mean']:.4f} (median={agg[prefix + 'mrr_median']:.4f})")
        print(f"    AP       = {agg[prefix + 'ap_mean']:.4f}")
        print(f"    P@5      = {agg[prefix + 'p@5_mean']:.4f}")
        print(f"    R@5      = {agg[prefix + 'r@5_mean']:.4f}")
        print(f"    NDCG@10  = {agg[prefix + 'ndcg@10_mean']:.4f}")


def _generate_report(agg: dict, all_metrics: List[dict], eval_dir: Path,
                     params: dict, report_path: Path):
    """Generate a markdown report."""
    sf_mrr = agg.get("sf_mrr_mean", 0)
    bm25_mrr = agg.get("bm25_mrr_mean", 0)
    sf_ap = agg.get("sf_ap_mean", 0)
    bm25_ap = agg.get("bm25_ap_mean", 0)

    lines = []
    lines.append("# Quran Benchmark Report\n")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Queries**: {agg.get('num_queries', 0)}")
    lines.append(f"**Pipeline params**: `{json.dumps(params)}`\n")

    lines.append("## Aggregate Results\n")
    lines.append("| Metric | Semantic Folding | BM25 | Δ | Winner |")
    lines.append("|--------|:---------------:|:----:|:-:|:------:|")
    for metric, fmt in [("mrr", ".4f"), ("ap", ".4f"), ("p@5", ".4f"), ("r@5", ".4f"), ("ndcg@10", ".4f")]:
        sf_val = agg.get(f"sf_{metric}_mean", 0)
        bm_val = agg.get(f"bm25_{metric}_mean", 0)
        delta = sf_val - bm_val
        winner = "SF" if sf_val > bm_val else ("BM25" if bm_val > sf_val else "Tie")
        lines.append(f"| {metric.upper()} | {sf_val:{fmt}} | {bm_val:{fmt}} | {delta:+.4f} | {winner} |")

    lines.append("\n## Per-Query Breakdown\n")
    lines.append("| ID | Category | #Rel | SF MRR | BM25 MRR | SF AP | BM25 AP | Winner |")
    lines.append("|---|:--------:|:----:|:------:|:--------:|:-----:|:-------:|:------:|")
    for qm in sorted(all_metrics, key=lambda x: x["id"]):
        winner = "SF" if qm["sf_mrr"] > qm["bm25_mrr"] else ("BM25" if qm["bm25_mrr"] > qm["sf_mrr"] else "Tie")
        lines.append(f"| {qm['id']} | {qm['category']} | {qm['num_relevant']} | "
                     f"{qm['sf_mrr']:.4f} | {qm['bm25_mrr']:.4f} | "
                     f"{qm['sf_ap']:.4f} | {qm['bm25_ap']:.4f} | {winner} |")

    lines.append("\n## Where SF Wins\n")
    sf_wins = [qm for qm in all_metrics if qm["sf_mrr"] > qm["bm25_mrr"]]
    bm25_wins = [qm for qm in all_metrics if qm["bm25_mrr"] > qm["sf_mrr"]]
    ties = [qm for qm in all_metrics if qm["sf_mrr"] == qm["bm25_mrr"]]
    lines.append(f"- **SF wins**: {len(sf_wins)} queries")
    lines.append(f"- **BM25 wins**: {len(bm25_wins)} queries")
    lines.append(f"- **Ties**: {len(ties)} queries\n")

    if sf_wins:
        lines.append("### SF wins on:\n")
        for qm in sf_wins[:10]:
            lines.append(f"- {qm['id']} ({qm['category']}): SF={qm['sf_mrr']:.4f} vs BM25={qm['bm25_mrr']:.4f}")
    if bm25_wins:
        lines.append("\n### BM25 wins on:\n")
        for qm in bm25_wins[:10]:
            lines.append(f"- {qm['id']} ({qm['category']}): BM25={qm['bm25_mrr']:.4f} vs SF={qm['sf_mrr']:.4f}")

    with open(report_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================
# Phase 3 — Report (re-generate from existing eval dir)
# ============================================================================

def phase3_report(eval_dir: Path):
    """Re-generate report from existing evaluation."""
    metrics_path = eval_dir / "all_metrics.json"
    config_path = eval_dir / "config.yml"
    if not metrics_path.exists():
        logger.error(f"No metrics found in {eval_dir}")
        return

    with open(metrics_path) as f:
        all_metrics = json.load(f)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    params = config.get("pipeline", {})
    agg = _compute_aggregate(all_metrics)
    report_path = eval_dir / "quran_benchmark_report.md"
    _generate_report(agg, all_metrics, eval_dir, params, report_path)
    _print_summary(agg, len(all_metrics))
    logger.success(f"Report regenerated -> {report_path}")


# ============================================================================
# Main
# ============================================================================

def cli_main():
    parser = argparse.ArgumentParser(
        description="Quran Benchmark for Semantic Folding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["index", "evaluate", "report", "all"],
                        help="Operation mode (omit for interactive TUI)")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Pre-built run directory (for --mode evaluate)")
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="Evaluation directory (for --mode report)")

    # Pipeline params
    parser.add_argument("--grid-size", type=int, default=PIPELINE_DEFAULTS["grid_size"])
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default=PIPELINE_DEFAULTS["method"])
    parser.add_argument("--spreading-steps", type=int, default=PIPELINE_DEFAULTS["spreading_steps"])
    parser.add_argument("--top-k", type=int, default=PIPELINE_DEFAULTS["top_k"])
    parser.add_argument("--weighting", choices=["uniform", "frequency", "idf"], default=PIPELINE_DEFAULTS["weighting"])
    parser.add_argument("--top-percent", type=float, default=PIPELINE_DEFAULTS["top_percent"])
    parser.add_argument("--no-morton", action="store_true")

    args = parser.parse_args()

    params = {k: getattr(args, k, v) for k, v in PIPELINE_DEFAULTS.items()}
    params["morton"] = not args.no_morton

    # Interactive mode
    if args.mode is None:
        _interactive_main(params)
        return

    # Report mode
    if args.mode == "report":
        if not args.eval_dir:
            logger.error("--eval-dir required for report mode")
            sys.exit(1)
        phase3_report(args.eval_dir)
        return

    # Index mode
    if args.mode == "index":
        run_dir = phase1_index(params)
        if run_dir:
            logger.success(f"Index ready: {run_dir}")
        return

    # Evaluate mode
    if args.mode == "evaluate":
        if not args.run_dir:
            logger.error("--run-dir required for evaluate mode")
            sys.exit(1)
        eval_dir = phase2_evaluate(args.run_dir, params)
        if eval_dir:
            logger.success(f"Evaluation done: {eval_dir}")
        return

    # All mode
    if args.mode == "all":
        run_dir = phase1_index(params)
        if not run_dir:
            logger.error("Index phase failed")
            sys.exit(1)
        eval_dir = phase2_evaluate(run_dir, params)
        if eval_dir:
            logger.success(f"All phases complete!")
        return


def _interactive_main(params: dict):
    """Simple interactive menu."""
    while True:
        print(f"\n{'='*50}")
        print("  Quran Benchmark Runner")
        print(f"{'='*50}")
        print("  1. Index + Evaluate (full run)")
        print("  2. Index only (Steps 1-5)")
        print("  3. Evaluate (requires --run-dir)")
        print("  4. Regenerate report")
        print("  5. Exit")

        choice = input("\n  Choice [1-5]: ").strip()
        if choice == "1":
            run_dir = phase1_index(params)
            if run_dir:
                phase2_evaluate(run_dir, params)
        elif choice == "2":
            phase1_index(params)
        elif choice == "4":
            eval_dirs = sorted(EVALS_DIR.glob("eval_*"))
            if not eval_dirs:
                print("  No evaluations found.")
            else:
                print("  Available evaluations:")
                for i, d in enumerate(eval_dirs, 1):
                    print(f"    {i}. {d.name}")
                sel = input("  Select: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(eval_dirs):
                    phase3_report(eval_dirs[int(sel) - 1])
        elif choice == "5":
            print("  Goodbye!")
            break


if __name__ == "__main__":
    cli_main()
