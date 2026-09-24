"""Quran top_k=100 benchmark: Pure SF / BM25 / SPLADE-only / SF+Linear / SF+RRF.

Reuses canonical run_20260721_101923 index (Steps 1-5) and cached SPLADE doc
vectors; re-runs Step 7 per method in BATCH mode (--query-file) at top_k=100,
using simplified queries (same as the quran runner's _extract_key_query_terms).
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "semantic_folding/dataset_benchmark/quran"))

import yaml  # noqa: E402
from run_benchmark import (  # noqa: E402
    PROJECT_ROOT, RUNS_DIR, EVALS_DIR, SEMANTIC_FOLDING,
    load_qa_pairs, compute_metrics, bm25_retrieve,
    _extract_key_query_terms, QA_PATH,
)

RUN_DIR = RUNS_DIR / "run_20260721_101923"
OUT_BASE = ROOT / "outputs/quran_benchmark/topk100"
TOP_K = 100
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
STEP7 = SEMANTIC_FOLDING / "query_processor.py"


def build_args(query_file, run_dir, out_json, splade=None):
    args = [
        str(STEP7),
        "--query-file", str(query_file),
        "--fingerprints", str(run_dir / "phrase_fingerprints"),
        "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
        "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
        "--output", str(out_json),
        "--grid-size", "64",
        "--top-k", str(TOP_K),
        "--weighting", "idf",
        "--spreading-steps", "1",
        "--keep-verbs", "--min-word-length", "3",
        "--simple-query",
    ]
    if splade:
        args += ["--splade", "--splade-model", SPLADE_MODEL,
                 "--corpus", str(run_dir / "corpus.txt"),
                 "--fusion-method", splade["fusion_method"],
                 "--hybrid-alpha", str(splade.get("hybrid_alpha", 0.3))]
        if splade["fusion_method"] == "rrf":
            args += ["--rrf-k", str(splade.get("rrf_k", 60))]
    return args


def run_step7_batch(run_dir, qa_pairs, tag, splade=None, timeout=3600):
    """Single batch step7 call over all queries; returns bench_dir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = EVALS_DIR / f"eval_{ts}_{tag}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    bench_config = {
        "phase2": {"run_dir": str(run_dir), "timestamp": ts, "tag": tag,
                   "splade": bool(splade), "fusion": splade, "batch": True,
                   "top_k": TOP_K, "simplified_queries": True},
        "pipeline": {"grid_size": 64, "spreading_steps": 1, "top_k": TOP_K,
                     "weighting": "idf", "min_word_length": 3,
                     "keep_verbs": True},
    }
    with open(bench_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(bench_config, f, default_flow_style=False)

    query_file = bench_dir / "queries.txt"
    with open(query_file, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            simplified = _extract_key_query_terms(qa["question"], run_dir)
            f.write(simplified.replace("\n", " ") + "\n")

    result_json = bench_dir / "batch_results.json"
    cmd = [sys.executable] + build_args(query_file, run_dir, result_json, splade)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    import subprocess
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=False,
                          timeout=timeout, env=env, encoding="utf-8",
                          errors="replace")
    ok = proc.returncode == 0
    elapsed = time.time() - t0
    if not ok:
        (bench_dir / "FAILED").write_text("step7 rc=%d\n" % proc.returncode,
                                          encoding="utf-8")
        print(f"[{tag}] batch FAILED rc={proc.returncode} after {elapsed:.0f}s",
              flush=True)
        return None

    raw = json.load(open(result_json, encoding="utf-8"))
    if len(raw) != len(qa_pairs):
        print(f"[{tag}] WARNING batch results {len(raw)} vs qa {len(qa_pairs)}",
              flush=True)

    all_metrics = []
    results_log = bench_dir / "results_log.csv"
    with open(results_log, "w", newline="", encoding="utf-8") as csv_f:
        w = csv.writer(csv_f)
        w.writerow(["qid", "query", "mrr", "ap", "p@5", "r@5", "ndcg@10",
                    "found_at"])
        for qa, entry in zip(qa_pairs, raw):
            full_results = entry.get("results", [])
            metrics = compute_metrics(full_results, qa["relevant"])
            all_metrics.append({"id": qa["id"], "category": qa.get("category", ""),
                                "num_relevant": len(qa["relevant"]),
                                "sf_mrr": metrics["mrr"], "sf_ap": metrics["ap"],
                                "sf_p@5": metrics["p@5"], "sf_r@5": metrics["r@5"],
                                "sf_ndcg@10": metrics["ndcg@10"]})
            (bench_dir / f"per_query_{qa['id']}.json").write_text(json.dumps({
                "id": qa["id"], "query": entry.get("query", ""),
                "retrieved": full_results[:20],
            }, indent=2), encoding="utf-8")
            w.writerow([qa["id"], qa["question"][:50], f"{metrics['mrr']:.4f}",
                        f"{metrics['ap']:.4f}", f"{metrics['p@5']:.4f}",
                        f"{metrics['r@5']:.4f}", f"{metrics['ndcg@10']:.4f}",
                        metrics.get("found_at", "none")])

    agg = defaultdict(list)
    for m in all_metrics:
        for k, v in m.items():
            if k.startswith("sf_"):
                agg[k].append(v)
    summary = {"num_queries": len(all_metrics), "elapsed_s": round(elapsed, 1),
               "top_k": TOP_K}
    for k, vals in agg.items():
        summary[f"mean_{k}"] = sum(vals) / len(vals)
    for k in ["sf_mrr", "sf_ap", "sf_p@5", "sf_r@5", "sf_ndcg@10"]:
        vals = agg[k]
        summary[f"median_{k}"] = sorted(vals)[len(vals) // 2]
    (bench_dir / "aggregate.json").write_text(
        json.dumps({"num_queries": len(all_metrics), **{
            f"{k}_mean": sum(v) / len(v) for k, v in agg.items()}}, indent=2,
            ensure_ascii=False), encoding="utf-8")
    (bench_dir / "all_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"[{tag}] {len(all_metrics)} queries  MRR={summary['mean_sf_mrr']:.4f} "
          f"AP={summary['mean_sf_ap']:.4f} ({elapsed:.0f}s)", flush=True)
    return bench_dir


def main():
    run_dir = RUN_DIR.resolve()
    qa_pairs = load_qa_pairs(QA_PATH)
    print(f"top_k={TOP_K} qa_pairs={len(qa_pairs)} run_dir={run_dir}", flush=True)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Pure SF (no splade)
    bd = run_step7_batch(run_dir, qa_pairs, "pure_sf_tk100", splade=None)
    if bd:
        s = json.load(open(bd / "aggregate.json", encoding="utf-8"))
        results["pure_sf"] = {"eval_dir": bd.name,
                              "mrr": round(s["sf_mrr_mean"], 4),
                              "ap": round(s["sf_ap_mean"], 4),
                              "n": s["num_queries"], "top_k": TOP_K}

    # 2. BM25 (top_k=100)
    corpus_lines = [ln.strip() for ln in
                    open(run_dir / "corpus.txt", encoding="utf8")]
    corpus_ids = [str(i + 1) for i in range(len(corpus_lines))]
    bm = [compute_metrics(bm25_retrieve(qa["question"], corpus_lines,
                                        corpus_ids, top_k=TOP_K),
                          qa["relevant"])
          for qa in qa_pairs]
    mrr = [m["mrr"] for m in bm]
    ap = [m["ap"] for m in bm]
    results["bm25"] = {"mrr": round(float(sum(mrr) / len(mrr)), 4),
                       "ap": round(float(sum(ap) / len(ap)), 4),
                       "n": len(bm), "top_k": TOP_K}
    print(f"[bm25] MRR={results['bm25']['mrr']:.4f} AP={results['bm25']['ap']:.4f} "
          f"n={len(bm)}", flush=True)

    # 3. SPLADE-only (linear alpha=0)
    bd = run_step7_batch(run_dir, qa_pairs, "splade_alpha0_tk100",
                         splade={"fusion_method": "linear", "hybrid_alpha": 0.0})
    if bd:
        s = json.load(open(bd / "aggregate.json", encoding="utf-8"))
        results["splade_only"] = {"eval_dir": bd.name,
                                  "mrr": round(s["sf_mrr_mean"], 4),
                                  "ap": round(s["sf_ap_mean"], 4),
                                  "n": s["num_queries"], "top_k": TOP_K}

    # 4. SF+SPLADE linear (alpha=0.3)
    bd = run_step7_batch(run_dir, qa_pairs, "splade_linear03_tk100",
                         splade={"fusion_method": "linear", "hybrid_alpha": 0.3})
    if bd:
        s = json.load(open(bd / "aggregate.json", encoding="utf-8"))
        results["splade_linear"] = {"eval_dir": bd.name,
                                    "mrr": round(s["sf_mrr_mean"], 4),
                                    "ap": round(s["sf_ap_mean"], 4),
                                    "n": s["num_queries"], "top_k": TOP_K}

    # 5. SF+SPLADE RRF (k=60)
    bd = run_step7_batch(run_dir, qa_pairs, "splade_rrf_tk100",
                         splade={"fusion_method": "rrf", "rrf_k": 60})
    if bd:
        s = json.load(open(bd / "aggregate.json", encoding="utf-8"))
        results["splade_rrf"] = {"eval_dir": bd.name,
                                 "mrr": round(s["sf_mrr_mean"], 4),
                                 "ap": round(s["sf_ap_mean"], 4),
                                 "n": s["num_queries"], "top_k": TOP_K}

    (OUT_BASE / "results.json").write_text(
        json.dumps({"run_dir": str(run_dir), "top_k": TOP_K,
                    "results": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()