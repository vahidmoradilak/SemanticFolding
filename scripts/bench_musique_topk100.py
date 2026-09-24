"""MuSiQue top_k=100 benchmark: Pure SF / BM25 / SPLADE-only / SF+Linear / SF+RRF.

Reuses canonical run_20260710_162617 index (Steps 1-5) and cached SPLADE doc
vectors; re-runs Step 7 per method with --top-k 100 in BATCH mode
(--query-file, SPLADE loaded once).
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
sys.path.insert(0, str(ROOT / "semantic_folding/dataset_benchmark"))
sys.path.insert(0, str(ROOT / "semantic_folding/dataset_benchmark/musique"))

import yaml  # noqa: E402
from run_benchmark import (  # noqa: E402
    BENCHMARKS_DIR, load_musique_entries, load_query_results,
    compute_metrics, filter_results_to_candidates, run_step, update_run_status,
    STEP_SCRIPTS,
)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "custom_ar_en_run_benchmark",
    ROOT / "semantic_folding/dataset_benchmark/custom_ar_en/run_benchmark.py")
_custom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_custom)
run_bm25 = _custom.run_bm25

RUN_DIR = ROOT / "outputs/musique_benchmark/runs/run_20260710_162617"
OUT_BASE = ROOT / "outputs/musique_benchmark/topk100"
TOP_K = 100
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


def build_args(query_file, params, run_dir, out_json, splade=None):
    """splade: None or dict(fusion_method, hybrid_alpha, rrf_k)."""
    args = [
        "--query-file", str(query_file),
        "--fingerprints", str(run_dir / "phrase_fingerprints"),
        "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
        "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
        "--grid-size", str(params["grid_size"]),
        "--top-k", str(TOP_K),
        "--weighting", params["weighting"],
        "--spreading-steps", str(params["spreading_steps"]),
        "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
        "--output", str(out_json),
    ]
    if splade:
        args += ["--splade", "--splade-model", SPLADE_MODEL,
                 "--corpus", str(run_dir / "corpus.txt"),
                 "--fusion-method", splade["fusion_method"],
                 "--hybrid-alpha", str(splade.get("hybrid_alpha", 0.3))]
        if splade["fusion_method"] == "rrf":
            args += ["--rrf-k", str(splade.get("rrf_k", 60))]
    return args


def run_step7_batch(run_dir, entries, active, params, tag, splade=None):
    """Single batch step7 call over all active queries."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = BENCHMARKS_DIR / f"benchmark_{ts}_{tag}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir = bench_dir / "per_query"
    per_query_dir.mkdir(exist_ok=True)

    with open(run_dir / "query_doc_map.json", encoding="utf-8") as f:
        query_doc_map = json.load(f)
    with open(run_dir / "query_gold.json", encoding="utf-8") as f:
        query_gold = json.load(f)

    bench_config = {
        "phase2": {"mode": "benchmark", "timestamp": ts,
                   "run_dir": str(run_dir), "tag": tag,
                   "splade": bool(splade), "fusion": splade, "batch": True,
                   "top_k": TOP_K},
        "pipeline": {k: v for k, v in params.items()},
    }
    with open(bench_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(bench_config, f, default_flow_style=False)

    update_run_status(bench_dir, "running")

    query_file = bench_dir / "queries.txt"
    with open(query_file, "w", encoding="utf-8") as f:
        for i in active:
            f.write(entries[i]["question"].replace("\n", " ") + "\n")

    result_json = bench_dir / "batch_results.json"
    t0 = time.time()
    ok = run_step(STEP_SCRIPTS[7],
                  build_args(query_file, params, run_dir, result_json, splade),
                  ROOT, f"Step 7 {tag}",
                  timeout=int(os.environ.get("MUSIQUE_STEP7_TIMEOUT", "7200")))
    elapsed = time.time() - t0
    if not ok:
        update_run_status(bench_dir, "failed")
        print(f"[{tag}] batch FAILED after {elapsed:.0f}s", flush=True)
        return None

    raw = json.load(open(result_json, encoding="utf-8"))
    if len(raw) != len(active):
        print(f"[{tag}] WARNING batch results {len(raw)} vs active {len(active)}",
              flush=True)

    all_metrics = []
    results_log = bench_dir / "results_log.csv"
    with open(results_log, "w", newline="", encoding="utf-8") as csv_f:
        w = csv.writer(csv_f)
        w.writerow(["query_idx", "query", "mrr", "ap", "p@1", "p@2",
                    "p@3", "p@5", "r@2", "ndcg@2", "found_at"])
        for qi, (i, entry) in enumerate(zip(active, raw)):
            q_idx_str = str(i)
            candidate_ids = query_doc_map.get(q_idx_str, [])
            gold_ids = query_gold.get(q_idx_str, [])
            full_results = entry.get("results", [])
            candidate_results = filter_results_to_candidates(full_results, candidate_ids)
            query_out_dir = per_query_dir / f"{i:04d}"
            query_out_dir.mkdir(exist_ok=True)
            (query_out_dir / "filtered_results.json").write_text(json.dumps({
                "query_idx": i, "query": entry.get("query", ""), "gold": gold_ids,
                "candidates": candidate_ids,
                "filtered_ranked": [(d, s) for d, s in candidate_results],
                "full_top10": [(d, s) for d, s in full_results[:10]],
            }, indent=2), encoding="utf-8")
            metrics = compute_metrics(candidate_results, gold_ids,
                                      top_k_list=[1, 2, 3, 5, TOP_K])
            all_metrics.append(metrics)
            w.writerow([i, entry.get("query", "")[:60], f"{metrics['mrr']:.4f}",
                        f"{metrics['ap']:.4f}", f"{metrics['p@1']:.4f}",
                        f"{metrics['p@2']:.4f}", f"{metrics['p@3']:.4f}",
                        f"{metrics['p@5']:.4f}", f"{metrics['r@2']:.4f}",
                        f"{metrics['ndcg@2']:.4f}", metrics.get("found_at", "none")])

    agg = defaultdict(list)
    for m in all_metrics:
        for k, v in m.items():
            agg[k].append(v)
    summary = {"num_queries": len(all_metrics), "failed": 0,
               "elapsed_s": round(elapsed, 1), "top_k": TOP_K}
    for k, vals in agg.items():
        summary[f"mean_{k}"] = sum(vals) / len(vals)
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    update_run_status(bench_dir, "completed")
    print(f"[{tag}] {len(all_metrics)} queries  MRR={summary['mean_mrr']:.4f} "
          f"AP={summary['mean_ap']:.4f} ({elapsed:.0f}s)", flush=True)
    return bench_dir


def main():
    run_dir = RUN_DIR.resolve()
    cfg = yaml.safe_load(open(run_dir / "config.yml", encoding="utf-8"))
    p = cfg["pipeline"]
    params = {k: p[k] for k in ["grid_size", "top_k", "weighting",
                                "spreading_steps", "min_word_length"]}
    meta = json.load(open(run_dir / "metadata.json", encoding="utf-8"))
    query_end = int(sys.argv[1]) if len(sys.argv) > 1 else meta["query_end"]
    entries = load_musique_entries("dev")
    print(f"params={params} top_k={TOP_K} queries=[0,{query_end}) "
          f"entries={len(entries)}", flush=True)

    query_gold = json.load(open(run_dir / "query_gold.json", encoding="utf-8"))
    active = sorted(int(k) for k in query_gold.keys() if int(k) < query_end)
    print(f"active (gold) queries inside range: {len(active)}", flush=True)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Pure SF (no splade)
    bd = run_step7_batch(run_dir, entries, active, params, "pure_sf_tk100",
                         splade=None)
    if bd:
        s = json.load(open(bd / "summary.json", encoding="utf-8"))
        results["pure_sf"] = {"bench_dir": bd.name, "mrr": round(s["mean_mrr"], 4),
                              "ap": round(s["mean_ap"], 4), "n": s["num_queries"],
                              "failed": s["failed"], "top_k": TOP_K}

    # 2. BM25
    qdm = json.load(open(run_dir / "query_doc_map.json", encoding="utf-8"))
    corpus_texts = [l.split(", ", 1)[-1].strip()
                    for l in open(run_dir / "corpus.txt", encoding="utf-8")
                    if l.strip()]
    gold_sets = [query_gold[str(i)] for i in active]
    cand_sets = [qdm[str(i)] for i in active]
    queries = [entries[i]["question"] for i in active]
    bm = run_bm25(corpus_texts, queries, gold_sets, cand_sets, top_k=TOP_K)
    mrr = [m["mrr"] for m in bm]
    ap = [m["ap"] for m in bm]
    results["bm25"] = {"mrr": round(float(sum(mrr) / len(mrr)), 4),
                       "ap": round(float(sum(ap) / len(ap)), 4),
                       "n": len(bm), "failed": 0, "top_k": TOP_K}
    print(f"[bm25] MRR={results['bm25']['mrr']:.4f} AP={results['bm25']['ap']:.4f} "
          f"n={len(bm)}", flush=True)

    # 3. SPLADE-only (linear alpha=0)
    bd = run_step7_batch(run_dir, entries, active, params, "splade_alpha0_tk100",
                         splade={"fusion_method": "linear", "hybrid_alpha": 0.0})
    if bd:
        s = json.load(open(bd / "summary.json", encoding="utf-8"))
        results["splade_only"] = {"bench_dir": bd.name,
                                  "mrr": round(s["mean_mrr"], 4),
                                  "ap": round(s["mean_ap"], 4),
                                  "n": s["num_queries"], "failed": s["failed"],
                                  "top_k": TOP_K}

    # 4. SF+SPLADE linear (alpha=0.3)
    bd = run_step7_batch(run_dir, entries, active, params, "splade_linear03_tk100",
                         splade={"fusion_method": "linear", "hybrid_alpha": 0.3})
    if bd:
        s = json.load(open(bd / "summary.json", encoding="utf-8"))
        results["splade_linear"] = {"bench_dir": bd.name,
                                    "mrr": round(s["mean_mrr"], 4),
                                    "ap": round(s["mean_ap"], 4),
                                    "n": s["num_queries"], "failed": s["failed"],
                                    "top_k": TOP_K}

    # 5. SF+SPLADE RRF (k=60)
    bd = run_step7_batch(run_dir, entries, active, params, "splade_rrf_tk100",
                         splade={"fusion_method": "rrf", "rrf_k": 60})
    if bd:
        s = json.load(open(bd / "summary.json", encoding="utf-8"))
        results["splade_rrf"] = {"bench_dir": bd.name,
                                 "mrr": round(s["mean_mrr"], 4),
                                 "ap": round(s["mean_ap"], 4),
                                 "n": s["num_queries"], "failed": s["failed"],
                                 "top_k": TOP_K}

    (OUT_BASE / "results.json").write_text(
        json.dumps({"run_dir": str(run_dir), "params": params, "top_k": TOP_K,
                    "results": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()