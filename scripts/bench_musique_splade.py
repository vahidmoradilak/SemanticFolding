"""Pure SF / SPLADE α=0 / BM25 for MuSiQue on the canonical run dir.

Mirrors musique phase2_benchmark but runs SPLADE in BATCH mode
(--query-file, loaded once) and reuses canonical pure-SF baseline.
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

import yaml
from run_benchmark import (
    BENCHMARKS_DIR, load_musique_entries, load_query_results,
    compute_metrics, filter_results_to_candidates, run_step, update_run_status,
    STEP_SCRIPTS,
)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "custom_ar_en_run_benchmark",
    ROOT / "semantic_folding/dataset_benchmark/custom_ar_en/run_benchmark.py")
_custom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_custom)
run_bm25 = _custom.run_bm25

RUN_DIR = ROOT / "outputs/musique_benchmark/runs/run_20260710_162617"
OUT_BASE = ROOT / "outputs/musique_benchmark/splade_alpha0"
CANONICAL_PURE = BENCHMARKS_DIR / "benchmark_20260710_175934"

SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


def build_args(query_file, params, run_dir, use_splade, out_json):
    args = [
        "--query-file", str(query_file),
        "--fingerprints", str(run_dir / "phrase_fingerprints"),
        "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
        "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
        "--grid-size", str(params["grid_size"]),
        "--top-k", str(params["top_k"]),
        "--weighting", params["weighting"],
        "--spreading-steps", str(params["spreading_steps"]),
        "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
        "--output", str(out_json),
    ]
    if use_splade:
        args += ["--splade", "--splade-model", SPLADE_MODEL,
                 "--fusion-method", "linear", "--hybrid-alpha", "0.0",
                 "--corpus", str(run_dir / "corpus.txt")]
    return args


def run_splade_batch(run_dir, entries, active, params, tag):
    """Single batch step7 call over all active queries (SPLADE loaded once)."""
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
                   "splade": True, "hybrid_alpha": 0.0, "batch": True},
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
                  build_args(query_file, params, run_dir, True, result_json),
                  ROOT, "Step 7 SPLADE batch",
                  timeout=int(os.environ.get("MUSIQUE_STEP7_TIMEOUT", "3600")))
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
                    "p@3", "p@5", "r@2", "ndcg@2", "found_at", "elapsed_s"])
        for qi, (i, entry) in enumerate(zip(active, raw)):
            q_idx_str = str(i)
            candidate_ids = query_doc_map.get(q_idx_str, [])
            gold_ids = query_gold.get(q_idx_str, [])
            full_results = entry.get("results", [])
            candidate_results = filter_results_to_candidates(full_results, candidate_ids)
            query_out_dir = per_query_dir / f"{i:04d}"
            query_out_dir.mkdir(exist_ok=True)
            (query_out_dir / "candidate_docs.json").write_text(
                json.dumps({"candidate_ids": candidate_ids, "gold_ids": gold_ids},
                           indent=2), encoding="utf-8")
            (query_out_dir / "filtered_results.json").write_text(json.dumps({
                "query_idx": i, "query": entry.get("query", ""), "gold": gold_ids,
                "candidates": candidate_ids,
                "filtered_ranked": [(d, s) for d, s in candidate_results],
                "full_top10": [(d, s) for d, s in full_results[:10]],
                "elapsed_s": q_idx_elapsed(entry),
            }, indent=2), encoding="utf-8")
            metrics = compute_metrics(candidate_results, gold_ids,
                                      top_k_list=[1, 2, 3, 5, params["top_k"]])
            all_metrics.append(metrics)
            w.writerow([i, entry.get("query", "")[:60], f"{metrics['mrr']:.4f}",
                        f"{metrics['ap']:.4f}", f"{metrics['p@1']:.4f}",
                        f"{metrics['p@2']:.4f}", f"{metrics['p@3']:.4f}",
                        f"{metrics['p@5']:.4f}", f"{metrics['r@2']:.4f}",
                        f"{metrics['ndcg@2']:.4f}", metrics.get("found_at", "none"),
                        "batch"])

    agg = defaultdict(list)
    for m in all_metrics:
        for k, v in m.items():
            agg[k].append(v)
    summary = {"num_queries": len(all_metrics), "failed": 0,
               "elapsed_s": round(elapsed, 1)}
    for k, vals in agg.items():
        summary[f"mean_{k}"] = sum(vals) / len(vals)
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    update_run_status(bench_dir, "completed")
    print(f"[{tag}] {len(all_metrics)} queries  MRR={summary['mean_mrr']:.4f} "
          f"AP={summary['mean_ap']:.4f} ({elapsed:.0f}s)", flush=True)
    return bench_dir


def q_idx_elapsed(entry):
    md = entry.get("metadata", {})
    return round(md.get("timing", {}).get("e2e_ms", 0.0) / 1000.0, 1)


def main():
    run_dir = RUN_DIR.resolve()
    cfg = yaml.safe_load(open(run_dir / "config.yml", encoding="utf-8"))
    p = cfg["pipeline"]
    params = {k: p[k] for k in ["grid_size", "top_k", "weighting",
                                "spreading_steps", "min_word_length"]}
    meta = json.load(open(run_dir / "metadata.json", encoding="utf-8"))
    query_end = int(sys.argv[1]) if len(sys.argv) > 1 else meta["query_end"]
    entries = load_musique_entries("dev")
    print(f"params={params} queries=[0,{query_end}) entries={len(entries)}",
          flush=True)

    query_gold = json.load(open(run_dir / "query_gold.json", encoding="utf-8"))
    active = sorted(int(k) for k in query_gold.keys() if int(k) < query_end)
    print(f"active (gold) queries inside range: {len(active)}", flush=True)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    results = {}

    # Canonical pure SF baseline (from run_20260710_162617)
    if CANONICAL_PURE.exists():
        cs = json.load(open(CANONICAL_PURE / "summary.json", encoding="utf-8"))
        results["pure_sf"] = {"bench_dir": CANONICAL_PURE.name,
                              "mrr": round(cs.get("mean_mrr", 0.0), 4),
                              "ap": round(cs.get("mean_ap", 0.0), 4),
                              "n": cs.get("num_queries"),
                              "failed": cs.get("failed", 0),
                              "source": "canonical benchmark_20260710_175934"}
        print(f"[pure_sf] canonical: MRR={results['pure_sf']['mrr']:.4f} "
              f"AP={results['pure_sf']['ap']:.4f} n={results['pure_sf']['n']}",
              flush=True)

    if "--splade-only" in sys.argv:
        bd = run_splade_batch(run_dir, entries, active, params, "splade_alpha0")
        if bd:
            s = json.load(open(bd / "summary.json", encoding="utf-8"))
            results["splade_alpha0"] = {"bench_dir": bd.name,
                                        "mrr": round(s["mean_mrr"], 4),
                                        "ap": round(s["mean_ap"], 4),
                                        "n": s["num_queries"],
                                        "failed": s["failed"]}
    else:
        bd = run_splade_batch(run_dir, entries, active, params, "splade_alpha0")
        if bd:
            s = json.load(open(bd / "summary.json", encoding="utf-8"))
            results["splade_alpha0"] = {"bench_dir": bd.name,
                                        "mrr": round(s["mean_mrr"], 4),
                                        "ap": round(s["mean_ap"], 4),
                                        "n": s["num_queries"],
                                        "failed": s["failed"]}

    if "--bm25" in sys.argv:
        qdm = json.load(open(run_dir / "query_doc_map.json", encoding="utf-8"))
        corpus_texts = [l.split(", ", 1)[-1].strip()
                        for l in open(run_dir / "corpus.txt", encoding="utf-8")
                        if l.strip()]
        gold_sets = [query_gold[str(i)] for i in active]
        cand_sets = [qdm[str(i)] for i in active]
        queries = [entries[i]["question"] for i in active]
        bm = run_bm25(corpus_texts, queries, gold_sets, cand_sets,
                      top_k=params["top_k"])
        mrr = [m["mrr"] for m in bm]
        ap = [m["ap"] for m in bm]
        results["bm25"] = {"mrr": round(float(sum(mrr) / len(mrr)), 4),
                           "ap": round(float(sum(ap) / len(ap)), 4),
                           "n": len(bm), "failed": 0}
        print(f"[bm25] MRR={results['bm25']['mrr']:.4f} AP={results['bm25']['ap']:.4f} "
              f"n={len(bm)}", flush=True)

    (OUT_BASE / "results.json").write_text(
        json.dumps({"run_dir": str(run_dir), "params": params, "results": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()