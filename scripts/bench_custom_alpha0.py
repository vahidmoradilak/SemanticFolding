"""Pure SF / SPLADE-only (α=0) / BM25 for custom-format bilingual datasets.

Reuses an existing index run dir (queries.txt, gold, doc map, fingerprints,
cached SPLADE doc vectors) and evaluates three methods on the SAME inputs:
  - pure_sf      : step7 without SPLADE
  - splade_only  : step7 --splade --fusion-method linear --hybrid-alpha 0
  - bm25         : in-process BM25 (top_k from run)

Usage:
    python scripts/bench_custom_alpha0.py <run_dir> <out_dir> [--top-k N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "semantic_folding/dataset_benchmark/custom_ar_en"))

from run_benchmark import (
    compute_metrics, aggregate, run_bm25, run_step6_parallel,
    load_corpus, load_queries, load_gold,
)

STEP7 = ROOT / "semantic_folding/query_processor.py"


def build_step7_args(run_dir: Path, corpus_path: Path, params: dict, use_splade: bool):
    args = [
        "--query-file", str(run_dir / "queries.txt"),
        "--fingerprints", str(run_dir / "phrase_fingerprints"),
        "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
        "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
        "--grid-size", str(params["grid_size"]),
        "--top-k", str(params["top_k"]),
        "--weighting", params["weighting"],
        "--spreading-steps", str(params["spreading_steps"]),
        "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
    ]
    if use_splade:
        args += [
            "--splade", "--splade-model", "naver/splade-cocondenser-ensembledistil",
            "--fusion-method", "linear",
            "--hybrid-alpha", "0.0",
            "--corpus", str(corpus_path),
        ]
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--weighting", default="idf")
    ap.add_argument("--spreading-steps", type=int, default=1)
    ap.add_argument("--min-word-length", type=int, default=2)
    ap.add_argument("--skip-splade", action="store_true")
    ap.add_argument("--skip-bm25", action="store_true")
    ap.add_argument("--skip-pure", action="store_true")
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()

    # Load run inputs
    docs = load_corpus(run_dir / "corpus.txt")
    queries = [ln.rstrip("\n") for ln in open(run_dir / "queries.txt", encoding="utf-8") if ln.strip()]
    query_gold = json.load(open(run_dir / "query_gold.json", encoding="utf-8"))
    query_doc_map = json.load(open(run_dir / "query_doc_map.json", encoding="utf-8"))
    active_indices = sorted(int(k) for k in query_gold.keys())
    doc_ids = [d["doc_id"] for d in docs]
    corpus_texts = [d["text"] for d in docs]
    corpus_path = run_dir / "corpus.txt"

    params = {
        "grid_size": args.grid_size,
        "weighting": args.weighting,
        "spreading_steps": args.spreading_steps,
        "min_word_length": args.min_word_length,
        "top_k": args.top_k or 100,
    }
    top_k = params["top_k"]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    methods = []
    if not args.skip_pure:
        methods.append("pure_sf")
    if not args.skip_splade:
        methods.append("splade_only")

    for vname in methods:
        use_splade = vname == "splade_only"
        cmd = build_step7_args(run_dir, corpus_path, params, use_splade)
        output_json = out_dir / f"{vname}.json"
        print(f"\n==== {vname} (splade={use_splade}) ====")
        t0 = time.time()
        ok = run_step6_parallel(cmd, output_json, f"Step 7 {vname}",
                                num_workers=args.parallel, timeout=3600)
        elapsed = time.time() - t0
        if not ok:
            print(f"  {vname} FAILED after {elapsed:.0f}s")
            manifest[vname] = {"error": "failed"}
            continue
        all_results = json.load(open(output_json, encoding="utf-8"))
        var_metrics = []
        for i in active_indices:
            scores = all_results[i] if i < len(all_results) else None
            if scores is None:
                continue
            raw = scores.get("results", []) if isinstance(scores, dict) else scores
            cand = set(query_doc_map.get(str(i), []))
            filtered = [(d, s) for d, s in raw if d in cand][:top_k]
            var_metrics.append(compute_metrics(filtered, query_gold.get(str(i), [])))
        summary = aggregate(var_metrics)
        summary["elapsed_s"] = round(elapsed, 1)
        with open(out_dir / f"{vname}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        manifest[vname] = {k: v for k, v in summary.items() if k in
                           ("num_queries", "mean_mrr", "mean_ap", "mean_p@1", "elapsed_s")}
        print(f"  {vname}: MRR={summary['mean_mrr']:.4f} AP={summary['mean_ap']:.4f} "
              f"n={summary.get('num_queries')} ({elapsed:.0f}s)")

    if not args.skip_bm25:
        print(f"\n==== bm25 (top_k={top_k}) ====")
        t0 = time.time()
        gold_sets = [query_gold[str(i)] for i in active_indices]
        cand_sets = [query_doc_map[str(i)] for i in active_indices]
        active_queries = [queries[i] for i in active_indices]
        bm25_metrics = run_bm25(corpus_texts, active_queries, gold_sets, cand_sets, top_k=top_k)
        bm25_summary = aggregate(bm25_metrics)
        bm25_summary["elapsed_s"] = round(time.time() - t0, 1)
        with open(out_dir / "bm25_summary.json", "w", encoding="utf-8") as f:
            json.dump(bm25_summary, f, indent=2)
        manifest["bm25"] = {k: v for k, v in bm25_summary.items() if k in
                            ("num_queries", "mean_mrr", "mean_ap", "mean_p@1", "elapsed_s")}
        print(f"  bm25: MRR={bm25_summary['mean_mrr']:.4f} AP={bm25_summary['mean_ap']:.4f} "
              f"n={bm25_summary.get('num_queries')}")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir), "params": params, "results": manifest},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()