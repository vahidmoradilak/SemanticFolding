"""SPLADE-only (α=0) benchmark for pooled pipelines (Mr.TyDi-ar, MIRACL-ar).

Runs step7 in batch mode against the pooled index (same params as
bench_pooled.py: grid=64, tsne, tp=0.10, mwl=2, top_k=100, weighting=idf),
plus `--splade --fusion-method linear --hybrid-alpha 0 --corpus`.

Usage:
    python scripts/bench_pooled_splade.py <pooled_dir> [--pure] [--dry]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "semantic_folding/dataset_benchmark/custom_ar_en"))
sys.path.insert(0, str(ROOT))

from run_benchmark import run_step6_parallel, compute_metrics
import numpy as np

STEP7 = ROOT / "semantic_folding/query_processor.py"

P = dict(grid=64, tp=0.10, mwl=2, top_k=100)


def build_args(out: Path, use_splade: bool):
    args = [
        "--query-file", str(out / "queries.txt"),
        "--fingerprints", str(out / "s4"),
        "--doc-fingerprints", str(out / "s5"),
        "--idf-weights", str(out / "s2" / "idf_weights.json"),
        "--grid-size", str(P["grid"]),
        "--top-k", str(P["top_k"]),
        "--weighting", "idf",
        "--spreading-steps", "1",
        "--keep-verbs",
        "--min-word-length", str(P["mwl"]),
    ]
    if use_splade:
        args += [
            "--splade",
            "--splade-model", "naver/splade-cocondenser-ensembledistil",
            "--fusion-method", "linear",
            "--hybrid-alpha", "0.0",
            "--corpus", str(out / "corpus.txt"),
        ]
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pooled_dir", type=Path)
    ap.add_argument("--pure", action="store_true",
                    help="also run pure SF (default only splade)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    out = args.pooled_dir.resolve()
    gold = json.load(open(out / "query_gold.json", encoding="utf-8"))
    cand_map = json.load(open(out / "query_doc_map.json", encoding="utf-8"))
    rows = [str(i) for i in range(len(gold))]

    cand_sets = [set(cand_map[r]) for r in rows]
    gold_sets = [gold[r] for r in rows]

    def evaluate(results_json, tag):
        raw = json.load(open(results_json, encoding="utf-8"))
        rrs, h1s, aps = [], [], []
        n_eval = 0
        for gi, r in enumerate(rows):
            if gi >= len(raw):
                break
            res = raw[gi].get("results", []) if isinstance(raw[gi], dict) else []
            filt = [(d, s) for d, s in res if d in cand_sets[gi]][:20]
            m = compute_metrics(filt, gold_sets[gi])
            fa = m["found_at"]
            rrs.append(1.0 / fa if fa else 0.0)
            h1s.append(1.0 if fa == 1 else 0.0)
            aps.append(m["ap"])
            n_eval += 1
        return {"n_eval": n_eval,
                "mrr": round(float(np.mean(rrs)), 4),
                "hit1": round(float(np.mean(h1s)), 4),
                "ap": round(float(np.mean(aps)), 4)}

    if args.dry:
        print(f"[DRY] {out.name}: splade={'ab' in ''} dry run")
        return

    if args.pure:
        t0 = time.time()
        print("=== pure SF ===")
        run_step6_parallel(build_args(out, False), out / "sf_results.json",
                           "SF", num_workers=1)
        s = evaluate(out / "sf_results.json", "sf")
        print(f"pure SF: {s} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    print("=== SPLADE-only α=0 ===")
    timeout = int(os.environ.get("POOLED_SPLADE_TIMEOUT", "3600"))
    run_step6_parallel(build_args(out, True), out / "splade_alpha0_results.json",
                       "SPLADE", num_workers=1, timeout=timeout)
    s = evaluate(out / "splade_alpha0_results.json", "splade")
    print(f"SPLADE α=0: {s} ({time.time()-t0:.0f}s)")
    json.dump(s, open(out / "splade_alpha0_summary.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()