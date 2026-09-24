"""Compute Pure SF and SPLADE-only (linear alpha=0) for generic-format datasets.

Both methods are run on the SAME canonical run dir (same corpus, same query set,
same top_k from the run's config.yml) so results are directly comparable.
Also evaluates BM25 on the same run dir via bm25_benchmark.run_bm25_benchmark.

Usage:
    python scripts/bench_splade_only_generic.py [--datasets a b c] [--dry] [--skip-splade] [--skip-bm25]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantic_folding.dataset_benchmark.generic_benchmark import GenericBenchmarkRunner, logger
from semantic_folding.dataset_benchmark.adapters import get_adapter
from semantic_folding.dataset_benchmark.bm25_benchmark import run_bm25_benchmark

# dataset -> (run_dir_relative, jsonl_relative) using canonical "final" runs
DATASETS = {
    "belebele": ("outputs/belebele_benchmark/runs/run_20260727_133411",
                 "data/datasets/belebele/converted/belebele_eng_Latn.jsonl"),
    "narrativeqa": ("outputs/narrativeqa_benchmark/runs/run_20260718_155900",
                    "data/datasets/narrativeqa/converted/narrativeqa.jsonl"),
    "pubmedqa": ("outputs/pubmedqa_benchmark/runs/run_20260718_161316",
                 "data/datasets/pubmedqa/converted/pubmedqa_pqa_labeled.jsonl"),
    "popqa": ("outputs/popqa_benchmark/runs/run_20260718_163002",
              "data/datasets/popqa/converted/popqa.jsonl"),
    "scifact": ("outputs/scifact_benchmark/runs/run_20260719_113649",
                "data/datasets/scifact/converted/scifact.jsonl"),
    "scidocs": ("outputs/scidocs_benchmark/runs/run_20260719_161630",
                "data/datasets/scidocs/converted/scidocs.jsonl"),
    "nfcorpus": ("outputs/nfcorpus_benchmark/runs/run_20260719_092850",
                 "data/datasets/nfcorpus/converted/nfcorpus.jsonl"),
}


def load_run_pipeline(run_dir: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(open(run_dir / "config.yml", encoding="utf-8"))
    return cfg.get("pipeline", {})


def build_params(run_dir: Path, method: str) -> dict:
    """Params = saved run pipeline config + method override."""
    params = dict(load_run_pipeline(run_dir))
    if method == "pure":
        params["splade"] = False
        params.pop("corpus_path", None)
    elif method == "splade_only":
        params["splade"] = True
        params["fusion_method"] = "linear"
        params["hybrid_alpha"] = 0.0
        params["corpus_path"] = str(run_dir / "corpus.txt")
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--skip-splade", action="store_true")
    ap.add_argument("--skip-bm25", action="store_true")
    args = ap.parse_args()

    ds_names = args.datasets or list(DATASETS.keys())
    manifest = {}
    for ds in ds_names:
        rel_run, rel_jsonl = DATASETS[ds]
        run_dir = ROOT / rel_run
        jsonl = ROOT / rel_jsonl
        if not run_dir.exists() or not jsonl.exists():
            logger.error(f"skip {ds}: missing run/jsonl")
            continue

        adapter = get_adapter(ds)
        run_dir_abs = run_dir.resolve()
        entry = {"run": str(run_dir_abs), "jsonl": str(jsonl)}
        top_k = build_params(run_dir, "pure").get("top_k", 100)

        if args.dry:
            print(f"[DRY] {ds}: run={run_dir_abs} top_k={top_k}")
            continue

        # ── Pure SF ───────────────────────────────────────────────────
        params = build_params(run_dir, "pure")
        runner_pure = GenericBenchmarkRunner(adapter, params)
        print(f"\n==== {ds} PURE SF ====")
        bd = runner_pure.phase2_benchmark(run_dir_abs, jsonl)
        if bd:
            s = json.load(open(bd / "summary.json", encoding="utf-8"))
            entry["pure"] = {
                "dir": str(bd), "n": s.get("num_queries"),
                "mrr": s.get("mean_mrr"), "ap": s.get("mean_ap")}
            print(f"{ds} PureSF: MRR={s.get('mean_mrr'):.4f} AP={s.get('mean_ap'):.4f} "
                  f"n={s.get('num_queries')}")

        # ── SPLADE-only alpha=0 ───────────────────────────────────────
        if not args.skip_splade:
            params = build_params(run_dir, "splade_only")
            runner_sp = GenericBenchmarkRunner(adapter, params)
            print(f"\n==== {ds} SPLADE-ONLY alpha=0 ====")
            bd = runner_sp.phase2_benchmark(run_dir_abs, jsonl)
            if bd:
                s = json.load(open(bd / "summary.json", encoding="utf-8"))
                entry["splade_only"] = {
                    "dir": str(bd), "n": s.get("num_queries"),
                    "mrr": s.get("mean_mrr"), "ap": s.get("mean_ap")}
                print(f"{ds} SPLADE: MRR={s.get('mean_mrr'):.4f} AP={s.get('mean_ap'):.4f} "
                      f"n={s.get('num_queries')}")

        # ── BM25 (top_k defaults to run config top_k for parity) ──────
        if not args.skip_bm25:
            print(f"\n==== {ds} BM25 (top_k={top_k}) ====")
            bd = run_bm25_benchmark(ds, jsonl, run_dir_abs, query_start=0,
                                    query_end=None, top_k=top_k)
            if bd:
                s = json.load(open(bd / "summary.json", encoding="utf-8"))
                entry["bm25"] = {
                    "dir": str(bd), "n": s.get("num_queries"),
                    "failed": s.get("failed"),
                    "mrr": s.get("mean_mrr"), "ap": s.get("mean_ap")}
                print(f"{ds} BM25: MRR={s.get('mean_mrr'):.4f} AP={s.get('mean_ap'):.4f} "
                      f"n={s.get('num_queries')}")

        manifest[ds] = entry
        with open(ROOT / "outputs" / "splade_only_manifest.json", "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[saved] outputs/splade_only_manifest.json ({len(manifest)} datasets)")


if __name__ == "__main__":
    main()