"""
Comprehensive comparison: BM25 + SF+SPLADE on bilingual and 4 languages.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_folding.dataset_benchmark.adapters import get_adapter
from semantic_folding.dataset_benchmark.generic_benchmark import GenericBenchmarkRunner
from semantic_folding.dataset_benchmark.bm25_benchmark import run_bm25_benchmark as bm25_run

PIPELINE_BASE = {
    "grid_size": 64, "spreading_steps": 1, "top_k": 100,
    "weighting": "idf", "top_percent": 0.10, "smoothing_sigma": 1.5,
    "keep_verbs": True, "min_word_length": 3, "min_freq": 1,
    "morton": True, "method": "umap", "umap_n_neighbors": 15,
    "umap_min_dist": 0.0, "random_seed": 42, "doc_norm": "l2",
    "splade_model": "naver/splade-cocondenser-ensembledistil",
}

NUM_QUERIES = 100
SPLADE_TIMEOUT = 7200  # 2h per splade run

# ── bilingual paths ──
BILINGUAL_JSONL = Path("data/datasets/belebele/converted/belebele_bilingual_arb_eng.jsonl")
BILINGUAL_RUN_DIR = Path("outputs/belebele_benchmark/runs/run_20260727_154000")
BILINGUAL_RAW = Path("data/datasets/belebele/raw/all/bilingual_arb_eng.jsonl")

# ── 4 language run dirs (from latest SF-only runs) ──
LANG_RUN_DIRS = [
    ("eng_Latn", Path("outputs/belebele_benchmark/runs/run_20260727_133411")),
    ("fra_Latn", Path("outputs/belebele_benchmark/runs/run_20260727_133818")),
    ("arb_Arab", Path("outputs/belebele_benchmark/runs/run_20260727_134548")),
    ("pes_Arab", Path("outputs/belebele_benchmark/runs/run_20260727_135131")),
]
LANG_NAMES = {"eng_Latn": "English", "fra_Latn": "French", "arb_Arab": "Arabic", "pes_Arab": "Persian"}

results = {}  # key: lang_code, value: dict of method->{mrr,ap}

def read_summary(bench_dir):
    p = bench_dir / "summary.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return d.get("mean_mrr", 0), d.get("mean_ap", 0)
    return 0, 0

def run_splade(runner, run_dir, jsonl_path, fusion_method, label):
    print(f"\n  [{label}] SF+SPLADE ({fusion_method}) starting...")
    runner.params["splade"] = True
    runner.params["fusion_method"] = fusion_method
    runner.params["corpus_path"] = str(run_dir / "corpus.txt")
    t0 = time.time()
    bd = runner.phase2_benchmark(run_dir, jsonl_path, query_start=0, query_end=NUM_QUERIES)
    if not bd:
        print(f"  [{label}] SPLADE {fusion_method} FAILED")
        return 0, 0
    mrr, ap = read_summary(bd)
    print(f"  [{label}] SF+SPLADE ({fusion_method}) done in {time.time()-t0:.0f}s -> MRR={mrr:.4f} AP={ap:.4f}")
    return mrr, ap

def run_bm25(run_dir, jsonl_path, label):
    print(f"\n  [{label}] BM25 starting...")
    t0 = time.time()
    bd = bm25_run(dataset="belebele", jsonl_path=jsonl_path, run_dir=run_dir,
                   query_start=0, query_end=NUM_QUERIES, top_k=100)
    if not bd:
        print(f"  [{label}] BM25 FAILED")
        return 0, 0
    mrr, ap = read_summary(bd)
    print(f"  [{label}] BM25 done in {time.time()-t0:.0f}s -> MRR={mrr:.4f} AP={ap:.4f}")
    return mrr, ap

# ===================================================================
#  Bilingual: BM25 + SPLADE (linear & RRF)
# ===================================================================
print("=" * 70)
print("  BILINGUAL (Arabic+English)")
print("=" * 70)

adapter = get_adapter("belebele", language="bilingual_arb_eng")
runner = GenericBenchmarkRunner(adapter, params=dict(PIPELINE_BASE))

res = {}

# BM25
res["bm25_mrr"], res["bm25_ap"] = run_bm25(BILINGUAL_RUN_DIR, BILINGUAL_JSONL, "bilingual")

# SF+SPLADE linear
res["splade_lin_mrr"], res["splade_lin_ap"] = run_splade(
    GenericBenchmarkRunner(adapter, params=dict(PIPELINE_BASE)),
    BILINGUAL_RUN_DIR, BILINGUAL_JSONL, "linear", "bilingual")

# SF+SPLADE RRF (vectors already cached, should be fast)
res["splade_rrf_mrr"], res["splade_rrf_ap"] = run_splade(
    GenericBenchmarkRunner(adapter, params=dict(PIPELINE_BASE)),
    BILINGUAL_RUN_DIR, BILINGUAL_JSONL, "rrf", "bilingual")

results["bilingual"] = res

# ===================================================================
#  4 Languages: SPLADE linear only (best fusion method)
# ===================================================================
print("\n" + "=" * 70)
print("  4 LANGUAGES — SF+SPLADE linear")
print("=" * 70)

for lang_code, run_dir in LANG_RUN_DIRS:
    name = LANG_NAMES[lang_code]
    print(f"\n  --- {name} ({lang_code}) ---")

    adapter = get_adapter("belebele", language=lang_code)
    jsonl_path = Path(f"data/datasets/belebele/converted/belebele_{lang_code}.jsonl")

    runner = GenericBenchmarkRunner(adapter, params=dict(PIPELINE_BASE))
    mrr, ap = run_splade(runner, run_dir, jsonl_path, "linear", name)
    results[f"{lang_code}_splade"] = {"mrr": mrr, "ap": ap}

# ===================================================================
#  Report
# ===================================================================
print("\n\n" + "=" * 70)
print("  COMPREHENSIVE COMPARISON REPORT")
print("=" * 70)

# Bilingual comparison
print("\n--- Bilingual (Arabic+English) ---")
print(f"{'Method':25s} {'MRR':>8s} {'AP':>8s}")
print("-" * 45)
# SF results from earlier run
sf_mrr, sf_ap = 0.9733, 0.9733
print(f"{'Pure SF':25s} {sf_mrr:>8.4f} {sf_ap:>8.4f}")
print(f"{'BM25':25s} {res['bm25_mrr']:>8.4f} {res['bm25_ap']:>8.4f}")
print(f"{'SF+SPLADE linear':25s} {res['splade_lin_mrr']:>8.4f} {res['splade_lin_ap']:>8.4f}")
print(f"{'SF+SPLADE RRF':25s} {res['splade_rrf_mrr']:>8.4f} {res['splade_rrf_ap']:>8.4f}")
delta_bm25 = sf_mrr - res['bm25_mrr']
delta_splade_lin = sf_mrr - res['splade_lin_mrr']
delta_splade_rrf = sf_mrr - res['splade_rrf_mrr']
print(f"\nΔ vs SF:  BM25={delta_bm25:+.4f}  SPLADE(lin)={delta_splade_lin:+.4f}  SPLADE(rrf)={delta_splade_rrf:+.4f}")

# 4 languages comparison
print("\n--- 4 Languages: Pure SF vs SF+SPLADE linear ---")
print(f"{'Language':12s} {'SF MRR':>8s} {'SPLADE MRR':>10s} {'Δ':>8s}")
print("-" * 42)
sf_4lang = {
    "eng_Latn": (0.9950, 0.9950),
    "fra_Latn": (0.9350, 0.9350),
    "arb_Arab": (0.8521, 0.8521),
    "pes_Arab": (0.8731, 0.8731),
}
for lang_code, _ in LANG_RUN_DIRS:
    name = LANG_NAMES[lang_code]
    sf_mrr_l, sf_ap_l = sf_4lang[lang_code]
    spl = results.get(f"{lang_code}_splade", {})
    mrr_spl = spl.get("mrr", 0)
    delta = mrr_spl - sf_mrr_l
    print(f"{name:12s} {sf_mrr_l:>8.4f} {mrr_spl:>10.4f} {delta:>+8.4f}")

# Save results
out = Path("outputs/belebele_multilingual/full_comparison_results.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf8") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}")
print("\nDone!")
