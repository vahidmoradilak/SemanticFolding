"""
Bilingual (Arabic+English) Belebele benchmark.
Converts bilingual_arb_eng.jsonl to MuSiQue format and runs SF benchmark.
"""
import json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_folding.dataset_benchmark.adapters import get_adapter
from semantic_folding.dataset_benchmark.generic_benchmark import GenericBenchmarkRunner

BILINGUAL_PATH = Path("data/datasets/belebele/raw/all/bilingual_arb_eng.jsonl")
CONVERTED_DIR = Path("data/datasets/belebele/converted")
OUT_PATH = CONVERTED_DIR / "belebele_bilingual_arb_eng.jsonl"

PIPELINE_PARAMS = {
    "grid_size": 64,
    "spreading_steps": 1,
    "top_k": 100,
    "weighting": "idf",
    "top_percent": 0.10,
    "smoothing_sigma": 1.5,
    "keep_verbs": True,
    "min_word_length": 3,
    "min_freq": 1,
    "morton": True,
    "method": "umap",
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.0,
    "random_seed": 42,
    "doc_norm": "l2",
}

NUM_QUERIES = 100

# ---------------------------------------------------------------
# Step 1: Convert bilingual JSONL to MuSiQue format
# ---------------------------------------------------------------
print("Converting bilingual JSONL to MuSiQue format...")

rows = []
with open(BILINGUAL_PATH, encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"  Loaded {len(rows)} bilingual records")

random.seed(42)
all_passages = [r.get("flores_passage", "") for r in rows]
answer_map = {1: "mc_answer1", 2: "mc_answer2", 3: "mc_answer3", 4: "mc_answer4"}

entries = []
n_written = 0
n_skipped = 0

for row in rows:
    if n_written >= NUM_QUERIES:
        break
    passage = row.get("flores_passage", "").strip()
    question = row.get("question", "").strip()
    correct_num = row.get("correct_answer_num", 0)
    if not passage or not question or not correct_num:
        n_skipped += 1
        continue
    correct_key = answer_map.get(correct_num, "mc_answer1")
    mc_answer = row.get(correct_key, "").strip()
    if not mc_answer:
        n_skipped += 1
        continue
    distractor_indices = random.sample(
        range(len(all_passages)),
        min(19, len(all_passages) - 1),
    )
    distractor_indices = [i for i in distractor_indices if all_passages[i] != passage][:19]
    paragraphs = [{
        "idx": 0,
        "title": "passage",
        "paragraph_text": passage,
        "is_supporting": True,
    }]
    for i, di in enumerate(distractor_indices):
        paragraphs.append({
            "idx": i + 1,
            "title": f"distractor_{i:04d}",
            "paragraph_text": all_passages[di],
            "is_supporting": False,
        })
    entries.append({
        "id": f"belebele_bilingual_{n_written:04d}",
        "question": question,
        "answer": mc_answer,
        "answer_num": correct_num,
        "dialect": "arb_Arab+eng_Latn",
        "language": "bilingual_arb_eng",
        "paragraphs": paragraphs,
    })
    n_written += 1

CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"  Wrote {n_written} entries -> {OUT_PATH} (skipped {n_skipped})")

# ---------------------------------------------------------------
# Step 2: Run benchmark
# ---------------------------------------------------------------
adapter = get_adapter("belebele", language="bilingual_arb_eng")

runner = GenericBenchmarkRunner(adapter, params=dict(PIPELINE_PARAMS))

print(f"\n  [Phase 1] Indexing {NUM_QUERIES} queries...")
t0 = time.time()
run_dir = runner.phase1_index(OUT_PATH, max_queries=NUM_QUERIES)
if not run_dir:
    print("  [ERROR] Index phase failed")
    sys.exit(1)
print(f"  Index done in {time.time()-t0:.0f}s -> {run_dir.name}")

print(f"\n  [Phase 2] Benchmarking {NUM_QUERIES} queries...")
t0 = time.time()
bench_dir = runner.phase2_benchmark(run_dir, OUT_PATH, query_start=0, query_end=NUM_QUERIES)
if not bench_dir:
    print("  [ERROR] Benchmark phase failed")
    sys.exit(1)
print(f"  Benchmark done in {time.time()-t0:.0f}s -> {bench_dir.name}")

# Read results
agg_path = bench_dir / "summary.json"
if agg_path.exists():
    with open(agg_path) as f:
        agg = json.load(f)
    print(f"\n  RESULTS: MRR={agg.get('mean_mrr',0):.4f}  AP={agg.get('mean_ap',0):.4f}")
    print(f"  Queries: {agg.get('num_queries',0)}  Failed: {agg.get('failed',0)}")
else:
    print(f"  [WARN] No summary.json at {agg_path}")

runner.phase3_report(bench_dir)
print("\nDone!")
