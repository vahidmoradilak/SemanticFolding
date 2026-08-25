"""Dense-retrieval baseline (sentence-transformers MiniLM) on the SciFact benchmark run."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "semantic_folding" / "dataset_benchmark" / "custom_ar_en"))
from run_benchmark import compute_metrics  # noqa: E402

RUN = ROOT / "outputs" / "scifact_benchmark" / "runs" / "run_20260719_113649"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

rows = [json.loads(l) for l in open(ROOT / "data/datasets/scifact/converted/scifact.jsonl",
                                    encoding="utf-8")]
questions = [r["question"] for r in rows]

docs = []
for ln in open(RUN / "corpus.txt", encoding="utf-8"):
    if ln.strip():
        did, txt = ln.split(",", 1)
        docs.append((did.strip(), txt.strip()))
doc_ids = [d for d, _ in docs]
doc_texts = [t for _, t in docs]

gold = json.load(open(RUN / "query_gold.json", encoding="utf-8"))

from sentence_transformers import SentenceTransformer  # noqa: E402
model = SentenceTransformer(MODEL)
D = model.encode(doc_texts, batch_size=64, convert_to_numpy=True,
                 show_progress_bar=False, normalize_embeddings=True)
Q = model.encode(questions, batch_size=64, convert_to_numpy=True,
                 show_progress_bar=False, normalize_embeddings=True)
scores = Q @ D.T

rr, h1, nd = [], [], []
for qi in range(len(questions)):
    order = np.argsort(-scores[qi])[:20]
    retrieved = [(doc_ids[r], float(scores[qi][r])) for r in order]
    m = compute_metrics(retrieved, gold.get(str(qi), []))
    fa = m["found_at"]
    rr.append(1.0 / fa if fa else 0.0)
    h1.append(1 if fa == 1 else 0)
    nd.append(m.get("ndcg@20", 0.0))

res = {"model": MODEL, "n_queries": len(questions), "n_docs": len(docs),
       "mrr": round(float(np.mean(rr)), 4), "hit1": round(float(np.mean(h1)), 4),
       "ndcg20": round(float(np.mean(nd)), 4)}
print(json.dumps(res, indent=2))
out = ROOT / "outputs" / "scifact_benchmark" / "dense_baseline_summary.json"
json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
print("saved ->", out)
