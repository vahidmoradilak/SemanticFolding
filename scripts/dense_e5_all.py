"""E5-multilingual dense baseline across all existing benchmarks (task A).

Questions come from each canonical run dir (queries.txt when present,
otherwise the converted jsonl / MuSiQue results_log.csv). Gold + candidate
universe are exactly those of the published SF/BM25 numbers.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_paths = ROOT / "semantic_folding" / "dataset_benchmark" / "custom_ar_en"
import sys
sys.path.insert(0, str(sys_paths))
from run_benchmark import compute_metrics  # noqa: E402

MODEL = "intfloat/multilingual-e5-small"

QURAN_RUN = sorted((ROOT / "outputs" / "quran_benchmark" / "runs").glob("run_*"),
                   key=lambda p: p.name)[-1]

DATASETS = [
    ("SciFact",  ROOT / "outputs/scifact_benchmark/runs/run_20260719_113649",
     ROOT / "data/datasets/scifact/converted/scifact.jsonl", "jsonl"),
    ("nfcorpus", ROOT / "outputs/nfcorpus_benchmark/runs/run_20260719_102758",
     ROOT / "data/datasets/nfcorpus/converted/nfcorpus.jsonl", "jsonl"),
    ("Belebele", ROOT / "outputs/belebele_benchmark/runs/run_20260717_154235",
     ROOT / "data/datasets/belebele/converted/belebele_eng_Latn.jsonl", "jsonl"),
    ("PubMedQA", ROOT / "outputs/pubmedqa_benchmark/runs/run_20260717_161150",
     ROOT / "data/datasets/pubmedqa/converted/pubmedqa_pqa_labeled.jsonl", "jsonl"),
    ("PopQA(500)", ROOT / "outputs/popqa_benchmark/runs/run_20260718_065633",
     ROOT / "data/datasets/popqa/converted/popqa.jsonl", "jsonl"),
    ("MuSiQue", ROOT / "outputs/musique_benchmark/runs/run_20260710_162617",
     ROOT / "outputs/musique_benchmark/benchmarks/benchmark_20260710_175934/results_log.csv", "csv"),
    ("AR-EN(488)", ROOT / "outputs/custom_ar_en_benchmark/runs/run_20260818_100234",
     None, "queriestxt"),
    ("Quran", QURAN_RUN,
     ROOT / "data/quran/quran_qa.jsonl", "quran"),
]


def load_gold_quran(src: Path):
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    return ([r["question"] for r in rows],
            {str(i): r["relevant"] for i, r in enumerate(rows)})


import re

DOC_RE = re.compile(r"^doc_\d{6},")


def load_corpus(run: Path, single_line: bool = False):
    """Corpus lines may wrap across physical lines; a new record starts at doc_XXXXXX,
    unless single_line (e.g., Quran: every line = one ayah with numeric id)."""
    docs, cur = [], None
    for ln in open(run / "corpus.txt", encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip():
            continue
        starts = (not single_line) and bool(DOC_RE.match(ln))
        if single_line or starts:
            did, txt = ln.split(",", 1)
            if cur and not single_line:
                docs.append(cur)
            cur = (did.strip(), txt.strip())
            if single_line:
                docs.append(cur); cur = None
        elif cur is not None:
            cur = (cur[0], (cur[1] + " " + ln.strip()).strip())
    if cur:
        docs.append(cur)
    return docs


def load_questions(run: Path, src: Path, kind: str, n_gold: int):
    qt = run / "queries.txt"
    if kind == "queriestxt" or (qt.exists() and sum(1 for _ in open(qt, encoding="utf-8") if _.strip()) == n_gold):
        return [l.rstrip("\n") for l in open(qt, encoding="utf-8") if l.strip()]
    if kind == "jsonl":
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        return [rows[int(k)]["question"] for k in sorted((int(x) for x in
                 json.load(open(run / "query_gold.json", encoding="utf-8")).keys()))]
    if kind == "csv":  # musique results_log.csv: col0=qid col1=question
        import csv
        rows = list(csv.reader(open(src, encoding="utf-8")))
        return {int(r[0]): r[1] for r in rows}


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    out = {}
    for name, run, src, kind in DATASETS:
        if kind == "quran":
            questions, gold = load_gold_quran(src)
            active = sorted(int(k) for k in gold.keys())
        else:
            gold = json.load(open(run / "query_gold.json", encoding="utf-8"))
            active = sorted(int(k) for k in gold.keys())
            questions = load_questions(run, src, kind, len(active))
        if isinstance(questions, dict):
            questions = [questions[i] for i in active]
        assert len(questions) == len(active), f"{name}: q={len(questions)} gold={len(active)}"

        docs = load_corpus(run, single_line=(kind == "quran"))
        doc_ids = [d for d, _ in docs]

        D = model.encode([f"passage: {t}" for _, t in docs], batch_size=128,
                         convert_to_numpy=True, normalize_embeddings=True)
        Q = model.encode([f"query: {q}" for q in questions], batch_size=128,
                         convert_to_numpy=True, normalize_embeddings=True)
        S = Q @ D.T

        rr, h1, nd = [], [], []
        for qi, gi in enumerate(active):
            order = np.argsort(-S[qi])[:20]
            retrieved = [(doc_ids[r], float(S[qi][r])) for r in order]
            m = compute_metrics(retrieved, gold.get(str(gi), []))
            fa = m["found_at"]
            rr.append(1.0 / fa if fa else 0.0)
            h1.append(1 if fa == 1 else 0)
            nd.append(m.get("ndcg@20", 0.0))

        res = {"mrr": round(float(np.mean(rr)), 4),
               "hit1": round(float(np.mean(h1)), 4),
               "ndcg20": round(float(np.mean(nd)), 4)}
        out[name] = res
        print(f"{name:12s} docs={len(docs):>6} q={len(active):>4} -> {res}")

    dest = ROOT / "outputs" / "dense_e5_summary.json"
    json.dump({"model": MODEL, "results": out}, open(dest, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("saved ->", dest)


if __name__ == "__main__":
    main()
