"""Benchmark a pooled jsonl subset through SF pipeline + BM25 + E5."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "semantic_folding" / "dataset_benchmark" / "custom_ar_en"))
from run_benchmark import (STEP_SCRIPTS, compute_metrics, run_bm25,  # noqa: E402
                           run_step, run_step6_parallel)


def load_pooled(jsonl_path):
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    docs, key2id = [], {}
    q_doc_map, q_gold, questions = {}, {}, []
    for qi, r in enumerate(rows):
        cands, golds = [], []
        for pp in r["paragraphs"]:
            k = hashlib.sha1((pp["title"] + "\t" + pp["text"]).encode()).hexdigest()[:12]
            if k not in key2id:
                key2id[k] = f"doc_{len(docs):06d}"
                docs.append(pp["text"])
            did = key2id[k]
            if did not in cands:
                cands.append(did)
            if pp["idx"] == 0:
                golds.append(did)
        questions.append(r["question"])
        q_doc_map[str(qi)] = cands
        q_gold[str(qi)] = golds
    return rows, docs, questions, q_doc_map, q_gold


def main(jsonl_path, name):
    out = ROOT / "outputs" / f"{name}_benchmark" / "pooled"
    out.mkdir(parents=True, exist_ok=True)
    rows, docs, questions, cand_map, gold = load_pooled(ROOT / jsonl_path)

    corpus_path = out / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, t in enumerate(docs):
            f.write(f"doc_{i:06d}, passage {t}\n")
    with open(out / "queries.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(questions) + "\n")
    json.dump(cand_map, open(out / "query_doc_map.json", "w", encoding="utf-8"), indent=1)
    json.dump(gold, open(out / "query_gold.json", "w", encoding="utf-8"), indent=1)

    P = dict(grid=64, seed=42, nn=15, md=0.1, met="cosine", sigma=1.5,
             tp=0.10, mwl=2, mf=3)
    if (out / "sf_results.json").exists():
        print("[resume] index + SF results already present")
    else:
        ok = True
        ok &= run_step(STEP_SCRIPTS[1], ["--corpus", str(corpus_path), "--output",
                   str(out / "s1"), "--keep-verbs", "--min-word-length", str(P["mwl"]),
                   "--min-freq", str(P["mf"])], "S1", timeout=1800)
        ok &= run_step(STEP_SCRIPTS[2], ["--vocab", str(out / "s1/vocabulary.csv"),
                   "--mapping", str(out / "s1/phrase_to_contexts.json"), "--corpus",
                   str(corpus_path), "--output", str(out / "s2")], "S2", timeout=1800)
        ok &= run_step(STEP_SCRIPTS[3], ["--matrix", str(out / "s2/term_context_matrix.npz"),
                   "--metadata", str(out / "s2/term_context_matrix.json"), "--output",
                   str(out / "s3"), "--grid-size", str(P["grid"]), "--method", "tsne",
                   "--random-seed", str(P["seed"]), "--n-neighbors", str(P["nn"]),
                   "--min-dist", str(P["md"]), "--metric", P["met"]], "S3", timeout=3600)
        ok &= run_step(STEP_SCRIPTS[4], ["--coordinates", str(out / "s3/context_coordinates.json"),
                   "--metadata", str(out / "s2/term_context_matrix.json"), "--output",
                   str(out / "s4"), "--grid-size", str(P["grid"]), "--smoothing-sigma",
                   str(P["sigma"]), "--morton"], "S4", timeout=1800)
        ok &= run_step(STEP_SCRIPTS[5], ["--corpus", str(corpus_path), "--fingerprints",
                   str(out / "s4"), "--idf-weights", str(out / "s2/idf_weights.json"),
                   "--output", str(out / "s5"), "--grid-size", str(P["grid"]),
                   "--top-percent", str(P["tp"]), "--normalize-method", "l2",
                   "--min-word-length", str(P["mwl"]), "--smoothing-sigma", str(P["sigma"]),
                   "--min-peak-distance", "2", "--morton"], "S5", timeout=3600)
        if not ok:
            sys.exit("indexing failed")

        s7 = ["--query-file", str(out / "queries.txt"), "--fingerprints", str(out / "s4"),
              "--doc-fingerprints", str(out / "s5"), "--idf-weights",
              str(out / "s2/idf_weights.json"), "--grid-size", str(P["grid"]),
              "--top-k", "100", "--weighting", "idf", "--spreading-steps", "1",
              "--keep-verbs", "--min-word-length", str(P["mwl"])]
        run_step6_parallel(s7, out / "sf_results.json", "SF", num_workers=1)

    gold_sets = [gold[str(i)] for i in range(len(rows))]
    cand_sets = [cand_map[str(i)] for i in range(len(rows))]
    bm_metrics = run_bm25(docs, questions, gold_sets, cand_sets, top_k=20)

    sf_raw = json.load(open(out / "sf_results.json", encoding="utf-8"))

    def eval_ranked(get_ranked):
        rr, h1 = [], []
        for gi in range(len(rows)):
            filt = [(d, s) for d, s in get_ranked(gi)
                    if d in cand_set_list[gi]][:20]
            m = compute_metrics(filt, gold[str(gi)])
            fa = m["found_at"]
            rr.append(1.0 / fa if fa else 0.0)
            h1.append(1.0 if fa == 1 else 0.0)
        return {"mrr": round(float(np.mean(rr)), 4),
                "hit1": round(float(np.mean(h1)), 4)}

    cand_set_list = [set(c) for c in cand_sets]
    summary = {"n_queries": len(rows), "pool": len(cand_sets[0])}
    summary["sf"] = eval_ranked(lambda gi: sf_raw[gi].get("results", []))
    summary["bm25"] = {
        "mrr": round(float(np.mean([m["mrr"] for m in bm_metrics])), 4),
        "hit1": round(float(np.mean([m["p@1"] for m in bm_metrics])), 4)}

    # dense E5
    try:
        from sentence_transformers import SentenceTransformer
        mdl = SentenceTransformer("intfloat/multilingual-e5-small")
        Dv = mdl.encode([f"passage: {t}" for t in docs], batch_size=128,
                        convert_to_numpy=True, normalize_embeddings=True)
        Qv = mdl.encode([f"query: {q}" for q in questions], batch_size=128,
                        convert_to_numpy=True, normalize_embeddings=True)
        S = Qv @ Dv.T
        doc_ids_sorted = [f"doc_{i:06d}" for i in range(len(docs))]

        def e5_ranked(gi):
            order = np.argsort(-S[gi])[:100]
            return [(doc_ids_sorted[r], float(S[gi][r])) for r in order]

        summary["e5"] = eval_ranked(e5_ranked)
    except Exception as exc:
        summary["e5"] = {"error": str(exc)[:120]}

    from scipy.stats import wilcoxon

    def rr_vec(tag):
        out_v = []
        for gi in range(len(rows)):
            if tag == "bm25":
                out_v.append(bm_metrics[gi]["mrr"])
            else:
                raw = (sf_raw[gi].get("results", []) if tag == "sf" else e5_ranked(gi))
                filt = [(d, s) for d, s in raw if d in cand_set_list[gi]][:20]
                fa = compute_metrics(filt, gold[str(gi)])["found_at"]
                out_v.append(1.0 / fa if fa else 0.0)
        return np.array(out_v)

    for a, b in [("sf", "bm25"), ("e5", "bm25"), ("sf", "e5")]:
        va, vb = rr_vec(a), rr_vec(b)
        nz = (va - vb) != 0
        p = float(wilcoxon(va[nz], vb[nz]).pvalue) if nz.sum() >= 5 else None
        summary[f"p_{a}_vs_{b}"] = p
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
