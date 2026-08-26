"""Build pooled multilingual subsets: Mr.TyDi-ar & MIRACL-ar -> converted jsonl."""
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTBASE = ROOT / "data" / "datasets"
POOL_SIZE = int(os.environ.get("POOL_SIZE", "100"))
NEG_POOL = 5000
N_QUERIES = 200
SEED = 42


def iter_gz_jsonl(path):
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_rows(queries, neg_pool_texts, name):
    rng = random.Random(SEED)
    rows = []
    for qid, qtext, pos in queries[:N_QUERIES]:
        paras = [{"idx": i, "title": "", "text": t} for i, (_, t) in enumerate(pos)]
        seen = {t for _, t in pos}
        need = POOL_SIZE - len(paras)
        for t in rng.sample(neg_pool_texts, min(need + 10, len(neg_pool_texts))):
            if t in seen:
                continue
            seen.add(t)
            paras.append({"idx": len(paras), "title": "", "text": t})
            if len(paras) >= POOL_SIZE:
                break
        rows.append({"id": str(qid), "question": qtext,
                     "answer": pos[0][1][:120], "paragraphs": paras})
    out = OUTBASE / name / "converted"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{name}.jsonl"
    with open(dest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} queries x {len(rows[0]['paragraphs'])} -> {dest}")
    return dest


def sample_pool(stream_iter, exclude_ids, want=NEG_POOL, cap=400000, p=0.01, seed=SEED,
                idkey="docid"):
    rng = random.Random(seed)
    pool, scanned = [], 0
    for rec in stream_iter:
        scanned += 1
        did = rec.get(idkey)
        txt = rec.get("text", "")
        if not txt or did in exclude_ids:
            continue
        if len(pool) < want and rng.random() < p:
            pool.append(txt)
        if len(pool) >= want or scanned >= cap:
            break
    return pool


def do_tydi():
    from huggingface_hub import hf_hub_download
    corp = hf_hub_download("castorini/mr-tydi-corpus",
                           "mrtydi-v1.1-arabic/corpus.jsonl.gz", repo_type="dataset")
    dev = hf_hub_download("castorini/mr-tydi",
                          "mrtydi-v1.1-arabic/dev.jsonl.gz", repo_type="dataset")

    queries, need = [], {}
    for rec in iter_gz_jsonl(dev):
        poss = [pp["docid"] for pp in rec.get("positive_passages", [])]
        if not poss:
            continue
        for d in poss:
            need.setdefault(d, "")
        queries.append((rec["query_id"], rec["query"], poss))
        if len(queries) >= N_QUERIES:
            break

    pool = []
    rng = random.Random(SEED + 1)
    for rec in iter_gz_jsonl(corp):
        d = rec.get("docid")
        if d in need:
            need[d] = rec.get("text", "")
        elif len(pool) < NEG_POOL and rng.random() < 0.0025:
            txt = rec.get("text", "")
            if txt:
                pool.append(txt)
        if len(pool) >= NEG_POOL:
            break

    ok = [(q, t, [(d, need[d]) for d in ids]) for q, t, ids in queries]
    ok = [(q, t, pp) for q, t, pp in ok if all(x for _, x in pp)]
    print(f"tydi: queries={len(ok)} neg_pool={len(pool)}")
    return build_rows(ok, pool, "mrtydi_ar")


def do_miracl():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    devf = hf_hub_download("miracl/miracl", "ar/dev/0000.parquet",
                           repo_type="dataset", revision="refs/convert/parquet")
    dev = pd.read_parquet(devf)
    queries, exclude = [], set()
    for _, rec in dev.iterrows():
        poss = [(pp["docid"], (str(pp.get("title", "")) + " " + str(pp["text"])).strip())
                for pp in rec["positive_passages"] if pp.get("text")]
        if not poss:
            continue
        exclude.update(d for d, _ in poss)
        queries.append((rec["query_id"], rec["query"], poss))
        if len(queries) >= N_QUERIES:
            break

    rng = random.Random(SEED + 2)
    pool = []
    for shard in ["ar/train/0000.parquet"]:
        f = hf_hub_download("miracl/miracl-corpus", shard,
                            repo_type="dataset", revision="refs/convert/parquet")
        df = pd.read_parquet(f, columns=["docid", "title", "text"])
        mask = ~df["docid"].isin(exclude)
        sub = df[mask].sample(n=min(NEG_POOL * 3, int(mask.sum())), random_state=SEED)
        for t in (sub["title"].fillna("") + " " + sub["text"].fillna("")).str.strip():
            if t and len(pool) < NEG_POOL:
                pool.append(t)
            elif len(pool) >= NEG_POOL:
                break
    print(f"miracl: queries={len(queries)} neg_pool={len(pool)}")
    ok = [(q, t, pp) for q, t, pp in queries]
    return build_rows(ok, pool, "miracl_ar")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "tydi"):
        do_tydi()
    if which in ("both", "miracl"):
        do_miracl()
