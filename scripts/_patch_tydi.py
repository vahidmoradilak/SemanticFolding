from pathlib import Path

p = Path(__file__).resolve().parents[1] / "scripts" / "build_pooled_multilingual.py"
t = p.read_text(encoding="utf-8")

start = t.index("    pos_ids = set()")
end = t.index("def build_rows")

new_block = '''    queries, need_ids = [], set()
    for rec in iter_gz_jsonl(dev):
        poss = [pp["docid"] for pp in rec.get("positive_passages", [])]
        if not poss:
            continue
        need_ids.update(poss)
        queries.append((rec["query_id"], rec["query"], poss))
        if len(queries) >= N_QUERIES:
            break

    rng = random.Random(SEED)
    pool, pos_text = [], {}
    for rec in iter_gz_jsonl(corp):
        did = rec.get("docid")
        if did in need_ids:
            pos_text[did] = rec.get("text", "")
        elif len(pos_text) == len(need_ids) and len(pool) < NEG_POOL \\
                and rng.random() < 0.01:
            txt = rec.get("text", "")
            if txt:
                pool.append(txt)

    queries = [(q, t, [(d, pos_text.get(d, "")) for d in ids]) for q, t, ids in queries]
    queries = [(q, t, pp) for q, t, pp in queries if all(txt for _, txt in pp)]
    print(f"tydi: queries={len(queries)} neg_pool={len(pool)}")
    return build_rows(queries, pool, "mrtydi_ar")


'''

t = t[:start] + new_block + t[end:]
p.write_text(t, encoding="utf-8")
print("patched do_tydi")
