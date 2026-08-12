"""
Benchmark: Mixed Arabic-English queries against a user-supplied bilingual corpus.

Accepts three external files:
  --corpus  corpus.txt   (doc_XXXXXX, passage <arabic> | <english>) per line
  --queries queries.jsonl ({"id": "...", "question": "..."} per line)
  --gold    gold.json    ({"<query_id>": ["doc_XXXXXX", ...]})

Each query is turned into a mixed-language query string via SF phrase
extraction (Arabic + English), then retrieval runs against the bilingual
corpus. Candidates for every query = all corpus documents.

Variants: Pure SF, SF+SPLADE Linear, SF+SPLADE RRF, BM25
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "semantic_folding"))

from lib import get_logger, extract_raw_phrases_ar_fa, normalize_arabic_phrase
from lib import detect_language as _detect_lang

logger = get_logger("custom_ar_en")

# ── spaCy setup ─────────────────────────────────────────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    nlp = None

# ── Constants ───────────────────────────────────────────────────────────────
_OUTPUTS = PROJECT_ROOT / "outputs" / "custom_ar_en_benchmark"
_SEP = " | "

STEP_SCRIPTS = {
    1: PROJECT_ROOT / "semantic_folding" / "phrase_extractor.py",
    2: PROJECT_ROOT / "semantic_folding" / "term_context.py",
    3: PROJECT_ROOT / "semantic_folding" / "semantic_space.py",
    4: PROJECT_ROOT / "semantic_folding" / "phrase_fingerprints.py",
    5: PROJECT_ROOT / "semantic_folding" / "doc_fingerprints.py",
    6: PROJECT_ROOT / "semantic_folding" / "query_processor.py",
}

DEFAULT_PARAMS = {
    "grid_size": 64,
    "method": "umap",
    "random_seed": 42,
    "perplexity": 50,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "umap_metric": "cosine",
    "tsne_iter": 1000,
    "morton": True,
    "smoothing_sigma": 1.5,
    "top_percent": 0.10,
    "weighting": "idf",
    "spreading_steps": 1,
    "min_word_length": 2,
    "min_freq": 1,
    "keep_verbs": True,
    "top_k": 20,
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Query generation (SF phrase extraction)
# ═════════════════════════════════════════════════════════════════════════════

def _clean_ar_phrase(p: str) -> str:
    p = normalize_arabic_phrase(p)
    if not p:
        return ""
    p = re.sub(r'[^\w\s\u0600-\u06FF]', '', p).strip()
    return p


def _extract_ar_phrases(text: str) -> List[str]:
    raw = extract_raw_phrases_ar_fa(text)
    valid = []
    for p in raw:
        clean = _clean_ar_phrase(p)
        if clean and len(clean) >= 2:
            valid.append(clean)
    return valid


def _extract_en_phrases(text: str) -> List[str]:
    if not SPACY_OK or not nlp:
        return text.split()[:10]
    from phrase_extractor import extract_raw_phrases_spacy, normalize_hyphens
    clean = normalize_hyphens(text)
    doc = nlp(clean)
    return extract_raw_phrases_spacy(doc)


def make_mixed_query(question: str) -> str:
    parts = question.split(_SEP, maxsplit=1)
    if len(parts) < 2:
        return question
    ar_q, en_q = parts[0].strip(), parts[1].strip()
    ar_phrases = _extract_ar_phrases(ar_q)
    en_phrases = _extract_en_phrases(en_q)
    mixed = " ".join(ar_phrases + en_phrases)
    return mixed if mixed.strip() else question


# ═════════════════════════════════════════════════════════════════════════════
# 2. Input loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_corpus(corpus_path: Path) -> List[Dict[str, str]]:
    """Parse corpus.txt lines: 'doc_XXXXXX, passage <text>'."""
    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "," not in line:
                logger.warning(f"[CORPUS] Skipping line without comma: {line[:60]}")
                continue
            doc_id, text = line.split(",", 1)
            docs.append({"doc_id": doc_id.strip(), "text": text.strip()})
    logger.info(f"Loaded {len(docs)} documents from {corpus_path}")
    return docs


def load_queries(queries_path: Path) -> List[Dict[str, str]]:
    """Parse queries.jsonl: each line {"id": "...", "question": "..."}."""
    queries = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    logger.info(f"Loaded {len(queries)} queries from {queries_path}")
    return queries


def load_gold(gold_path: Path) -> Dict[str, List[str]]:
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)
    logger.info(f"Loaded gold for {len(gold)} queries from {gold_path}")
    return gold


# ═════════════════════════════════════════════════════════════════════════════
# 3. Pipeline helpers
# ═════════════════════════════════════════════════════════════════════════════

def run_step(script: Path, args: List[str], step_name: str, timeout: int = 1800) -> bool:
    cmd = [sys.executable, str(script)] + [a for a in args if a]
    logger.info(f"[{step_name}] starting: {' '.join(str(a) for a in cmd)}")
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace", env=env,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr[-1000:].replace("\n", " | ")
            logger.error(f"[{step_name}] FAILED (rc={result.returncode}): {stderr_tail}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"[{step_name}] TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"[{step_name}] ERROR: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 4. Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(retrieved, relevant, top_k_list=None):
    if top_k_list is None:
        top_k_list = [1, 2, 3, 5, 20]
    retrieved_ids = [doc_id for doc_id, _ in retrieved]
    rel_set = set(relevant)

    found_at = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            found_at = rank
            break

    mrr = 1.0 / found_at if found_at else 0.0

    hits = 0
    prec_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            hits += 1
            prec_sum += hits / rank
    ap = prec_sum / len(rel_set) if rel_set else 0.0

    metrics = {"mrr": mrr, "ap": ap, "found_at": found_at}
    for k in top_k_list:
        top_k = retrieved_ids[:k]
        n_rel = sum(1 for d in top_k if d in rel_set)
        metrics[f"p@{k}"] = n_rel / k
        metrics[f"r@{k}"] = n_rel / len(rel_set) if rel_set else 0.0

        dcg = sum(1.0 / np.log2(rank + 1) for rank, d in enumerate(top_k, 1) if d in rel_set)
        ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(rel_set), k) + 1))
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal > 0 else 0.0

    return metrics


def aggregate(metrics_list):
    agg = defaultdict(list)
    for m in metrics_list:
        for k, v in m.items():
            agg[k].append(v)
    summary = {"num_queries": len(metrics_list)}
    for k, vals in agg.items():
        summary[f"mean_{k}"] = sum(vals) / len(vals)
        summary[f"min_{k}"] = min(vals)
        summary[f"max_{k}"] = max(vals)
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# 5. BM25
# ═════════════════════════════════════════════════════════════════════════════

def run_bm25(corpus_texts, queries_texts, gold_sets, candidate_sets, top_k=20):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        max_features=50000,
    )
    cv = vectorizer.fit_transform(corpus_texts)
    doc_len = cv.sum(axis=1).A.ravel()
    avg_dl = doc_len.mean()
    n_docs = len(corpus_texts)
    df = (cv > 0).sum(axis=0).A.ravel()
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    k1, b = 1.2, 0.75
    results = []
    for q_text in queries_texts:
        qv = vectorizer.transform([q_text])
        q_indices = qv.indices
        q_data = qv.data

        scores = np.zeros(n_docs, dtype=np.float64)
        for idx, weight in zip(q_indices, q_data):
            tf = cv[:, idx].toarray().ravel()
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_dl)
            scores += weight * idf[idx] * numerator / (denominator + 1e-10)

        results.append(scores)

    all_metrics = []
    for i, scores in enumerate(results):
        sorted_idx = np.argsort(scores)[::-1]
        ranked = [(f"doc_{idx:06d}", float(scores[idx])) for idx in sorted_idx if scores[idx] > 0]
        cand_set = set(candidate_sets[i])
        filtered = [(d, s) for d, s in ranked if d in cand_set][:top_k]
        metrics = compute_metrics(filtered, gold_sets[i])
        all_metrics.append(metrics)

    return all_metrics


# ═════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═════════════════════════════════════════════════════════════════════════════

def _retrieval_backend_args(params):
    if params.get("retrieval_backend") != "lancedb":
        return []
    args = ["--retrieval-backend", "lancedb"]
    if params.get("lancedb_path"):
        args.extend(["--lancedb-path", str(params["lancedb_path"])])
    if params.get("lancedb_exact"):
        args.append("--lancedb-exact")
    if params.get("lancedb_limit", 200) != 200:
        args.extend(["--lancedb-limit", str(params["lancedb_limit"])])
    return args


def run_step6_parallel(step6_args, output_json, step_name, num_workers=1,
                       timeout=3600) -> bool:
    """Run Step 6 across ``num_workers`` concurrent subprocesses.

    The query file is sharded evenly (one shard per worker); each worker runs
    query_processor.py on its own shard + a separate output file, then the
    per-query results are merged (in original order) into ``output_json``.

    With ``num_workers <= 1`` this is a plain single subprocess run.
    """
    if "--query-file" not in step6_args:
        return run_step(STEP_SCRIPTS[6], step6_args + ["--output", str(output_json)],
                        step_name, timeout=timeout)

    qf_idx = step6_args.index("--query-file")
    query_file = Path(step6_args[qf_idx + 1])
    lines = [ln.rstrip("\n") for ln in open(query_file, encoding="utf-8") if ln.strip()]
    n_total = len(lines)

    if n_total == 0:
        logger.error(f"[{step_name}] empty query file: {query_file}")
        return False
    if num_workers <= 1:
        return run_step(STEP_SCRIPTS[6], step6_args + ["--output", str(output_json)],
                        step_name, timeout=timeout)

    import concurrent.futures as cf
    import shutil

    num_workers = min(num_workers, n_total)
    shard_dir = query_file.parent / f"_shards_{datetime.now().strftime('%H%M%S')}_{os.getpid()}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{step_name}] parallel x{num_workers}: {n_total} queries "
                f"-> {shard_dir.name}/")

    # Remaining shared args (drop the original --query-file flag AND its path)
    base_args = []
    _skip_next = False
    for _a in step6_args:
        if _a == "--query-file":
            _skip_next = True
            continue
        if _skip_next:
            _skip_next = False
            continue
        base_args.append(_a)

    shard_specs = []
    per = int(np.ceil(n_total / num_workers))
    for a in range(0, n_total, per):
        shard_specs.append((a, min(a + per, n_total)))

    def _run_one(a, b):
        shard_q = shard_dir / f"queries_{a:06d}.txt"
        shard_out = shard_dir / f"out_{a:06d}.json"
        with open(shard_q, "w", encoding="utf-8") as f:
            for j in range(a, b):
                f.write(lines[j] + "\n")
        sub_args = base_args + ["--query-file", str(shard_q), "--output", str(shard_out)]
        ok = run_step(STEP_SCRIPTS[6], sub_args,
                      f"{step_name} [shard {a}-{b}]", timeout=timeout)
        return (a, ok, shard_out)

    merged = [None] * n_total
    with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(_run_one, a, b) for a, b in shard_specs]
        for fut in cf.as_completed(futs):
            a, ok, shard_out = fut.result()
            if not ok:
                logger.error(f"[{step_name}] parallel shard starting at {a} FAILED")
                shutil.rmtree(shard_dir, ignore_errors=True)
                return False
            shard_data = json.load(open(shard_out, encoding="utf-8"))
            for offset, entry in enumerate(shard_data):
                merged[a + offset] = entry

    if any(e is None for e in merged):
        logger.error(f"[{step_name}] merged results incomplete ({merged.count(None)} missing)")
        shutil.rmtree(shard_dir, ignore_errors=True)
        return False

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    shutil.rmtree(shard_dir, ignore_errors=True)
    logger.info(f"[{step_name}] merged {n_total} results -> {output_json.name}")
    return True


def build_variants(query_file, run_dir, corpus_path, params):
    rba = _retrieval_backend_args(params)
    return {
        "pure_sf": {
            "display": "Pure SF",
            "step7_args": [
                "--query-file", str(query_file),
                "--fingerprints", str(run_dir / "phrase_fingerprints"),
                "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
                "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
                "--grid-size", str(params["grid_size"]),
                "--top-k", str(params["top_k"]),
                "--weighting", params["weighting"],
                "--spreading-steps", str(params["spreading_steps"]),
                "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
            ] + rba,
        },
        "splade_linear": {
            "display": "SF+SPLADE Linear (\u03b1=0.3)",
            "step7_args": [
                "--query-file", str(query_file),
                "--fingerprints", str(run_dir / "phrase_fingerprints"),
                "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
                "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
                "--grid-size", str(params["grid_size"]),
                "--top-k", str(params["top_k"]),
                "--weighting", params["weighting"],
                "--spreading-steps", str(params["spreading_steps"]),
                "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
                "--splade", "--splade-model", "naver/splade-cocondenser-ensembledistil",
                "--hybrid-alpha", "0.3",
                "--corpus", str(corpus_path),
            ] + rba,
        },
        "splade_rrf": {
            "display": "SF+SPLADE RRF (k=60)",
            "step7_args": [
                "--query-file", str(query_file),
                "--fingerprints", str(run_dir / "phrase_fingerprints"),
                "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
                "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
                "--grid-size", str(params["grid_size"]),
                "--top-k", str(params["top_k"]),
                "--weighting", params["weighting"],
                "--spreading-steps", str(params["spreading_steps"]),
                "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
                "--splade", "--splade-model", "naver/splade-cocondenser-ensembledistil",
                "--hybrid-alpha", "0.3",
                "--fusion-method", "rrf", "--rrf-k", "60",
                "--corpus", str(corpus_path),
            ] + rba,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Custom bilingual AR|EN benchmark runner")
    parser.add_argument("--corpus", type=Path, required=True, help="corpus.txt: 'doc_XXXXXX, passage <arabic> | <english>'")
    parser.add_argument("--queries", type=Path, required=True, help="queries.jsonl: {\"id\": \"...\", \"question\": \"...\"}")
    parser.add_argument("--gold", type=Path, required=True, help="gold.json: {\"<query_id>\": [\"doc_XXXXXX\", ...]}")
    parser.add_argument("--top-k", type=int, default=DEFAULT_PARAMS["top_k"])
    parser.add_argument("--grid-size", type=int, default=DEFAULT_PARAMS["grid_size"])
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default=DEFAULT_PARAMS["method"])
    parser.add_argument("--top-percent", type=float, default=DEFAULT_PARAMS["top_percent"])
    parser.add_argument("--weighting", choices=["uniform", "frequency", "idf"], default=DEFAULT_PARAMS["weighting"])
    parser.add_argument("--spreading-steps", type=int, default=DEFAULT_PARAMS["spreading_steps"])
    parser.add_argument("--no-morton", action="store_true")
    parser.add_argument("--lancedb", action="store_true",
                        help="Build a LanceDB ANN doc index (after Step 5) and use it for scoring")
    parser.add_argument("--lancedb-path", type=Path, default=None,
                        help="LanceDB database dir (default: <run_dir>/lancedb)")
    parser.add_argument("--lancedb-exact", action="store_true", default=False,
                        help="Use exact scan inside LanceDB instead of ANN index")
    parser.add_argument("--lancedb-limit", type=int, default=200,
                        help="Max ANN candidates returned per query")
    parser.add_argument("--splade-only", action="store_true",
                        help="Only run SPLADE variants + BM25 (skip pure_sf)")
    parser.add_argument("--pure-sf-only", action="store_true",
                        help="Only run the pure_sf variant + BM25 (skip SPLADE)")
    parser.add_argument("--reuse-run", type=Path, default=None,
                        help="Reuse an existing run dir (skip Steps 1-5 + LanceDB build)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of concurrent Step 7 subprocesses (shards the query file)")
    parser.add_argument("--step6-only", action="store_true",
                        help="Only re-run Step 6 (requires --reuse-run)")
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    params.update({
        "grid_size": args.grid_size,
        "method": args.method,
        "top_percent": args.top_percent,
        "weighting": args.weighting,
        "spreading_steps": args.spreading_steps,
        "top_k": args.top_k,
        "morton": not args.no_morton,
        "retrieval_backend": "lancedb" if args.lancedb else "numpy",
        "lancedb_path": str(args.lancedb_path) if args.lancedb_path else None,
        "lancedb_exact": args.lancedb_exact,
        "lancedb_limit": args.lancedb_limit,
    })

    # ── Load inputs ──────────────────────────────────────────────────────
    docs = load_corpus(args.corpus)
    query_rows = load_queries(args.queries)
    gold_map = load_gold(args.gold)

    n_docs = len(docs)
    doc_ids = [d["doc_id"] for d in docs]
    corpus_texts = [d["text"] for d in docs]
    corpus_id_to_row = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # ── Build mixed queries (SF phrase extraction) ───────────────────────
    logger.info("Generating mixed-language query strings...")
    queries = []
    for row in query_rows:
        q = make_mixed_query(row.get("question", ""))
        queries.append(q)

    n_queries = len(queries)
    logger.info(f"Built {n_queries} query strings")

    # ── Build gold + candidate maps (candidates = all docs) ──────────────
    query_gold = {}
    query_doc_map = {}
    n_skipped = 0
    for idx, row in enumerate(query_rows):
        qid = row.get("id", str(idx))
        gold_ids = gold_map.get(qid, [])
        if not gold_ids:
            logger.warning(f"[Q{idx}] id '{qid}' has no gold entries; skipping")
            n_skipped += 1
            continue
        mapped_gold = []
        for gid in gold_ids:
            if gid in corpus_id_to_row:
                mapped_gold.append(gid)
            else:
                logger.warning(f"[Q{idx}] gold doc '{gid}' not in corpus; dropped")
        if not mapped_gold:
            logger.warning(f"[Q{idx}] no valid gold docs remain; skipping")
            n_skipped += 1
            continue
        query_gold[str(idx)] = mapped_gold
        query_doc_map[str(idx)] = list(doc_ids)

    if not query_gold:
        logger.error("No queries with gold found. Aborting.")
        sys.exit(1)

    active_indices = sorted(int(k) for k in query_gold.keys())
    logger.info(f"{len(active_indices)} queries have gold ({n_skipped} skipped)")

    # ── Create run directory ─────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.reuse_run is not None:
        run_dir = Path(args.reuse_run)
        if not run_dir.is_dir():
            logger.error(f"Reuse run dir not found: {run_dir}")
            sys.exit(1)
        logger.info(f"Reusing run dir: {run_dir}")
        corpus_path = run_dir / "corpus.txt"
        query_file = run_dir / "queries.txt"
        with open(run_dir / "query_doc_map.json", encoding="utf-8") as f:
            query_doc_map = json.load(f)
        with open(run_dir / "query_gold.json", encoding="utf-8") as f:
            query_gold = json.load(f)
        # Rebuild docs/queries from the run dir so later code is identical
        query_rows = None

        # Load docs from the copied corpus.txt
        docs = load_corpus(corpus_path)
        n_docs = len(docs)
        doc_ids = [d["doc_id"] for d in docs]
        corpus_texts = [d["text"] for d in docs]
        corpus_id_to_row = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        with open(query_file, encoding="utf-8") as f:
            queries = [ln.rstrip("\n") for ln in f if ln.strip()]
        n_queries = len(queries)

        active_indices = sorted(int(k) for k in query_gold.keys())
        n_skipped = 0

        # LanceDB: reuse the index already built in the run dir
        if params.get("retrieval_backend") == "lancedb":
            idx_path = run_dir / "lancedb"
            if not idx_path.is_dir():
                logger.error(f"Lancedb index not found in reuse dir: {idx_path}. "
                             f"Run without --reuse-run to build it.")
                sys.exit(1)
            params["lancedb_path"] = str(idx_path)

        have_indexed = True  # skip Steps 1-5
    else:
        run_dir = _OUTPUTS / "runs" / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run dir: {run_dir}")

        corpus_path = run_dir / "corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc_id, text in zip(doc_ids, corpus_texts):
                f.write(f"{doc_id}, passage {text}\n")
        logger.info(f"Corpus: {n_docs} passages -> {corpus_path}")

        query_file = run_dir / "queries.txt"
        with open(query_file, "w", encoding="utf-8") as f:
            for idx in range(n_queries):
                f.write(queries[idx] + "\n")

        with open(run_dir / "query_doc_map.json", "w", encoding="utf-8") as f:
            json.dump(query_doc_map, f, indent=2)
        with open(run_dir / "query_gold.json", "w", encoding="utf-8") as f:
            json.dump(query_gold, f, indent=2)

        have_indexed = False

    # ── Phase 1: Steps 1-5 (skipped when --reuse-run) ───────────────────────
    step_ok = True
    if not have_indexed:
        out = run_dir / "extracted_phrases"
        step_ok &= run_step(STEP_SCRIPTS[1], [
            "--corpus", str(corpus_path), "--output", str(out),
            "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
            "--min-freq", str(params["min_freq"]),
        ], "Step 1 phrase_extractor", timeout=600)
        if not step_ok:
            logger.error("Step 1 failed"); return

        out = run_dir / "term_context_matrix"
        step_ok &= run_step(STEP_SCRIPTS[2], [
            "--vocab", str(run_dir / "extracted_phrases" / "vocabulary.csv"),
            "--mapping", str(run_dir / "extracted_phrases" / "phrase_to_contexts.json"),
            "--corpus", str(corpus_path), "--output", str(out),
        ], "Step 2 term_context", timeout=600)
        if not step_ok:
            logger.error("Step 2 failed"); return

        out = run_dir / "semantic_space"
        step_ok &= run_step(STEP_SCRIPTS[3], [
            "--matrix", str(run_dir / "term_context_matrix" / "term_context_matrix.npz"),
            "--metadata", str(run_dir / "term_context_matrix" / "term_context_matrix.json"),
            "--output", str(out),
            "--grid-size", str(params["grid_size"]),
            "--method", params["method"],
            "--random-seed", str(params["random_seed"]),
            "--n-neighbors", str(params["umap_n_neighbors"]),
            "--min-dist", str(params["umap_min_dist"]),
            "--metric", params["umap_metric"],
        ], "Step 3 semantic_space", timeout=900)
        if not step_ok:
            logger.error("Step 3 failed"); return

        morton_flag = "--morton" if params["morton"] else "--no-morton"
        out = run_dir / "phrase_fingerprints"
        step_ok &= run_step(STEP_SCRIPTS[4], [
            "--coordinates", str(run_dir / "semantic_space" / "context_coordinates.json"),
            "--metadata", str(run_dir / "term_context_matrix" / "term_context_matrix.json"),
            "--output", str(out),
            "--grid-size", str(params["grid_size"]),
            "--smoothing-sigma", str(params["smoothing_sigma"]),
            morton_flag,
        ], "Step 4 phrase_fingerprints", timeout=600)
        if not step_ok:
            logger.error("Step 4 failed"); return

        out = run_dir / "doc_fingerprints"
        step_ok &= run_step(STEP_SCRIPTS[5], [
            "--corpus", str(corpus_path),
            "--fingerprints", str(run_dir / "phrase_fingerprints"),
            "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
            "--output", str(out),
            "--grid-size", str(params["grid_size"]),
            "--top-percent", str(params["top_percent"]),
            "--normalize-method", "l2",
            "--min-word-length", str(params["min_word_length"]),
            "--smoothing-sigma", str(params["smoothing_sigma"]),
            "--min-peak-distance", "2",
            morton_flag,
        ], "Step 5 doc_fingerprints", timeout=600)
        if not step_ok:
            logger.error("Step 5 failed"); return

        # ── Retrieval backend index (LanceDB) ──────────────────────────────
        if params["retrieval_backend"] == "lancedb":
            try:
                from lance_storage import LanceStorage
                lancedb_path = params.get("lancedb_path") or str(run_dir / "lancedb")
                params["lancedb_path"] = lancedb_path
                logger.info(f"  [BACKEND] building LanceDB ANN index at {lancedb_path}")
                storage = LanceStorage(lancedb_path)
                idx_info = storage.build_document_index(str(out))
                idx_info["backend"] = "lancedb"
                idx_info["path"] = lancedb_path
                with open(run_dir / "retrieval_backend.json", "w", encoding="utf-8") as f:
                    json.dump(idx_info, f, indent=2)
                logger.info(f"  [BACKEND] LanceDB ANN index built "
                            f"({idx_info['num_docs']} docs, {idx_info['build_seconds']}s)")
            except Exception as exc:
                logger.error(f"  [BACKEND] LanceDB index build FAILED: {exc}")
                return

    # ── Phase 2: Run variants ────────────────────────────────────────────
    bench_base = _OUTPUTS / "benchmarks" / f"benchmark_{ts}"
    bench_base.mkdir(parents=True, exist_ok=True)

    variants = build_variants(query_file, run_dir, corpus_path, params)

    all_variant_metrics = {}
    order = ["pure_sf", "splade_linear", "splade_rrf"]
    if args.pure_sf_only:
        order = ["pure_sf"]
    elif args.splade_only:
        order = ["splade_linear", "splade_rrf"]

    for vname in order:
        vcfg = variants[vname]
        vdir = bench_base / vname
        vdir.mkdir(parents=True, exist_ok=True)
        output_json = vdir / "all_results.json"

        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {vcfg['display']} ({vname})")
        logger.info(f"{'='*60}")

        t0 = time.time()
        ok = run_step6_parallel(vcfg["step7_args"], output_json,
                                f"Step 6 {vname}", num_workers=args.parallel,
                                timeout=3600)
        elapsed = time.time() - t0

        if not ok:
            logger.error(f"  [{vname}] FAILED after {elapsed:.0f}s")
            all_variant_metrics[vname] = {"error": "failed", "elapsed_s": elapsed}
            continue

        with open(output_json, encoding="utf-8") as f:
            all_results = json.load(f)

        var_metrics = []
        for i in active_indices:
            scores_list = all_results[i] if i < len(all_results) else None
            if scores_list is None:
                continue
            raw = scores_list.get("results", []) if isinstance(scores_list, dict) else scores_list
            gold_ids = query_gold.get(str(i), [])
            cand_ids = query_doc_map.get(str(i), [])
            cand_set = set(cand_ids)
            filtered = [(d, s) for d, s in raw if d in cand_set][:params["top_k"]]
            var_metrics.append(compute_metrics(filtered, gold_ids))

        summary = aggregate(var_metrics)
        summary["elapsed_s"] = round(elapsed, 1)
        summary["display"] = vcfg["display"]

        with open(vdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        all_variant_metrics[vname] = summary
        logger.info(f"  [{vname}] MRR={summary['mean_mrr']:.4f} AP={summary['mean_ap']:.4f} ({elapsed:.0f}s)")

    # ── BM25 ─────────────────────────────────────────────────────────────
    if not args.step6_only:
        logger.info(f"\n{'='*60}")
        logger.info("Running BM25 baseline")
        logger.info(f"{'='*60}")
        t0 = time.time()
        gold_sets = [query_gold[str(i)] for i in active_indices]
        candidate_sets = [query_doc_map[str(i)] for i in active_indices]
        active_queries = [queries[i] for i in active_indices]
        bm25_metrics = run_bm25(corpus_texts, active_queries, gold_sets, candidate_sets,
                                top_k=params["top_k"])
        bm25_elapsed = time.time() - t0

        bm25_summary = aggregate(bm25_metrics)
        bm25_summary["elapsed_s"] = round(bm25_elapsed, 1)
        bm25_summary["display"] = "BM25"

        bm25_dir = bench_base / "bm25"
        bm25_dir.mkdir(exist_ok=True)
        with open(bm25_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(bm25_summary, f, indent=2)

        all_variant_metrics["bm25"] = bm25_summary
        logger.info(f"  [BM25] MRR={bm25_summary['mean_mrr']:.4f} AP={bm25_summary['mean_ap']:.4f} ({bm25_elapsed:.0f}s)")

    # ── Comparison report ────────────────────────────────────────────────
    report_path = bench_base / "comparison_report.md"
    lines = [
        "# Custom Arabic-English Benchmark Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Corpus:** {n_docs} bilingual passages (Arabic | English) — {args.corpus.name}\n",
        f"**Queries:** {len(active_indices)} with gold (of {n_queries}) — {args.queries.name}\n",
        f"**Gold:** 1 relevant doc per query (from {args.gold.name})\n",
        f"**Candidates:** All {n_docs} passages\n",
        "---\n",
        "## Results\n",
        "| Variant | MRR | AP | P@1 | P@5 | R@5 | NDCG@20 | Found@ | Time |",
        "|---------|-----|-----|-----|-----|-----|---------|--------|------|",
    ]

    display_names = {
        "pure_sf": "**Pure SF**",
        "splade_linear": "SF+SPLADE Linear (\u03b1=0.3)",
        "splade_rrf": "SF+SPLADE RRF (k=60)",
        "bm25": "**BM25**",
    }

    report_order = order + (["bm25"] if not args.step6_only else [])

    for vname in report_order:
        s = all_variant_metrics.get(vname, {})
        if "error" in s:
            lines.append(f"| **{vname}** | ERROR | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 |")
        else:
            lines.append(
                f"| {s.get('display', vname)} "
                f"| {s.get('mean_mrr', 0):.4f} "
                f"| {s.get('mean_ap', 0):.4f} "
                f"| {s.get('mean_p@1', 0):.4f} "
                f"| {s.get('mean_p@5', 0):.4f} "
                f"| {s.get('mean_r@5', 0):.4f} "
                f"| {s.get('mean_ndcg@20', 0):.4f} "
                f"| {s.get('mean_found_at', 0):.1f} "
                f"| {s.get('elapsed_s', 0):.0f}s |"
            )

    lines.append(f"\n### Detailed Comparison\n")
    header = "| Metric | " + " | ".join(display_names[v] if v in display_names else v for v in report_order) + " |"
    sep = "|--------|" + "--------|" * len(report_order)
    lines.append(header)
    lines.append(sep)

    metrics_to_show = ["mrr", "ap", "p@1", "p@5", "r@5", "ndcg@20"]
    for mk in metrics_to_show:
        vals = []
        for vname in report_order:
            s = all_variant_metrics.get(vname, {})
            vals.append(f"{s.get(f'mean_{mk}', 0):.4f}" if "error" not in s else "ERR")
        lines.append(f"| **{mk}** | {' | '.join(vals)} |")

    lines.append("\n---\n")
    lines.append(f"### Config\n")
    lines.append(f"```yaml\n{json.dumps(params, indent=2)}\n```")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"\nReport: {report_path}")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON \u2014 Custom Arabic-English Benchmark")
    print("=" * 70)
    print(f"{'Variant':<30} {'MRR':>8} {'AP':>8} {'P@1':>8} {'P@5':>8} {'R@5':>8} {'NDCG@20':>10}")
    print("-" * 70)
    for vname in report_order:
        s = all_variant_metrics.get(vname, {})
        dn = display_names.get(vname, vname)
        dn_ascii = dn.replace("\u03b1", "alpha")
        if "error" in s:
            print(f"{dn_ascii:<30} {'ERROR':>8}")
        else:
            print(f"{dn_ascii:<30} {s.get('mean_mrr', 0):>8.4f} {s.get('mean_ap', 0):>8.4f} "
                  f"{s.get('mean_p@1', 0):>8.4f} {s.get('mean_p@5', 0):>8.4f} "
                  f"{s.get('mean_r@5', 0):>8.4f} {s.get('mean_ndcg@20', 0):>10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
