"""
Benchmark: Mixed Arabic-English queries against bilingual corpus.

Uses SF phrase extraction to build mixed-language queries from Belebele
(Arabic question + English question), then runs retrieval against the
bilingual passage corpus.

Variants: Pure SF, SF+SPLADE Linear, SF+SPLADE RRF, BM25
"""

import csv, json, os, re, subprocess, sys, time, yaml
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

logger = get_logger("mixed_ar_en")

# ── spaCy setup ─────────────────────────────────────────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    nlp = None

# ── Constants ───────────────────────────────────────────────────────────────
CSV_PATH = PROJECT_ROOT / "data/datasets/belebele/raw/all/corpus_belebele_ar_en_deduped.csv"
_OUTPUTS = PROJECT_ROOT / "outputs" / "mixed_ar_en_benchmark"
_SEP = " | "

STEP_SCRIPTS = {
    1: PROJECT_ROOT / "semantic_folding" / "phrase_extractor.py",
    2: PROJECT_ROOT / "semantic_folding" / "term_context.py",
    3: PROJECT_ROOT / "semantic_folding" / "semantic_space.py",
    4: PROJECT_ROOT / "semantic_folding" / "phrase_fingerprints.py",
    5: PROJECT_ROOT / "semantic_folding" / "doc_fingerprints.py",
    6: PROJECT_ROOT / "semantic_folding" / "query_processor.py",
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
# 2. Pipeline helpers
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
# 3. Metrics
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

    # AP
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

        # NDCG@k
        dcg = sum(1.0 / np.log2(rank + 1) for rank, d in enumerate(top_k, 1) if d in rel_set)
        ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(rel_set), k) + 1))
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal > 0 else 0.0

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# 4. BM25
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
        logger.info(f"  BM25 [{i:04d}] MRR={metrics['mrr']:.4f} AP={metrics['ap']:.4f}")

    return all_metrics


# ═════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # Read CSV
    logger.info(f"Reading deduped CSV: {CSV_PATH}")
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info(f"Loaded {len(rows)} rows")

    # Generate mixed queries
    logger.info("Generating mixed-language queries with SF extraction...")
    queries = []
    for i, row in enumerate(rows):
        q = make_mixed_query(row["question_1"])
        queries.append(q)
        if (i + 1) % 100 == 0:
            logger.info(f"  Generated {i+1}/{len(rows)} queries")

    # Build corpus (bilingual passages), query_doc_map, query_gold
    corpus_texts = [row["passage"] for row in rows]
    n_docs = len(corpus_texts)
    doc_ids = [f"doc_{i:06d}" for i in range(n_docs)]

    # Each query: ALL docs are candidates, only its own doc is gold
    query_doc_map = {str(i): doc_ids for i in range(n_docs)}
    query_gold = {str(i): [f"doc_{i:06d}"] for i in range(n_docs)}

    # Create run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _OUTPUTS / "runs" / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run dir: {run_dir}")

    # Save corpus
    corpus_path = run_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, (did, text) in enumerate(zip(doc_ids, corpus_texts)):
            f.write(f"{did}, passage {text}\n")
    logger.info(f"Corpus: {n_docs} passages -> {corpus_path}")

    # Save query file
    query_file = run_dir / "queries.txt"
    with open(query_file, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(q + "\n")

    # Save mapping files
    with open(run_dir / "query_doc_map.json", "w", encoding="utf-8") as f:
        json.dump(query_doc_map, f, indent=2)
    with open(run_dir / "query_gold.json", "w", encoding="utf-8") as f:
        json.dump(query_gold, f, indent=2)

    # ── Phase 1: Steps 1-5 ──────────────────────────────────────────────
    params = {
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

    step_ok = True

    # Step 1
    out = run_dir / "extracted_phrases"
    step_ok &= run_step(STEP_SCRIPTS[1], [
        "--corpus", str(corpus_path), "--output", str(out),
        "--keep-verbs", "--min-word-length", str(params["min_word_length"]),
        "--min-freq", str(params["min_freq"]),
    ], "Step 1 phrase_extractor", timeout=600)
    if not step_ok:
        logger.error("Step 1 failed")
        return

    # Step 2
    out = run_dir / "term_context_matrix"
    step_ok &= run_step(STEP_SCRIPTS[2], [
        "--vocab", str(run_dir / "extracted_phrases" / "vocabulary.csv"),
        "--mapping", str(run_dir / "extracted_phrases" / "phrase_to_contexts.json"),
        "--corpus", str(corpus_path), "--output", str(out),
    ], "Step 2 term_context", timeout=600)
    if not step_ok:
        logger.error("Step 2 failed")
        return

    # Step 3
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
        logger.error("Step 3 failed")
        return

    # Step 4
    out = run_dir / "phrase_fingerprints"
    morton_flag = "--morton" if params["morton"] else "--no-morton"
    step_ok &= run_step(STEP_SCRIPTS[4], [
        "--coordinates", str(run_dir / "semantic_space" / "context_coordinates.json"),
        "--metadata", str(run_dir / "term_context_matrix" / "term_context_matrix.json"),
        "--output", str(out),
        "--grid-size", str(params["grid_size"]),
        "--smoothing-sigma", str(params["smoothing_sigma"]),
        morton_flag,
    ], "Step 4 phrase_fingerprints", timeout=600)
    if not step_ok:
        logger.error("Step 4 failed")
        return

    # Step 5
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
        logger.error("Step 5 failed")
        return

    # ── Phase 2: Run variants ────────────────────────────────────────────
    bench_base = _OUTPUTS / "benchmarks" / f"benchmark_{ts}"
    bench_base.mkdir(parents=True, exist_ok=True)

    variants = {
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
            ],
        },
        "splade_linear": {
            "display": "SF+SPLADE Linear (α=0.3)",
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
            ],
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
            ],
        },
    }

    all_variant_metrics = {}

    for vname, vcfg in variants.items():
        vdir = bench_base / vname
        vdir.mkdir(parents=True, exist_ok=True)
        output_json = vdir / "all_results.json"

        if vname == "pure_sf":
            args = vcfg["step7_args"] + ["--output", str(output_json)]
        else:
            args = vcfg["step7_args"] + ["--output", str(output_json)]

        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {vcfg['display']} ({vname})")
        logger.info(f"{'='*60}")

        t0 = time.time()
        ok = run_step(STEP_SCRIPTS[6], args, f"Step 6 {vname}", timeout=3600)
        elapsed = time.time() - t0

        if not ok:
            logger.error(f"  [{vname}] FAILED after {elapsed:.0f}s")
            all_variant_metrics[vname] = {"error": "failed", "elapsed_s": elapsed}
            continue

        # Read results
        with open(output_json, encoding="utf-8") as f:
            all_results = json.load(f)

        # Compute per-query metrics
        var_metrics = []
        for i, (q_text, scores_list) in enumerate(zip(queries, all_results)):
            raw = scores_list.get("results", []) if isinstance(scores_list, dict) else scores_list
            gold_ids = query_gold.get(str(i), [])
            cand_ids = query_doc_map.get(str(i), [])
            cand_set = set(cand_ids)
            filtered = [(d, s) for d, s in raw if d in cand_set][:params["top_k"]]
            m = compute_metrics(filtered, gold_ids)
            var_metrics.append(m)

        # Aggregate
        agg = defaultdict(list)
        for m in var_metrics:
            for k, v in m.items():
                agg[k].append(v)

        summary = {"num_queries": len(var_metrics), "elapsed_s": round(elapsed, 1)}
        for k, vals in agg.items():
            summary[f"mean_{k}"] = sum(vals) / len(vals)
            summary[f"min_{k}"] = min(vals)
            summary[f"max_{k}"] = max(vals)

        # Save per-variant summary
        with open(vdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        all_variant_metrics[vname] = summary
        logger.info(f"  [{vname}] MRR={summary['mean_mrr']:.4f} AP={summary['mean_ap']:.4f} ({elapsed:.0f}s)")

    # ── BM25 ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Running BM25 baseline")
    logger.info(f"{'='*60}")
    t0 = time.time()
    gold_sets = [query_gold[str(i)] for i in range(n_docs)]
    candidate_sets = [query_doc_map[str(i)] for i in range(n_docs)]
    bm25_metrics = run_bm25(corpus_texts, queries, gold_sets, candidate_sets, top_k=params["top_k"])
    bm25_elapsed = time.time() - t0

    bm25_agg = defaultdict(list)
    for m in bm25_metrics:
        for k, v in m.items():
            bm25_agg[k].append(v)

    bm25_summary = {"num_queries": len(bm25_metrics), "elapsed_s": round(bm25_elapsed, 1)}
    for k, vals in bm25_agg.items():
        bm25_summary[f"mean_{k}"] = sum(vals) / len(vals)
        bm25_summary[f"min_{k}"] = min(vals)
        bm25_summary[f"max_{k}"] = max(vals)

    bm25_dir = bench_base / "bm25"
    bm25_dir.mkdir(exist_ok=True)
    with open(bm25_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(bm25_summary, f, indent=2)

    all_variant_metrics["bm25"] = bm25_summary
    logger.info(f"  [BM25] MRR={bm25_summary['mean_mrr']:.4f} AP={bm25_summary['mean_ap']:.4f} ({bm25_elapsed:.0f}s)")

    # ── Comparison report ───────────────────────────────────────────────
    report_path = bench_base / "comparison_report.md"
    lines = [
        "# Mixed Arabic-English Benchmark Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Corpus:** {n_docs} bilingual passages (Arabic | English)\n",
        f"**Queries:** {len(queries)} mixed-language (SF-extracted Arabic + English phrases)\n",
        f"**Gold:** 1 relevant doc per query (exact passage)\n",
        f"**Candidates:** All {n_docs} passages\n",
        "---\n",
        "## Results\n",
        "| Variant | MRR | AP | P@1 | P@5 | R@5 | NDCG@20 | Found@ | Time |",
        "|---------|-----|-----|-----|-----|-----|---------|--------|------|",
    ]

    for vname in ["pure_sf", "splade_linear", "splade_rrf", "bm25"]:
        s = all_variant_metrics.get(vname, {})
        if "error" in s:
            lines.append(f"| **{vname}** | ERROR | — | — | — | — | — | — | — |")
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

    # Display names
    display_names = {
        "pure_sf": "**Pure SF**",
        "splade_linear": "SF+SPLADE Linear (α=0.3)",
        "splade_rrf": "SF+SPLADE RRF (k=60)",
        "bm25": "**BM25**",
    }

    lines.append(f"\n### Detailed Comparison\n")
    header = "| Metric | Pure SF | SF+SPLADE Linear | SF+SPLADE RRF | BM25 |"
    sep = "|--------|---------|-------------------|----------------|------|"
    lines.append(header)
    lines.append(sep)

    metrics_to_show = ["mrr", "ap", "p@1", "p@5", "r@5", "ndcg@20"]
    for mk in metrics_to_show:
        vals = []
        for vname in ["pure_sf", "splade_linear", "splade_rrf", "bm25"]:
            s = all_variant_metrics.get(vname, {})
            vals.append(f"{s.get(f'mean_{mk}', 0):.4f}" if "error" not in s else "ERR")
        lines.append(f"| **{mk}** | {' | '.join(vals)} |")

    lines.append("\n---\n")
    lines.append(f"### Config\n")
    lines.append(f"```yaml\n{json.dumps(params, indent=2)}\n```")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"\nReport: {report_path}")

    # ── Print final comparison ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON — Mixed Arabic-English Benchmark")
    print("=" * 70)
    print(f"{'Variant':<30} {'MRR':>8} {'AP':>8} {'P@1':>8} {'P@5':>8} {'R@5':>8} {'NDCG@20':>10}")
    print("-" * 70)
    for vname in ["pure_sf", "splade_linear", "splade_rrf", "bm25"]:
        s = all_variant_metrics.get(vname, {})
        dn = display_names.get(vname, vname)
        if "error" in s:
            print(f"{dn:<30} {'ERROR':>8}")
        else:
            print(f"{dn:<30} {s.get('mean_mrr', 0):>8.4f} {s.get('mean_ap', 0):>8.4f} "
                  f"{s.get('mean_p@1', 0):>8.4f} {s.get('mean_p@5', 0):>8.4f} "
                  f"{s.get('mean_r@5', 0):>8.4f} {s.get('mean_ndcg@20', 0):>10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
