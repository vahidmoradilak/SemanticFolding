#!/usr/bin/env python3
"""
Fragment Retrieval Test: SF vs Dense vs BM25 on Quran ayah fragments.

Flow:
  1. Select 100 random ayahs from corpus
  2. Generate fragments (4 types × 4 lengths = 16 per ayah)
  3. Build SF fingerprints for all fragments (via customtext_fingerprints)
  4. Compute Dense embeddings for fragments
  5. Evaluate all 3 methods on each fragment
  6. Report aggregate results by type and length
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import get_logger
logger = get_logger("fragment_test")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = PROJECT_ROOT / "data" / "quran" / "quran_ayahs_clean.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fragment_test"
RUN_DIR = PROJECT_ROOT / "outputs" / "quran_benchmark" / "runs" / "run_20260710_193034"
QA_PATH = PROJECT_ROOT / "data" / "quran" / "quran_qa_semantic_gold_only.jsonl"

FRAGMENT_TYPES = ["arabic_prefix", "arabic_mid", "english_prefix", "mixed_prefix"]
FRAGMENT_LENGTHS = [10, 30, 50, 100]
N_SUBSET = 100
SEED = 42

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path):
    arabic, english = [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            arabic.append(parts[1])
            english.append(parts[2])
    return arabic, english


def load_qa(path):
    pairs = []
    with open(path, encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            pairs.append({
                "id": data["id"],
                "query": data["question"],
                "relevant": data["relevant"],
            })
    return pairs


# ---------------------------------------------------------------------------
# Fragment generation
# ---------------------------------------------------------------------------

def generate_fragments(arabic_texts, english_texts, indices,
                       types=None, lengths=None):
    if types is None:
        types = FRAGMENT_TYPES
    if lengths is None:
        lengths = FRAGMENT_LENGTHS

    fragments = []  # list of (fragment_id, fragment_text, ayah_idx, type_name, length)
    fragment_id = 0
    for ayah_idx in indices:
        ar = arabic_texts[ayah_idx]
        en = english_texts[ayah_idx]
        for ftype in types:
            for flen in lengths:
                if flen <= 0:
                    continue
                if ftype == "arabic_prefix":
                    text = ar[:flen]
                elif ftype == "arabic_mid":
                    start = len(ar) // 3
                    text = ar[start:start + flen]
                elif ftype == "english_prefix":
                    text = en[:flen]
                elif ftype == "mixed_prefix":
                    mixed = f"{ar[:50]}  |  {en[:50]}"
                    text = mixed[:flen]
                else:
                    continue

                # Guard: if fragment is too short, skip
                if len(text) < 3:
                    # But still create a fragment with what we have
                    pass

                fid = f"F_{ayah_idx:04d}_{ftype}_{flen}"
                fragments.append((fid, text, ayah_idx, ftype, flen))
                fragment_id += 1
    return fragments


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(retrieved_ids, relevant_ids, top_k_list=None):
    """Standard IR metrics for a single query."""
    if top_k_list is None:
        top_k_list = [1, 5, 10]
    rel_set = set(relevant_ids)
    n_rel = len(rel_set)

    mrr = 0.0
    ap = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            mrr = 1.0 / rank
            break

    hits = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            hits += 1
            ap += hits / rank
    ap /= n_rel if n_rel else 1

    metrics = {"MRR": mrr, "AP": ap, "found_at": mrr}
    for k in top_k_list:
        retrieved_k = retrieved_ids[:k]
        rel_k = sum(1 for d in retrieved_k if d in rel_set)
        metrics[f"P@{k}"] = rel_k / k
        metrics[f"R@{k}"] = rel_k / n_rel if n_rel else 0.0
    return metrics, mrr > 0


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def tokenize(text):
    """Simple tokenizer: Arabic char + ASCII words."""
    import re
    # Arabic chars + ASCII letters
    tokens = re.findall(r"[\u0600-\u06FF\u0750-\u077F\w]+", text.lower())
    return tokens


def bm25_retrieve(query, corpus_lines, doc_ids, corpus_tokens=None,
                  k1=1.5, b=0.75):
    """BM25 retrieval with Arabic Unicode support."""
    import math
    from collections import Counter

    N = len(corpus_lines)
    if corpus_tokens is None:
        corpus_tokens = [tokenize(line) for line in corpus_lines]

    avgdl = sum(len(toks) for toks in corpus_tokens) / N if N else 1
    q_tokens = tokenize(query)

    # IDF per query token
    idf_cache = {}
    for qt in q_tokens:
        n_q = sum(1 for toks in corpus_tokens if qt in toks)
        idf_cache[qt] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

    scores = []
    for i, toks in enumerate(corpus_tokens):
        doc_len = len(toks)
        tf = Counter(toks)
        score = 0.0
        for qt in q_tokens:
            if qt not in idf_cache:
                continue
            tf_val = tf.get(qt, 0)
            if tf_val == 0:
                continue
            score += (idf_cache[qt]
                      * (tf_val * (k1 + 1))
                      / (tf_val + k1 * (1 - b + b * doc_len / avgdl)))
        scores.append((doc_ids[i], score))

    scores.sort(key=lambda x: -x[1])
    return scores


# ---------------------------------------------------------------------------
# SF Fragment fingerprinting (runs customtext_fingerprints.py)
# ---------------------------------------------------------------------------

def build_sf_fragment_fingerprints(fragments_csv, output_dir):
    """Run customtext_fingerprints.py to generate SF fingerprints for fragments."""
    phrase_fp_dir = RUN_DIR / "phrase_fingerprints"
    idf_path = RUN_DIR / "term_context_matrix" / "idf_weights.json"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "semantic_folding" / "customtext_fingerprints.py"),
        "--corpus", str(fragments_csv),
        "--fingerprints", str(phrase_fp_dir),
        "--output", str(output_dir),
        "--grid-size", "64",
        "--top-percent", "0.05",
        "--normalize-method", "l2",
        "--morton",
        "--min-word-length", "2",
        "--keep-verbs",
        "--smoothing-sigma", "1.5",
    ]
    if idf_path.exists():
        cmd += ["--idf-weights", str(idf_path)]

    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf8", errors="replace",
    )
    elapsed = time.time() - t0
    logger.info(f"customtext_fingerprints finished in {elapsed:.1f}s")

    # Load the output fingerprints
    fp_path = output_dir / "doc_fingerprints.npz"
    meta_path = output_dir / "doc_fingerprints_meta.json"
    if not fp_path.exists():
        logger.error(f"SF fingerprints not found at {fp_path}")
        logger.error(result.stdout[-2000:])
        logger.error(result.stderr[-2000:])
        return None, None

    fp_matrix = np.load(str(fp_path))["fingerprints"].astype(np.float32)
    with open(meta_path, encoding="utf8") as f:
        meta = json.load(f)
    # meta maps fragment_id -> row_index
    return fp_matrix, meta


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def load_dense_model(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading dense model: {model_name}")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    logger.info(f"Loaded in {time.time()-t0:.1f}s")
    return model


def encode_and_rank_dense(fragments, doc_embeddings, doc_ids, model,
                          text_field="arabic", batch_size=64):
    """Encode fragments and rank against doc embeddings."""
    texts = [text for _, text, _, _, _ in fragments]
    frag_embeddings = model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # Compute similarities: (n_frags, n_docs)
    sims = np.dot(frag_embeddings, doc_embeddings.T)

    results = []
    for i, (fid, text, ayah_idx, ftype, flen) in enumerate(fragments):
        ranked = [(doc_ids[j], float(sims[i, j]))
                  for j in np.argsort(-sims[i])]
        results.append((fid, text, ayah_idx, ftype, flen, ranked))
    return results


# ---------------------------------------------------------------------------
# SF ranking
# ---------------------------------------------------------------------------

def sf_rank_fragments(fragments, sf_fp_matrix, sf_meta,
                      doc_fp_matrix, doc_ids):
    """Rank fragments against doc fingerprints using cosine similarity."""
    results = []
    for fid, text, ayah_idx, ftype, flen in fragments:
        if fid not in sf_meta:
            # Fragment couldn't be fingerprinted
            results.append((fid, text, ayah_idx, ftype, flen, []))
            continue
        row = sf_meta[fid]
        frag_fp = sf_fp_matrix[row:row+1]
        # Cosine similarity
        norms = np.linalg.norm(frag_fp, axis=1) * np.linalg.norm(doc_fp_matrix, axis=1)
        sims = np.dot(frag_fp, doc_fp_matrix.T).flatten() / (norms + 1e-10)
        ranked = [(doc_ids[j], float(sims[j]))
                  for j in np.argsort(-sims)]
        results.append((fid, text, ayah_idx, ftype, flen, ranked))
    return results


# ---------------------------------------------------------------------------
# BM25 ranking
# ---------------------------------------------------------------------------

def bm25_rank_fragments(fragments, corpus_lines, doc_ids):
    """BM25 for all fragments."""
    # Pre-tokenize corpus
    corpus_tokens = [tokenize(line) for line in corpus_lines]
    results = []
    for fid, text, ayah_idx, ftype, flen in fragments:
        retrieved = bm25_retrieve(text, corpus_lines, doc_ids,
                                  corpus_tokens=corpus_tokens)
        ranked = [(doc_id, score) for doc_id, score in retrieved]
        results.append((fid, text, ayah_idx, ftype, flen, ranked))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fragment Retrieval Test")
    parser.add_argument("--n-subset", type=int, default=N_SUBSET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--skip-sf", action="store_true",
                        help="Skip SF fingerprint building (reuse cached)")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip dense baseline")
    parser.add_argument("--skip-bm25", action="store_true",
                        help="Skip BM25 baseline")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir

    # Load corpus
    logger.info(f"Loading corpus: {CORPUS_PATH}")
    arabic_texts, english_texts = load_corpus(CORPUS_PATH)
    n_docs = len(arabic_texts)
    logger.info(f"Loaded {n_docs} ayahs")

    # Select subset
    random.seed(args.seed)
    selected = sorted(random.sample(range(n_docs), args.n_subset))
    logger.info(f"Selected {len(selected)} ayahs: {selected[:5]}...{selected[-5:]}")

    # Generate fragments
    fragments = generate_fragments(arabic_texts, english_texts, selected)
    logger.info(f"Generated {len(fragments)} fragments "
                f"({len(FRAGMENT_TYPES)} types × {len(FRAGMENT_LENGTHS)} lengths × {args.n_subset} ayahs)")

    # Write fragments CSV for SF pipeline
    frag_csv = output_dir / "fragments.csv"
    with open(frag_csv, "w", encoding="utf8") as f:
        for fid, text, ayah_idx, ftype, flen in fragments:
            # Escape: replace newlines in text (shouldn't happen, but safe)
            text_clean = text.replace("\n", " ").replace("\r", " ")
            f.write(f"{fid},{text_clean}\n")
    logger.info(f"Wrote fragments CSV: {frag_csv}")

    # Load doc fingerprints for SF evaluation
    doc_fp_dir = run_dir / "doc_fingerprints"
    doc_fp_path = doc_fp_dir / "doc_fingerprints.npz"
    doc_ids = [str(i) for i in range(1, n_docs + 1)]

    # =====================================================================
    # SF evaluation
    # =====================================================================
    sf_results = []
    if not args.skip_sf:
        logger.info("=" * 60)
        logger.info("SF: Building fragment fingerprints...")
        logger.info("=" * 60)

        sf_output_dir = output_dir / "sf_fragment_fingerprints"
        # Check if already built
        sf_fp_path = sf_output_dir / "doc_fingerprints.npz"
        sf_meta_path = sf_output_dir / "doc_fingerprints_meta.json"
        if sf_fp_path.exists() and sf_meta_path.exists():
            logger.info("Loading cached SF fragment fingerprints")
            sf_fp_matrix = np.load(str(sf_fp_path))["fingerprints"].astype(np.float32)
            with open(sf_meta_path, encoding="utf8") as f:
                sf_meta = json.load(f)
        else:
            sf_output_dir.mkdir(parents=True, exist_ok=True)
            sf_fp_matrix, sf_meta = build_sf_fragment_fingerprints(frag_csv, sf_output_dir)
            if sf_fp_matrix is None:
                logger.error("SF fingerprinting failed, skipping SF evaluation")
                args.skip_sf = True

        if not args.skip_sf:
            logger.info("Loading doc fingerprints for SF ranking...")
            doc_fp = np.load(str(doc_fp_path))["fingerprints"].astype(np.float32)
            logger.info(f"Doc fingerprints shape: {doc_fp.shape}")

            # Handle nested meta (doc_to_row key)
            if "doc_to_row" in sf_meta:
                sf_row_map = sf_meta["doc_to_row"]
            else:
                sf_row_map = sf_meta

            logger.info(f"SF fragment fingerprints: {len(sf_row_map)} entries (of {len(fragments)} total)")
            logger.info("Ranking fragments with SF...")
            sf_results = sf_rank_fragments(fragments, sf_fp_matrix, sf_row_map, doc_fp, doc_ids)

    # =====================================================================
    # Dense evaluation
    # =====================================================================
    dense_results = []
    if not args.skip_dense:
        logger.info("=" * 60)
        logger.info("Dense: Encoding fragments...")
        logger.info("=" * 60)

        # Build doc embeddings if not cached
        dense_cache_path = output_dir / "dense_doc_embeddings.npy"
        if dense_cache_path.exists():
            logger.info("Loading cached dense doc embeddings")
            dense_doc_emb = np.load(str(dense_cache_path))
        else:
            model = load_dense_model()
            logger.info("Encoding Arabic corpus for dense retrieval...")
            dense_doc_emb = model.encode(
                arabic_texts, batch_size=64,
                show_progress_bar=True, convert_to_numpy=True,
                normalize_embeddings=True,
            )
            np.save(str(dense_cache_path), dense_doc_emb)

        # Reload model if not already loaded
        if 'model' not in locals():
            model = load_dense_model()

        logger.info("Encoding and ranking fragments with Dense...")
        dense_results = encode_and_rank_dense(fragments, dense_doc_emb, doc_ids, model)

    # =====================================================================
    # BM25 evaluation
    # =====================================================================
    bm25_results = []
    if not args.skip_bm25:
        logger.info("=" * 60)
        logger.info("BM25: Ranking fragments...")
        logger.info("=" * 60)

        corpus_lines = [f"{ar} {en}" for ar, en in zip(arabic_texts, english_texts)]
        bm25_results = bm25_rank_fragments(fragments, corpus_lines, doc_ids)

    # =====================================================================
    # Aggregate results
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Aggregating results...")
    logger.info("=" * 60)

    # Build ground truth: for fragment from ayah N, the relevant doc is str(N+1)
    # (doc_ids are 1-indexed: ayah 0 -> doc_id "1")
    def eval_method(results, method_name):
        per_type = defaultdict(list)  # type -> list of MRR
        per_len = defaultdict(list)   # length -> list of MRR
        per_type_len = defaultdict(list)  # (type, length) -> list of MRR

        all_mrrs = []
        all_aps = []
        all_found = 0

        for fid, text, ayah_idx, ftype, flen, ranked in results:
            relevant_id = str(ayah_idx + 1)
            retrieved_ids = [doc_id for doc_id, _ in ranked]
            metrics, found = compute_metrics(retrieved_ids, [relevant_id])
            all_mrrs.append(metrics["MRR"])
            all_aps.append(metrics["AP"])
            if found:
                all_found += 1

            per_type[ftype].append(metrics["MRR"])
            per_len[flen].append(metrics["MRR"])
            per_type_len[(ftype, flen)].append(metrics["MRR"])

        n = len(all_mrrs)
        report = {
            "method": method_name,
            "n": n,
            "MRR": float(np.mean(all_mrrs)),
            "AP": float(np.mean(all_aps)),
            "SuccessRate": all_found / n if n else 0,
            "by_type": {k: float(np.mean(v)) for k, v in per_type.items()},
            "by_len": {k: float(np.mean(v)) for k, v in per_len.items()},
            "by_type_len": {str(k): float(np.mean(v)) for k, v in per_type_len.items()},
        }
        return report, per_type_len

    reports = []
    per_type_len_all = {}

    if sf_results:
        r, ptl = eval_method(sf_results, "SF")
        reports.append(r)
        per_type_len_all["SF"] = ptl

    if dense_results:
        r, ptl = eval_method(dense_results, "Dense")
        reports.append(r)
        per_type_len_all["Dense"] = ptl

    if bm25_results:
        r, ptl = eval_method(bm25_results, "BM25")
        reports.append(r)
        per_type_len_all["BM25"] = ptl

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("FRAGMENT RETRIEVAL TEST — FINAL REPORT")
    logger.info(f"  Subset: {args.n_subset} ayahs × {len(FRAGMENT_TYPES)} types × {len(FRAGMENT_LENGTHS)} lengths = {len(fragments)} fragments")
    logger.info(f"  Seed:   {args.seed}")
    logger.info("=" * 70)

    # Overall comparison
    logger.info(f"\n{'Method':<8s} {'MRR':<10s} {'AP':<10s} {'SuccessRate':<14s}")
    logger.info("-" * 42)
    for r in reports:
        logger.info(f"{r['method']:<8s} {r['MRR']:<10.4f} {r['AP']:<10.4f} {r['SuccessRate']:<10.1%}")

    # By fragment type
    logger.info("\n--- MRR by Fragment Type ---")
    header = f"{'Type':<20s}"
    for r in reports:
        header += f"{r['method']:<10s}"
    logger.info(header)
    logger.info("-" * (20 + 10 * len(reports)))
    for ftype in FRAGMENT_TYPES:
        line = f"{ftype:<20s} "
        for r in reports:
            if ftype in r["by_type"]:
                line += f"{r['by_type'][ftype]:<10.4f} "
            else:
                line += f"{'N/A':<10s} "
        logger.info(line)

    # By fragment length
    logger.info("\n--- MRR by Fragment Length ---")
    header = f"{'Length':<20s}"
    for r in reports:
        header += f"{r['method']:<10s}"
    logger.info(header)
    logger.info("-" * (20 + 10 * len(reports)))
    for flen in FRAGMENT_LENGTHS:
        line = f"{flen} chars{'':<14s} "
        for r in reports:
            if flen in r["by_len"]:
                line += f"{r['by_len'][flen]:<10.4f} "
            else:
                line += f"{'N/A':<10s} "
        logger.info(line)

    # Full grid by type × length
    logger.info("\n--- MRR by Type x Length ---")
    for r in reports:
        logger.info(f"\n{r['method']}:")
        header = f"{'Type\\Length':<20s}"
        for flen in FRAGMENT_LENGTHS:
            header += f" {flen:<6d}  "
        logger.info(header)
        logger.info("-" * (20 + 9 * len(FRAGMENT_LENGTHS)))
        for ftype in FRAGMENT_TYPES:
            line = f"{ftype:<20s}"
            for flen in FRAGMENT_LENGTHS:
                key = (ftype, flen)
                mrrs = per_type_len_all[r['method']].get(key, [])
                if mrrs:
                    line += f" {np.mean(mrrs):<.4f}  "
                else:
                    line += f" {'N/A':<6s}  "
            logger.info(line)

    # Save report
    report_path = output_dir / "fragment_report.json"
    with open(report_path, "w", encoding="utf8") as f:
        json.dump({
            "config": {
                "n_subset": args.n_subset,
                "seed": args.seed,
                "fragment_types": FRAGMENT_TYPES,
                "fragment_lengths": FRAGMENT_LENGTHS,
            },
            "reports": reports,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\nReport saved to: {report_path}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
