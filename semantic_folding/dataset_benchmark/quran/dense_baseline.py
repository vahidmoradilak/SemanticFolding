#!/usr/bin/env python3
"""
Dense Retrieval Baseline for Quran Benchmark
Uses multilingual Sentence Transformers for Arabic-text retrieval.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import get_logger
logger = get_logger("dense_baseline")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = PROJECT_ROOT / "data" / "quran" / "quran_ayahs_clean.txt"
QA_PATH = PROJECT_ROOT / "data" / "quran" / "quran_qa_semantic_gold_only.jsonl"


def load_corpus(path):
    arabic_texts, english_texts, doc_ids = [], [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            idx, ar, en = parts[0], parts[1], parts[2]
            doc_ids.append(idx)
            arabic_texts.append(ar)
            english_texts.append(en)
    return doc_ids, arabic_texts, english_texts


def load_qa(path):
    pairs = []
    with open(path, encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            pairs.append({
                "id": data["id"],
                "category": data.get("category", ""),
                "query": data["question"],
                "relevant": data["relevant"],
                "relevance": data["relevance"],
            })
    return pairs


def compute_metrics(retrieved_ids, relevant, top_k_list=None):
    if top_k_list is None:
        top_k_list = [1, 2, 3, 5, 10]
    rel_set = set(relevant)

    found_at = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            found_at = rank
            break
    mrr = 1.0 / found_at if found_at > 0 else 0.0

    ap = 0.0
    hits = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            hits += 1
            ap += hits / rank
    ap /= len(relevant) if relevant else 1

    metrics = {"MRR": mrr, "AP": ap}
    for k in top_k_list:
        retrieved_k = retrieved_ids[:k]
        rel_k = sum(1 for d in retrieved_k if d in rel_set)
        metrics[f"P@{k}"] = rel_k / k
        metrics[f"R@{k}"] = rel_k / len(relevant) if relevant else 0.0

    for k in top_k_list:
        dcg_k = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in rel_set:
                dcg_k += 1.0 / math.log2(rank + 1)
        num_rel = min(len(relevant), k)
        idcg_k = sum(1.0 / math.log2(i + 2) for i in range(num_rel))
        metrics[f"NDCG@{k}"] = dcg_k / idcg_k if idcg_k > 0 else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Dense Retrieval Baseline for Quran")
    parser.add_argument("--qa-file", default=str(QA_PATH),
                        help="Path to QA JSONL file")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Sentence Transformer model name")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for encoding")
    parser.add_argument("--text-field", choices=["arabic", "english", "both"], default="arabic",
                        help="Which text field to use for encoding")
    parser.add_argument("--device", default=None,
                        help="Device (cpu, cuda, etc.)")
    args = parser.parse_args()

    logger.info(f"Loading corpus from {CORPUS_PATH}")
    doc_ids, arabic_texts, english_texts = load_corpus(CORPUS_PATH)

    if args.text_field == "arabic":
        texts = arabic_texts
    elif args.text_field == "english":
        texts = english_texts
    else:
        texts = [f"{ar} {en}" for ar, en in zip(arabic_texts, english_texts)]

    n_docs = len(texts)
    logger.info(f"Corpus: {n_docs} documents")

    logger.info(f"Loading model: {args.model} ...")
    t0 = time.time()
    model = SentenceTransformer(args.model, device=args.device)
    logger.info(f"Model loaded ({time.time()-t0:.1f}s)")

    logger.info("Encoding corpus (~2-5 min for 6236 docs)...")
    t0 = time.time()
    doc_embeddings = model.encode(
        texts, batch_size=args.batch_size,
        show_progress_bar=True, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info(f"Encoded {doc_embeddings.shape[0]} docs in {time.time()-t0:.1f}s")

    qa_pairs = load_qa(args.qa_file)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")

    all_metrics = {}
    per_query = {}

    for qa in tqdm(qa_pairs, desc="Evaluating"):
        q_emb = model.encode(
            [qa["query"]], convert_to_numpy=True, normalize_embeddings=True,
        )
        sims = np.dot(doc_embeddings, q_emb.T).flatten()
        ranked = [doc_ids[i] for i in np.argsort(-sims)]

        relevant = qa["relevant"]
        relevance = qa["relevance"]
        metrics = compute_metrics(ranked, relevant)
        all_metrics[qa["id"]] = metrics

        rel_rank = next((i+1 for i, d in enumerate(ranked) if d in relevant), None)
        per_query[qa["id"]] = {
            "relevant_rank": rel_rank,
            "top1_id": ranked[0],
            "top1_sim": float(sims[0]),
        }

    # Aggregate
    metric_names = ["MRR", "AP", "P@5", "R@5", "P@10", "R@10", "NDCG@10"]
    agg = {}
    for metric in metric_names:
        vals = [all_metrics[qid][metric] for qid in all_metrics]
        agg[metric] = float(np.mean(vals))

    successes = sum(1 for qid in all_metrics if all_metrics[qid]["MRR"] > 0)
    logger.info("=" * 70)
    logger.info(f"Dense Retrieval Baseline  |  model={args.model}  |  text={args.text_field}")
    logger.info(f"QA: {args.qa_file} ({len(qa_pairs)} pairs)  |  Corpus: {n_docs} docs")
    logger.info("-" * 70)
    for m in ["MRR", "AP", "P@5", "R@5", "P@10", "R@10", "NDCG@10"]:
        logger.info(f"  {m:>8s} = {agg[m]:.4f}")
    logger.info(f"  SuccessRate = {successes}/{len(qa_pairs)} ({100*successes/len(qa_pairs):.1f}%)")
    logger.info("=" * 70)

    logger.info(f"\n{'ID':<6s} {'Category':<18s} {'MRR':<8s} {'P@5':<8s} {'R@5':<8s} {'NDCG@10':<10s} {'Rank':<6s}")
    logger.info("-" * 70)
    for qa in qa_pairs:
        m = all_metrics[qa["id"]]
        r = per_query[qa["id"]]["relevant_rank"]
        rr = str(r) if r else "NF"
        logger.info(f"{qa['id']:<6s} {qa.get('category',''):<18s} {m['MRR']:<8.4f} {m['P@5']:<8.4f} {m['R@5']:<8.4f} {m['NDCG@10']:<10.4f} {rr:<6s}")

    logger.info("\n#SUMMARY# "
        "dense MRR={:.4f} AP={:.4f} P@5={:.4f} R@5={:.4f} NDCG@10={:.4f} SR={:.1f}% (model={}, text={})".format(
        agg["MRR"], agg["AP"], agg["P@5"], agg["R@5"], agg["NDCG@10"],
        100*successes/len(qa_pairs), args.model, args.text_field))

    return agg


if __name__ == "__main__":
    main()
