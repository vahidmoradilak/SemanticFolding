#!/usr/bin/env python3
"""
Build semantic QA dataset from Quran corpus and doc fingerprints.

For each existing QA pair:
  - Reference ayah = first ayah in `relevant` list
  - question = Arabic text of that ayah
  - relevance = 2 for reference ayah, 1 for top-10 nearest in semantic space

Output: data/quran/quran_qa_semantic.jsonl

With --num-random N: generates N random Arabic ayahs + QA file at
  data/quran/quran_random_N.txt and data/quran/quran_random_N_qa.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "semantic_folding"))
from lib import get_logger

logger = get_logger("build_semantic_qa")

DATA_DIR = PROJ_ROOT / "data" / "quran"
RUN_DIR = PROJ_ROOT / "outputs" / "quran_benchmark" / "runs" / "run_20260710_193034"

QA_PATH = DATA_DIR / "quran_qa.jsonl"
CORPUS_PATH = DATA_DIR / "quran_ayahs_clean.txt"
FP_NPZ = RUN_DIR / "doc_fingerprints" / "doc_fingerprints.npz"
FP_META = RUN_DIR / "doc_fingerprints" / "doc_fingerprints_meta.json"
OUT_PATH = DATA_DIR / "quran_qa_semantic.jsonl"

TOP_K = 10  # number of near ayahs to add with relevance=1


def load_corpus(path: Path) -> dict:
    """Load corpus as {line_number: (arabic, english)}."""
    corpus = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            idx = parts[0].strip()
            arabic = parts[1].strip()
            english = parts[2].strip()
            corpus[idx] = (arabic, english)
    logger.info(f"Loaded {len(corpus)} ayahs from corpus")
    return corpus


def load_fingerprints(npz_path: Path, meta_path: Path):
    """Load doc fingerprints matrix and doc_to_row mapping."""
    fp = np.load(str(npz_path), allow_pickle=True)
    matrix = fp["fingerprints"]
    with open(meta_path, "r", encoding="utf8") as f:
        meta = json.load(f)
    doc_to_row = meta["doc_to_row"]
    logger.info(f"Loaded fingerprints: {matrix.shape}, {len(doc_to_row)} docs")
    return matrix, doc_to_row


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query_vec and every row in matrix."""
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-10
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        query_norm = 1e-10
    return (matrix @ query_vec) / (norms * query_norm)


def find_near_ayahs(ref_ayah: str, matrix: np.ndarray, doc_to_row: dict, top_k: int = TOP_K) -> list:
    """Find top-k nearest ayahs (excluding ref_ayah) by cosine similarity."""
    row_idx = doc_to_row.get(ref_ayah)
    if row_idx is None:
        logger.warning(f"Ayah {ref_ayah} not found in doc_to_row")
        return []
    query_vec = matrix[row_idx]
    sims = cosine_similarity_matrix(query_vec, matrix)
    # Exclude ref_ayah itself
    sims[row_idx] = -1.0
    # Get top-k indices
    top_indices = np.argsort(sims)[::-1][:top_k]
    # Convert row indices back to ayah numbers (1-based)
    row_to_doc = {v: k for k, v in doc_to_row.items()}
    near_ayahs = [row_to_doc[i] for i in top_indices if i in row_to_doc]
    return near_ayahs


def main():
    parser = argparse.ArgumentParser(description="Build semantic QA datasets")
    parser.add_argument("--num-random", type=int, default=0,
                        help="Generate N random Arabic ayahs + QA file")
    args = parser.parse_args()

    corpus = load_corpus(CORPUS_PATH)
    matrix, doc_to_row = load_fingerprints(FP_NPZ, FP_META)

    # Always build semantic QA from existing pairs
    qa_pairs = []
    with open(QA_PATH, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qa = json.loads(line)
            qa_pairs.append(qa)

    out_entries = []
    for qa in qa_pairs:
        qid = qa["id"]
        category = qa["category"]
        relevant = qa["relevant"]
        if not relevant:
            continue
        ref_ayah = relevant[0]
        if ref_ayah not in corpus:
            continue
        arabic_text = corpus[ref_ayah][0]
        near_ayahs = find_near_ayahs(ref_ayah, matrix, doc_to_row)
        relevance = {ref_ayah: 2}
        for near in near_ayahs:
            relevance[near] = 1
        out_entries.append({
            "id": qid, "category": category, "question": arabic_text,
            "relevant": [ref_ayah] + near_ayahs, "relevance": relevance,
        })

    with open(OUT_PATH, "w", encoding="utf8") as f:
        for entry in out_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(out_entries)} entries to {OUT_PATH}")

    # Generate random ayahs + QA if --num-random > 0
    if args.num_random > 0:
        build_random_qa(corpus, matrix, doc_to_row, args.num_random)


def build_random_qa(corpus: dict, matrix: np.ndarray, doc_to_row: dict, num: int = 30):
    """Generate num random Arabic ayahs + QA file."""
    all_ayah_nums = list(corpus.keys())
    random.seed(42)
    chosen = random.sample(all_ayah_nums, min(num, len(all_ayah_nums)))

    txt_path = DATA_DIR / f"quran_random_{num}.txt"
    qa_path = DATA_DIR / f"quran_random_{num}_qa.jsonl"

    with open(txt_path, "w", encoding="utf8") as f:
        for idx in chosen:
            f.write(corpus[idx][0] + "\n")
    logger.info(f"Wrote {len(chosen)} random Arabic ayahs to {txt_path}")

    text_to_ayah = {v[0]: k for k, v in corpus.items()}
    entries = []
    missed = 0
    for i, ayah_num in enumerate(chosen):
        arabic_text = corpus[ayah_num][0]
        near_ayahs = find_near_ayahs(ayah_num, matrix, doc_to_row)
        relevance = {ayah_num: 2}
        for near in near_ayahs:
            relevance[near] = 1
        entries.append({
            "id": f"R{i+1:03d}", "category": "random", "question": arabic_text,
            "relevant": [ayah_num] + near_ayahs, "relevance": relevance,
        })

    with open(qa_path, "w", encoding="utf8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(entries)} random QA entries to {qa_path} (missed={missed})")


if __name__ == "__main__":
    main()
