import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

GRID_SIZE = 128

RUN_DIR = Path("outputs/20260519_131648")

FP_PATH = RUN_DIR / "phrase_fingerprints" / "phrase_fingerprints.npz"

META_PATH = RUN_DIR / "phrase_fingerprints" / "phrase_fingerprints_meta.json"

OUTPUT_DIR = Path("outputs/20260519_131648/fingerprint_visualizations_semantic")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading fingerprints...")

data = np.load(FP_PATH)

fingerprints = data["fingerprints"]

with open(META_PATH, encoding="utf-8") as f:
    phrase_to_index = json.load(f)

print(f"Loaded {len(phrase_to_index)} phrases")


# =========================================================
# HELPERS
# =========================================================

def get_fingerprint_2d(phrase: str):

    if phrase not in phrase_to_index:
        # raise ValueError(f"Phrase not found: {phrase}")
        print(f"Phrase not found: {phrase}")
        return

    idx = phrase_to_index[phrase]

    fp = fingerprints[idx]

    fp_2d = fp.reshape((GRID_SIZE, GRID_SIZE))

    return fp_2d


def cosine_similarity(a, b):

    a = a.flatten()
    b = b.flatten()

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom


# =========================================================
# VISUALIZE SINGLE PHRASE
# =========================================================

def visualize_single_phrase(phrase: str):

    fp = get_fingerprint_2d(phrase)

    plt.figure(figsize=(8, 8))

    plt.imshow(fp, cmap="viridis") # color map: viridis, binary, hot, gray, coolwarm

    plt.title(f"Semantic Fingerprint\n{phrase}", fontweight="bold")

    plt.colorbar()

    save_path = OUTPUT_DIR / f"fingerprint_{phrase}.png"

    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close()

    print("Saved:", save_path)


# =========================================================
# VISUALIZE COMPARISON
# =========================================================

def visualize_comparison(phrase1: str, phrase2: str):

    fp1 = get_fingerprint_2d(phrase1)

    fp2 = get_fingerprint_2d(phrase2)

    overlap = np.minimum(fp1, fp2)

    similarity = cosine_similarity(fp1, fp2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # -----------------------------------------
    # Phrase 1
    # -----------------------------------------
    axes[0].imshow(fp1, cmap="viridis")

    axes[0].set_title(phrase1)

    # -----------------------------------------
    # Phrase 2
    # -----------------------------------------
    axes[1].imshow(fp2, cmap="viridis")

    axes[1].set_title(phrase2)

    # -----------------------------------------
    # Overlap
    # -----------------------------------------
    axes[2].imshow(overlap, cmap="hot")

    axes[2].set_title(
        f"Semantic Overlap\nSimilarity = {similarity:.4f}"
    )

    save_path = OUTPUT_DIR / f"{phrase1}_VS_{phrase2}.png"

    plt.savefig(save_path, bbox_inches="tight")

    plt.close()

    print("Saved:", save_path)


# =========================================================
# VISUALIZE BINARY SDR
# =========================================================

def visualize_binary_sdr(phrase: str):

    fp = get_fingerprint_2d(phrase)

    binary = (fp > 0).astype(int)

    plt.figure(figsize=(8, 8))

    plt.imshow(binary, cmap="binary")

    plt.title(f"Binary Semantic SDR\n{phrase}", fontweight="bold")

    save_path = OUTPUT_DIR / f"binary_sdr_{phrase}.png"

    plt.savefig(save_path, bbox_inches="tight")

    plt.close()

    print("Saved:", save_path)


# =========================================================
# TOP-K SIMILAR PHRASES
# =========================================================

def find_most_similar(phrase: str, top_k=10):

    target = get_fingerprint_2d(phrase)

    sims = []

    for other_phrase in phrase_to_index:

        if other_phrase == phrase:
            continue

        other_fp = get_fingerprint_2d(other_phrase)

        sim = cosine_similarity(target, other_fp)

        sims.append((other_phrase, sim))

    sims.sort(key=lambda x: x[1], reverse=True)

    print("\nMost similar phrases:\n")

    for p, s in sims[:top_k]:
        print(f"{s:.4f}  ->  {p}")


# =========================================================
# EXAMPLES
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------
    # single fingerprint
    # -----------------------------------------
    visualize_single_phrase("لله")

    # -----------------------------------------
    # binary SDR style
    # -----------------------------------------
    visualize_binary_sdr("لله")

    # -----------------------------------------
    # semantic comparison
    # -----------------------------------------
    visualize_comparison("لله", "god")

    # -----------------------------------------
    # similarity search
    # -----------------------------------------
    find_most_similar("لله", top_k=10)