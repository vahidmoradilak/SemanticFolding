import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =====================================================
# CONFIG
# =====================================================

GRID_SIZE = 128

RUN_DIR = Path("outputs/20260519_131648")

FP_PATH = RUN_DIR / "phrase_fingerprints" / "phrase_fingerprints.npz"

META_PATH = RUN_DIR / "phrase_fingerprints" / "phrase_fingerprints_meta.json"

OUTPUT_DIR = Path("outputs/20260519_131648/fingerprint_visualizations_binary")

OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading fingerprints...")

data = np.load(FP_PATH)

fingerprints = data["fingerprints"]

with open(META_PATH, encoding="utf-8") as f:
    phrase_to_index = json.load(f)

print(f"Loaded {len(phrase_to_index)} phrases")


# =====================================================
# HELPERS
# =====================================================

def get_binary_fingerprint(phrase: str):

    if phrase not in phrase_to_index:
        raise ValueError(f"Phrase not found: {phrase}")

    idx = phrase_to_index[phrase]

    fp = fingerprints[idx]

    fp_2d = fp.reshape((GRID_SIZE, GRID_SIZE))

    # strict binary SDR
    binary = (fp_2d > 0).astype(np.uint8)

    return binary

def visualize_binary_fingerprint(phrase: str):
    binary = get_binary_fingerprint(phrase)

    plt.figure(figsize=(10, 10))

    plt.imshow(binary, cmap="binary", interpolation="nearest")

    plt.title(f"Binary Semantic Fingerprint\n{phrase}", fontweight="bold")

    plt.axis("off")

    active_bits = int(binary.sum())

    print(f"Active bits: {active_bits}")

    save_path = OUTPUT_DIR / f"binary_{phrase}.png"

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print("Saved:", save_path)


def jaccard_similarity(a, b):

    intersection = np.logical_and(a, b).sum()

    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return intersection / union


# =====================================================
# TRUE BINARY OVERLAP VISUALIZATION
# =====================================================

def visualize_binary_overlap(phrase1: str, phrase2: str):

    fp1 = get_binary_fingerprint(phrase1)

    fp2 = get_binary_fingerprint(phrase2)

    # -------------------------------------------------
    # overlap regions
    # -------------------------------------------------

    overlap = np.logical_and(fp1, fp2).astype(np.uint8)

    # phrase1 only
    only_1 = np.logical_and(fp1, np.logical_not(fp2))

    # phrase2 only
    only_2 = np.logical_and(fp2, np.logical_not(fp1))

    similarity = jaccard_similarity(fp1, fp2)

    # -------------------------------------------------
    # STATS
    # -------------------------------------------------

    active_1 = int(fp1.sum())

    active_2 = int(fp2.sum())

    active_overlap = int(overlap.sum())

    print("\n====================================")
    print(f"Phrase 1: {phrase1}")
    print(f"Phrase 2: {phrase2}")
    print("====================================")

    print(f"Active bits phrase1 : {active_1}")
    print(f"Active bits phrase2 : {active_2}")
    print(f"Overlap active bits : {active_overlap}")

    print(f"Jaccard similarity  : {similarity:.4f}")

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # -----------------------------------------
    # phrase1
    # -----------------------------------------

    axes[0].imshow(
        fp1,
        cmap="binary",
        interpolation="nearest"
    )

    axes[0].set_title(
        f"{phrase1}\n({active_1} active bits)"
    )

    axes[0].axis("off")

    # -----------------------------------------
    # phrase2
    # -----------------------------------------

    axes[1].imshow(
        fp2,
        cmap="binary",
        interpolation="nearest"
    )

    axes[1].set_title(
        f"{phrase2}\n({active_2} active bits)"
    )

    axes[1].axis("off")

    # -----------------------------------------
    # overlap
    # -----------------------------------------

    axes[2].imshow(
        overlap,
        cmap="binary",
        interpolation="nearest"
    )

    axes[2].set_title(
        f"Semantic Overlap\n({active_overlap} shared bits)"
    )

    axes[2].axis("off")

    # -----------------------------------------
    # difference map
    # -----------------------------------------

    diff_map = np.zeros((GRID_SIZE, GRID_SIZE))

    # phrase1 only = 1
    diff_map[only_1] = 1

    # phrase2 only = 2
    diff_map[only_2] = 2

    # overlap = 3
    diff_map[overlap == 1] = 3

    axes[3].imshow(
        diff_map,
        interpolation="nearest"
    )

    axes[3].set_title(
        f"Difference Map\nJaccard={similarity:.4f}"
    )

    axes[3].axis("off")

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    filename = f"binary_{phrase1}_VS_{phrase2}.png"

    save_path = OUTPUT_DIR / filename

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0.2
    )

    plt.close()

    print("\nSaved:")
    print(save_path)


# =====================================================
# EXAMPLES
# =====================================================

if __name__ == "__main__":

    visualize_binary_fingerprint("لله")
    
    visualize_binary_overlap("لله", "god")