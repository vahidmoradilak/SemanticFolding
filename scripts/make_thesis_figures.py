"""Generate all thesis figures -> docs/thesis/fa/figures/*.png"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "thesis" / "fa" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── Fig 3-1: 8-step research roadmap ────────────────────────────────────────
steps = [
    ("1", "Split corpus into\nsemantic concepts"),
    ("2", "Preprocess\nconcepts"),
    ("3", "Vocabulary network\n(synonyms / OOV)"),
    ("4", "Build semantic\nmap"),
    ("5", "Fingerprint criteria\ntuning loop"),
    ("6", "Fingerprint DB\nindexing"),
    ("7", "Query processing\n& fingerprint search"),
    ("8", "Evaluation &\ncomparison"),
]
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
bw, bh = 0.215, 0.34
for i, (num, label) in enumerate(steps):
    row, col = divmod(i, 4)
    x = 0.02 + (col if row == 0 else 3 - col) * 0.245
    y = 0.60 if row == 0 else 0.10
    ax.add_patch(plt.Rectangle((x, y), bw, bh, fc="#eaf2fb", ec="#2b6cb0", lw=1.6))
    ax.text(x + bw / 2, y + bh - 0.075, num, ha="center", va="center",
            fontsize=13, fontweight="bold", color="#2b6cb0")
    ax.text(x + bw / 2, y + bh / 2 - 0.055, label, ha="center", va="center", fontsize=9)
    if col < 3:  # arrow to next box in same visual row
        ax.annotate("", xy=(x + bw + 0.031, y + bh / 2), xytext=(x + bw + 0.002, y + bh / 2),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
ax.annotate("", xy=(0.985, 0.58), xytext=(0.985, 0.46),
            arrowprops=dict(arrowstyle="-", lw=0))  # spacer
# connector: end of row1 -> start of row2
ax.annotate("", xy=(0.13, 0.44), xytext=(0.87, 0.44),
            arrowprops=dict(arrowstyle="-", color="#444", lw=1.5,
                            connectionstyle="bar,fraction=-0.25"))
fig.savefig(FIG / "fig_roadmap.png", bbox_inches="tight")
plt.close(fig)
print("fig_roadmap OK")

# ── Fig 3-2: pipeline architecture ──────────────────────────────────────────
chain = [
    ("Corpus", "#fefcbf"), ("S1 Phrases", "#feebc8"), ("S2 Term-Context\nMatrix", "#feebc8"),
    ("S3 Semantic Map\n(UMAP/t-SNE)", "#c6f6d5"), ("S4 Phrase\nFingerprints", "#bee3f8"),
    ("S5 Document\nFingerprints", "#bee3f8"), ("S7 Query Proc.\n+ SPLADE fusion", "#fed7d7"),
    ("Ranked docs", "#e9d8fd"),
]
fig, ax = plt.subplots(figsize=(13, 2.4))
ax.axis("off")
n = len(chain)
for i, (label, c) in enumerate(chain):
    x = i * (1.0 / n)
    ax.add_patch(plt.Rectangle((x + 0.004, 0.25), 0.115, 0.5, fc=c, ec="#333", lw=1.2))
    ax.text(x + 0.0615, 0.5, label, ha="center", va="center", fontsize=8.6)
    if i < n - 1:
        ax.annotate("", xy=(x + 0.125, 0.5), xytext=(x + 0.119, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
ax.text(0.5, 0.06, "Indexing (offline): S1-S5   |   Query time: S7 over pre-built SDRs",
        ha="center", fontsize=9, style="italic", color="#555")
fig.savefig(FIG / "fig_pipeline.png", bbox_inches="tight")
plt.close(fig)
print("fig_pipeline OK")

# ── Fig 4-1: main results grouped bars ──────────────────────────────────────
benchmarks = ["Belebele", "NarrativeQA", "PubMedQA", "PopQA", "MuSiQue",
              "Quran", "SciFact", "nfcorpus", "SciDocs", "AR-EN 488", "MIXED 488",
              "TyDi-ar*", "MIRACL-ar*"]
sf_best = [1.000, 1.000, 1.000, 0.990, 0.507, 0.358, 0.966, 0.655, 0.947, 0.8248, 0.8231,
           0.5436, 0.5420]
bm25 = [0.995, 0.980, 1.000, 1.000, 0.622, 0.155, 0.947, 0.686, 0.946, 0.7854, 0.7854,
        0.8806, 0.8152]
x = np.arange(len(benchmarks)); w = 0.38
fig, ax = plt.subplots(figsize=(11.5, 4.2))
ax.bar(x - w / 2, sf_best, w, label="Best SF variant", color="#2b6cb0")
ax.bar(x + w / 2, bm25, w, label="BM25", color="#dd6b20")
ax.set_xticks(x); ax.set_xticklabels(benchmarks, rotation=28, ha="right", fontsize=9)
ax.set_ylabel("MRR"); ax.set_ylim(0, 1.08)
ax.axhline(1.0, color="#999", ls=":", lw=0.8)
ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.3)
ax.set_title("Best Semantic Folding variant vs BM25 (MRR)")
fig.tight_layout()
fig.savefig(FIG / "fig_results_mrr.png", bbox_inches="tight")
plt.close(fig)
print("fig_results_mrr OK")

# ── Fig 4-2: SPLADE alpha sweep (Belebele) ──────────────────────────────────
alphas = [0.1, 0.2, 0.25, 0.27, 0.30, 0.35, 0.40, 0.50, 0.70]
mrrs = [0.97, 0.97, 0.98, 0.98, 0.98, 0.97, 0.97, 0.94, 0.92]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(alphas, mrrs, "o-", color="#2b6cb0", lw=1.8, label="SF+SPLADE Linear")
ax.plot([0.30], [0.98], "o", ms=11, mfc="#f6ad55", mec="#2b6cb0", zorder=5)
ax.annotate(r"$\alpha^*=0.3$", (0.30, 0.98), textcoords="offset points", xytext=(14, -14))
ax.axhline(0.995, color="#dd6b20", ls="--", lw=1.4, label="BM25")
ax.axhline(0.92, color="#777", ls=":", lw=1.2, label="Pure SF")
ax.set_xlabel(r"SPLADE weight $1-\alpha$  (SF weight $\alpha$)")
ax.set_ylabel("MRR"); ax.set_title("Linear-fusion sweep - Belebele (100 queries)")
ax.set_ylim(0.90, 1.01); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_alpha_sweep.png", bbox_inches="tight")
plt.close(fig)
print("fig_alpha_sweep OK")

# ── Fig 4-3: parameter ablation (AP) ────────────────────────────────────────
variants = ["A base\n(grid128)", "B spread=0", "C spread=2", "D top=5%", "E top=15%",
            "F uniform", "G sigma=1.0", "H sigma=2.0", "I grid=64"]
ap_vals = [0.836, 0.784, 0.836, 0.806, 0.779, 0.772, 0.836, 0.824, 0.869]
colors = ["#a0aec0"] * 8 + ["#38a169"]
fig, ax = plt.subplots(figsize=(9.5, 3.8))
bars = ax.bar(variants, ap_vals, color=colors)
ax.bar_label(bars, fmt="%.3f", fontsize=8.5)
ax.set_ylabel("Average Precision"); ax.set_ylim(0.70, 0.90)
ax.set_title("Parameter sweep - QA-sample corpus (5 queries, C00-C19)")
ax.grid(axis="y", alpha=0.3)
plt.xticks(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig_ablation_ap.png", bbox_inches="tight")
plt.close(fig)
print("fig_ablation_ap OK")

# ── Fig 3-3: semantic space scatter (UMAP coordinates) ──────────────────────
RUN = ROOT / "outputs" / "custom_ar_en_benchmark" / "runs" / "run_20260818_100234"
coords = json.load(open(RUN / "semantic_space" / "context_coordinates.json", encoding="utf-8"))
vals = list(coords.values())
xs = [v["x"] for v in vals]; ys = [v["y"] for v in vals]
fig, ax = plt.subplots(figsize=(5.6, 5.2))
sc = ax.scatter(xs, ys, s=12, alpha=0.75, c=np.arange(len(xs)), cmap="viridis")
ax.set_title("Semantic map - UMAP placement of 488 concepts")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.invert_yaxis()
ax.set_aspect("equal"); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "fig_semantic_space.png", bbox_inches="tight")
plt.close(fig)
print("fig_semantic_space OK")

# ── Fig 3-4: document fingerprint heatmap (Morton-decoded 64x64) ────────────
sys.path.insert(0, str(ROOT / "semantic_folding"))
from lib import morton_to_xy  # noqa: E402
from scipy.sparse import load_npz  # noqa: E402

z = np.load(RUN / "doc_fingerprints" / "doc_fingerprints.npz")
D = z["fingerprints"]
meta = json.load(open(RUN / "doc_fingerprints" / "doc_fingerprints_meta.json", encoding="utf-8"))
doc_to_row = meta["doc_to_row"]
g = 64


def decode(vec):
    grid = np.zeros((g, g), dtype=float)
    for idx, val in enumerate(vec):
        if val != 0:
            x_, y_ = morton_to_xy(int(idx), g)
            grid[y_, x_] = val
    return grid

g0 = decode(D[doc_to_row["doc_000000"]])
g1 = decode(D[doc_to_row["doc_000001"]])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
im0 = axes[0].imshow(g0, cmap="hot", interpolation="nearest")
axes[0].set_title("doc_000000 - fingerprint values")
im1 = axes[1].imshow(np.clip(g1, 0, None) > 0, cmap="Greys", interpolation="nearest")
axes[1].set_title("doc_000001 - active-cell mask")
for a in axes:
    a.set_xticks([]); a.set_yticks([])
fig.colorbar(im0, ax=axes[0], fraction=0.046)
fig.tight_layout()
fig.savefig(FIG / "fig_fingerprint_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("fig_fingerprint_heatmap OK")

print("ALL FIGURES DONE ->", FIG)
