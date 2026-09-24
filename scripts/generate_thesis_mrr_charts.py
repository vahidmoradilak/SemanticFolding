"""Thesis MRR comparison charts (10 benchmarks) -> outputs/thesis_charts/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/thesis_charts"

datasets = ["Belebele", "NarrativeQA", "PubMedQA", "PopQA", "SciFact",
            "SciDocs", "Nfcorpus", "MuSiQue", "Quran", "Belebele AR-EN"]

sf      = np.array([0.995, 0.990, 0.977, 0.985, 0.918, 0.958, 0.609, 0.565, 0.334, 0.817])
bm25    = np.array([0.995, 0.980, 1.000, 1.000, 0.947, 0.946, 0.686, 0.656, 0.155, 0.785])
splade  = np.array([1.000, 0.970, 1.000, 0.565, 0.958, 0.394, 0.679, 0.748, 0.356, 0.481])
linear  = np.array([1.000, 1.000, 0.988, 0.986, 0.966, 0.947, 0.655, 0.759, 0.320, 0.825])
rrf     = np.array([1.000, 1.000, 1.000, 0.990, 0.953, 0.828, 0.647, 0.664, 0.375, 0.683])
minilm  = np.array([0.952, 0.972, 0.974, 0.930, 0.861, 0.868, 0.401, 0.523, 0.318, 0.184])
e5      = np.array([0.971, 0.990, 0.981, 0.996, 0.865, 0.736, 0.405, 0.520, 0.235, 0.747])
tuned   = np.maximum.reduce([sf, linear, rrf])

COLORS = {
    "SF": "#2b6cb0", "BM25": "#dd6b20", "SPLADE": "#38a169",
    "MiniLM": "#805ad5", "E5": "#b7791f", "Linear": "#319795",
    "RRF": "#e53e3e", "Tuned SF": "#1a365d",
}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})


def grouped(ax, series, width=0.8):
    x = np.arange(len(datasets))
    w = width / len(series)
    # smaller font when many series share one dataset slot
    lab_fs = 7 if len(series) <= 2 else (6 if len(series) == 3 else 5)
    for i, (label, vals, color) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        kw = dict(hatch="//", edgecolor="white") if label == "Tuned SF" else {}
        bars = ax.bar(x + off, vals, w * 0.92, label=label, color=color, **kw)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, float(v) + 0.015, f"{float(v):.3f}",
                    ha="center", va="bottom", fontsize=lab_fs, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=28, ha="right")
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 1.28)  # extra headroom for 3-decimal value labels
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=min(len(series), 5),
              fontsize=9, frameon=False)


def linechart(ax, series):
    x = np.arange(len(datasets))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for i, (label, vals, color) in enumerate(series):
        ax.plot(x, vals, marker=markers[i % len(markers)], markersize=6,
                linewidth=1.8, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=28, ha="right")
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 1.08)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=min(len(series), 5),
              fontsize=9, frameon=False)


def save(fig, name, title):
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"{name} OK | Tuned={tuned.round(3).tolist()}")

# 1. SF vs BM25
fig, ax = plt.subplots(figsize=(12, 4.8))
grouped(ax, [("SF", sf, COLORS["SF"]), ("BM25", bm25, COLORS["BM25"])], width=0.55)
save(fig, "fig1_sf_vs_bm25", "SF vs BM25 (MRR across 10 benchmarks)")

# 2. SF vs dense (MiniLM, E5-multilingual)
fig, ax = plt.subplots(figsize=(12.5, 4.8))
grouped(ax, [("SF", sf, COLORS["SF"]), ("MiniLM", minilm, COLORS["MiniLM"]),
             ("E5-multilingual", e5, COLORS["E5"])], width=0.7)
save(fig, "fig2_sf_vs_dense", "SF vs Dense Retrievers (MRR)")

# 3. SF vs SPLADE
fig, ax = plt.subplots(figsize=(12, 4.8))
grouped(ax, [("SF", sf, COLORS["SF"]), ("SPLADE", splade, COLORS["SPLADE"])], width=0.55)
save(fig, "fig3_sf_vs_splade", "SF vs SPLADE (MRR across 10 benchmarks)")

# 4. All methods together
fig, ax = plt.subplots(figsize=(13.5, 5.2))
grouped(ax, [("SF", sf, COLORS["SF"]), ("BM25", bm25, COLORS["BM25"]),
             ("MiniLM", minilm, COLORS["MiniLM"]), ("E5-multilingual", e5, COLORS["E5"]),
             ("SPLADE", splade, COLORS["SPLADE"])], width=0.85)
save(fig, "fig4_all_methods", "All Methods Compared: SF, BM25, MiniLM, E5-multilingual, SPLADE (MRR)")

# 5. SF family (SF, SF+Linear, SF+RRF, Tuned SF)
fig, ax = plt.subplots(figsize=(13, 5.0))
grouped(ax, [("SF", sf, COLORS["SF"]), ("SF+Linear", linear, COLORS["Linear"]),
             ("SF+RRF", rrf, COLORS["RRF"]), ("Tuned SF", tuned, COLORS["Tuned SF"])], width=0.8)
save(fig, "fig5_sf_family", "SF Family: SF, SF+Linear, SF+RRF, Tuned SF = per-dataset max (MRR)")

# 6. Tuned SF instead of SF, all methods
fig, ax = plt.subplots(figsize=(13.5, 5.2))
grouped(ax, [("Tuned SF", tuned, COLORS["Tuned SF"]), ("BM25", bm25, COLORS["BM25"]),
             ("MiniLM", minilm, COLORS["MiniLM"]), ("E5-multilingual", e5, COLORS["E5"]),
             ("SPLADE", splade, COLORS["SPLADE"])], width=0.85)
save(fig, "fig6_tuned_all_methods",
     "All Methods Compared with Tuned SF (max of SF / SF+Linear / SF+RRF) (MRR)")

# 4b. Line chart version of step 4 (which line stays on top is visible)
fig, ax = plt.subplots(figsize=(13.5, 5.2))
linechart(ax, [("SF", sf, COLORS["SF"]), ("BM25", bm25, COLORS["BM25"]),
               ("MiniLM", minilm, COLORS["MiniLM"]), ("E5-multilingual", e5, COLORS["E5"]),
               ("SPLADE", splade, COLORS["SPLADE"])])
save(fig, "fig4b_all_methods_line", "All Methods Compared — Line View: SF, BM25, MiniLM, E5, SPLADE (MRR)")

# 6b. Line chart version of step 6 (Tuned SF vs rest)
fig, ax = plt.subplots(figsize=(13.5, 5.2))
linechart(ax, [("Tuned SF", tuned, COLORS["Tuned SF"]), ("BM25", bm25, COLORS["BM25"]),
               ("MiniLM", minilm, COLORS["MiniLM"]), ("E5-multilingual", e5, COLORS["E5"]),
               ("SPLADE", splade, COLORS["SPLADE"])])
save(fig, "fig6b_tuned_all_methods_line",
     "All Methods Compared — Line View with Tuned SF (MRR)")

# Tuned SF table for thesis appendix
with open(f"{OUT}/tuned_sf_table.md", "w", encoding="utf-8") as f:
    f.write("| Benchmark | SF | SF+Linear | SF+RRF | Tuned SF |\n|---|---|---|---|---|\n")
    for d, a, b, c, t in zip(datasets, sf, linear, rrf, tuned):
        f.write(f"| {d} | {a:.3f} | {b:.3f} | {c:.3f} | **{t:.3f}** |\n")
print("tuned_sf_table.md OK")

# ── Fig 7: mean-MRR ranking (horizontal, sorted) ──────────────────────────
order = [("MiniLM", minilm, COLORS["MiniLM"]), ("SPLADE", splade, COLORS["SPLADE"]),
         ("E5-multilingual", e5, COLORS["E5"]), ("SF+RRF", rrf, COLORS["RRF"]),
         ("SF", sf, COLORS["SF"]), ("BM25", bm25, COLORS["BM25"]),
         ("SF+Linear", linear, COLORS["Linear"]), ("Tuned SF", tuned, COLORS["Tuned SF"])]
order = sorted(order, key=lambda t: t[1].mean())
fig, ax = plt.subplots(figsize=(8.5, 4.6))
labels = [t[0] for t in order]
means = [float(t[1].mean()) for t in order]
cols = [t[2] for t in order]
bars = ax.barh(labels, means, color=cols)
ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=4)
ax.set_xlabel("Mean MRR (10 benchmarks)")
ax.set_xlim(0, 1.0)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
save(fig, "fig7_mean_mrr_ranking", "Mean MRR Ranking Across 10 Benchmarks (higher is better)")

# ── Fig 8: outright-win counts (Tuned-SF setup, ties shared) ──────────────
allM = np.stack([tuned, bm25, splade, minilm, e5], axis=1)
win_names = ["Tuned SF", "BM25", "SPLADE", "MiniLM", "E5-multilingual"]
win_cols = [COLORS["Tuned SF"], COLORS["BM25"], COLORS["SPLADE"], COLORS["MiniLM"], COLORS["E5"]]
wins = [sum(1 for i in range(len(datasets)) if allM[i, j] == allM[i].max())
        for j in range(len(win_names))]
idx = np.argsort(wins)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
bars = ax.barh([win_names[i] for i in idx], [wins[i] for i in idx],
               color=[win_cols[i] for i in idx])
ax.bar_label(bars, fmt="%d", fontsize=10, padding=4)
ax.set_xlabel("# benchmarks where method is best (ties shared, out of 10)")
ax.set_xlim(0, 10)
ax.set_xticks(range(0, 11))
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
save(fig, "fig8_best_counts", "How Often Is Each Method the Best? (Tuned-SF setup, ties shared)")

# ── Fig 9: diverging delta Tuned SF − BM25 per dataset ────────────────────
delta = tuned - bm25
sidx = np.argsort(delta)
fig, ax = plt.subplots(figsize=(8.5, 5.0))
dvals = delta[sidx]
dlabs = [datasets[i] for i in sidx]
dcols = ["#38a169" if v >= 0 else "#e53e3e" for v in dvals]
bars = ax.barh(dlabs, dvals, color=dcols)
ax.bar_label(bars, fmt="%+.3f", fontsize=8, padding=3)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlim(float(dvals.min()) - 0.03, float(dvals.max()) + 0.05)
ax.set_xlabel("Tuned SF − BM25 (MRR points; green = Tuned SF wins)")
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
save(fig, "fig9_delta_tuned_minus_bm25", "Where Does Tuned SF Beat BM25? (per-dataset MRR gap)")

# ── Fig 10: robustness — mean MRR vs std (bubble = worst-case min) ────────
methods = [("SF", sf, COLORS["SF"]), ("BM25", bm25, COLORS["BM25"]),
           ("SPLADE", splade, COLORS["SPLADE"]), ("SF+Linear", linear, COLORS["Linear"]),
           ("SF+RRF", rrf, COLORS["RRF"]), ("MiniLM", minilm, COLORS["MiniLM"]),
           ("E5-multilingual", e5, COLORS["E5"]), ("Tuned SF", tuned, COLORS["Tuned SF"])]
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for lab, v, c in methods:
    ax.scatter([float(v.mean())], [float(v.std())], s=90, color=c, label=lab, zorder=3)
    ax.annotate(f" {lab} (min {float(v.min()):.2f})", (float(v.mean()), float(v.std())),
                fontsize=8, va="center")
ax.set_xlabel("Mean MRR → (higher is better)")
ax.set_ylabel("Std across benchmarks → (lower = more stable)")
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="upper right", fontsize=8, frameon=True)
save(fig, "fig10_robustness_mean_vs_std", "Robustness: Mean MRR vs Cross-Benchmark Stability")
