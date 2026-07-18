#!/usr/bin/env python3
"""
Generate comparison charts for benchmark results.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['Belebele', 'NarrativeQA', 'PubMedQA', 'PopQA']
queries = [100, 50, 172, 200]

# MRR Results
pure_sf_mrr = [0.92, 0.91, 0.891, 0.84]
linear_mrr = [1.00, 1.00, 0.988, 0.986]
rrf_mrr = [1.00, 1.00, 1.00, 0.990]
bm25_mrr = [0.995, 0.98, 1.000, 1.000]

# AP Results
linear_ap = [1.00, 0.1609, 0.943, 0.641]
rrf_ap = [1.00, 0.2996, 0.946, 0.6975]
bm25_ap = [0.995, 0.776, 0.952, 1.000]

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: MRR Comparison
x = np.arange(len(datasets))
width = 0.2

bars1 = axes[0].bar(x - 1.5*width, pure_sf_mrr, width, label='Pure SF', color='#3498db', alpha=0.8)
bars2 = axes[0].bar(x - 0.5*width, linear_mrr, width, label='SF+SPLADE Linear', color='#2ecc71', alpha=0.8)
bars3 = axes[0].bar(x + 0.5*width, rrf_mrr, width, label='SF+SPLADE RRF', color='#e74c3c', alpha=0.8)
bars4 = axes[0].bar(x + 1.5*width, bm25_mrr, width, label='BM25', color='#9b59b6', alpha=0.8)

axes[0].set_xlabel('Dataset', fontsize=12)
axes[0].set_ylabel('MRR', fontsize=12)
axes[0].set_title('MRR Comparison Across Datasets', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets, fontsize=10)
axes[0].legend(loc='lower right', fontsize=9)
axes[0].set_ylim(0.7, 1.05)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=7)
for bar in bars2:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=7)
for bar in bars3:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=7)

# Chart 2: AP Comparison
bars5 = axes[1].bar(x - 1.0*width, linear_ap, width, label='SF+SPLADE Linear', color='#2ecc71', alpha=0.8)
bars6 = axes[1].bar(x, rrf_ap, width, label='SF+SPLADE RRF', color='#e74c3c', alpha=0.8)
bars7 = axes[1].bar(x + 1.0*width, bm25_ap, width, label='BM25', color='#9b59b6', alpha=0.8)

axes[1].set_xlabel('Dataset', fontsize=12)
axes[1].set_ylabel('AP', fontsize=12)
axes[1].set_title('AP Comparison Across Datasets', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(datasets, fontsize=10)
axes[1].legend(loc='lower right', fontsize=9)
axes[1].set_ylim(0, 1.1)

# Add value labels
for bar in bars5:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=7)
for bar in bars6:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=7)
for bar in bars7:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('D:/darsi/ms/Thesis/Dr.Banaie/code050302/SemanticFolding/outputs/benchmark_comparison.png', dpi=150, bbox_inches='tight')
print("Chart saved to outputs/benchmark_comparison.png")
