#!/usr/bin/env python3
"""
Generate comprehensive block diagram of Semantic Folding pipeline.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
colors = {
    'input': '#E8F4FD',
    'step1': '#FFE4B5',
    'step2': '#98FB98',
    'step3': '#87CEEB',
    'step4': '#DDA0DD',
    'step5': '#F0E68C',
    'step7': '#FFA07A',
    'output': '#90EE90',
    'fusion': '#FFD700',
    'benchmark': '#FF6B6B',
}

def draw_box(x, y, w, h, text, color, fontsize=10):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow(x1, y1, x2, y2, text=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2, text, ha='center', va='center',
                fontsize=8, color='gray')

# Title
ax.text(10, 13.5, 'Semantic Folding Pipeline', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#333333')
ax.text(10, 13.0, 'Complete System Architecture', ha='center', va='center',
        fontsize=14, color='#666666')

# === INPUT ===
draw_box(0.5, 11.5, 3, 1, 'Input Data\n(JSONL Format)', colors['input'], 11)
ax.text(2, 11.0, '• id, question, answer\n• paragraphs[]\n• is_supporting', 
        ha='center', va='center', fontsize=8, color='#333333')

# === PHASE 1: INDEXING ===
ax.text(10, 12.2, 'PHASE 1: INDEXING', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#0066CC')

# Step 1
draw_box(0.5, 9.5, 3.5, 1.2, 'Step 1: Phrase\nExtraction', colors['step1'], 11)
ax.text(2.25, 9.0, '• spaCy NLP\n• POS tagging\n• Unigram/Bigram/Trigram', 
        ha='center', va='center', fontsize=7, color='#333333')

# Step 2
draw_box(4.5, 9.5, 3.5, 1.2, 'Step 2: Term-Context\nMatrix (TF-IDF)', colors['step2'], 11)
ax.text(6.25, 9.0, '• IDF weighting\n• Sparse matrix\n• Vocabulary mapping', 
        ha='center', va='center', fontsize=7, color='#333333')

# Step 3
draw_box(8.5, 9.5, 3.5, 1.2, 'Step 3: Semantic\nSpace (UMAP/t-SNE)', colors['step3'], 11)
ax.text(10.25, 9.0, '• Dimension reduction\n• 2D grid placement\n• Collision resolution', 
        ha='center', va='center', fontsize=7, color='#333333')

# Step 4
draw_box(12.5, 9.5, 3.5, 1.2, 'Step 4: Phrase\nFingerprints', colors['step4'], 11)
ax.text(14.25, 9.0, '• Gaussian smoothing\n• Morton Z-order\n• Multi-hot encoding', 
        ha='center', va='center', fontsize=7, color='#333333')

# Step 5
draw_box(16.5, 9.5, 3.5, 1.2, 'Step 5: Document\nFingerprints', colors['step5'], 11)
ax.text(18.25, 9.0, '• TF-IDF union\n• Peak detection\n• Sparsification', 
        ha='center', va='center', fontsize=7, color='#333333')

# Arrows for Phase 1
draw_arrow(2, 11.5, 2, 10.7)
draw_arrow(4, 10.1, 4.5, 10.1)
draw_arrow(8, 10.1, 8.5, 10.1)
draw_arrow(12, 10.1, 12.5, 10.1)
draw_arrow(16, 10.1, 16.5, 10.1)

# === OUTPUTS ===
draw_box(0.5, 7.5, 4, 1, 'Outputs\n• corpus.txt\n• phrase_fingerprints.npz\n• doc_fingerprints.npz', 
         colors['output'], 9)

# === PHASE 2: QUERY PROCESSING ===
ax.text(10, 7.2, 'PHASE 2: QUERY PROCESSING', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#CC6600')

# Step 7
draw_box(6, 5.5, 4, 1.2, 'Step 7: Query\nProcessing', colors['step7'], 11)
ax.text(8, 5.0, '• Query extraction\n• Fingerprint construction\n• Spreading activation', 
        ha='center', va='center', fontsize=7, color='#333333')

# SPLADE
draw_box(10.5, 5.5, 3.5, 1.2, 'SPLADE\nScoring', colors['fusion'], 11)
ax.text(12.25, 5.0, '• Dense retrieval\n• Lexical matching\n• Neural scoring', 
        ha='center', va='center', fontsize=7, color='#333333')

# Fusion
draw_box(14.5, 5.5, 3, 1.2, 'Fusion\n(Linear/RRF)', colors['fusion'], 11)
ax.text(16, 5.0, '• Score combination\n• Weighted merging\n• Rank fusion', 
        ha='center', va='center', fontsize=7, color='#333333')

# Arrows for Phase 2
draw_arrow(2, 7.5, 6, 6.1)
draw_arrow(10, 6.1, 10.5, 6.1)
draw_arrow(14, 6.1, 14.5, 6.1)

# === OUTPUTS ===
draw_box(18, 5.5, 2, 1.2, 'Ranked\nResults', colors['output'], 11)

# Arrow to output
draw_arrow(17.5, 6.1, 18, 6.1)

# === BENCHMARK ===
ax.text(10, 4.2, 'PHASE 3: BENCHMARK', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#CC0000')

draw_box(0.5, 2.5, 4, 1.2, 'Benchmark\nEvaluation', colors['benchmark'], 11)
ax.text(2.5, 2.0, '• MRR, AP, P@K\n• NDCG, Recall\n• Speed metrics', 
        ha='center', va='center', fontsize=7, color='#333333')

draw_box(5, 2.5, 4, 1.2, 'BM25\nBaseline', colors['benchmark'], 11)
ax.text(7, 2.0, '• Lexical matching\n• TF-IDF scoring\n• Standard IR', 
        ha='center', va='center', fontsize=7, color='#333333')

draw_box(9.5, 2.5, 4, 1.2, 'Comparison\nAnalysis', colors['benchmark'], 11)
ax.text(11.5, 2.0, '• SF vs BM25\n• Speed analysis\n• Quality metrics', 
        ha='center', va='center', fontsize=7, color='#333333')

# Arrows for Benchmark
draw_arrow(18, 5.5, 11.5, 3.7)
draw_arrow(2, 7.5, 2.5, 3.7)
draw_arrow(4.5, 3.1, 5, 3.1)
draw_arrow(9, 3.1, 9.5, 3.1)

# === FINAL OUTPUT ===
draw_box(14, 2.5, 5, 1.2, 'Final Results\n• Comparison Tables\n• Charts\n• Reports', 
         colors['output'], 11)

draw_arrow(13.5, 3.1, 14, 3.1)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=colors['input'], label='Input'),
    mpatches.Patch(facecolor=colors['step1'], label='Steps 1-5 (Indexing)'),
    mpatches.Patch(facecolor=colors['step7'], label='Step 7 (Query)'),
    mpatches.Patch(facecolor=colors['fusion'], label='SPLADE/Fusion'),
    mpatches.Patch(facecolor=colors['benchmark'], label='Benchmark'),
    mpatches.Patch(facecolor=colors['output'], label='Output'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9)

plt.tight_layout()
plt.savefig('D:/darsi/ms/Thesis/Dr.Banaie/code050302/SemanticFolding/outputs/pipeline_block_diagram.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("Block diagram saved to outputs/pipeline_block_diagram.png")
