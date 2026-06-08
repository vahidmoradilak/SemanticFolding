# Retrieval Metrics & Evaluation Framework for Semantic Folding

**Module:** `query_processor.py` (Step 6) · `tools/compute_ir_metrics.py`  
**Version:** 3.1

---

## 1. Introduction: Semantic Folding in Context

The Semantic Folding framework represents words, phrases, and documents as **Sparse Distributed Representations (SDRs)** over a fixed 2-D semantic grid. The pipeline proceeds through six stages:

1. **Phrase Extraction** — domain vocabulary is built from raw text via noun-chunk parsing and n-gram discovery.
2. **Term-Context Matrix** — a sparse co-occurrence matrix of phrases × contexts is constructed, optionally weighted by TF-IDF.
3. **Semantic Space** — contexts are embedded onto a $g \times g$ integer grid via dimensionality reduction (t-SNE, UMAP, or PCA), producing 2-D coordinates for every context.
4. **Phrase Fingerprints** — each phrase is assigned a centroid on the grid (from its co-occurring contexts), convolved with a 2-D Gaussian kernel ($\sigma$), and linearised into a 1-D vector of length $g^2$ via Morton (Z-order) encoding.
5. **Document Fingerprints** — phrase fingerprints of matched terms are accumulated onto a reconstructed 2-D grid using the inverse Morton mapping, then sparsified via topology-preserving peak selection to a target density $\rho$.
6. **Query Processing** — a query is decomposed into phrases, weighted, converted to an SDR, optionally spread (topological dilation), and compared against all document SDRs via an asymmetric similarity score.

The output of Step 6 is a ranked list of documents for each query, with associated scores. Evaluating the quality of this ranking **requires a suite of metrics** that capture different facets of retrieval performance. This document defines each metric, explains its mathematical formulation, and guides interpretation in the context of SDR-based semantic retrieval.

---

## 2. Scoring Formula

### 2.1 Similarity Between Query and Document

Given a query fingerprint $\mathbf{q} \in \mathbb{R}^{g^2}$ (float-valued, IDF-weighted, L2-normalised) and a document fingerprint $\mathbf{d}_i \in \mathbb{R}^{g^2}$ (binary, sparsified), the score is:

\[
\text{score}(Q, D_i) = \frac{ \mathbf{q} \cdot \mathbf{d}_i^\mathsf{T} }{ \|\mathbf{q}\|_2 \cdot \sqrt{ \|\mathbf{d}_i\|_0 } }
\tag{1}
\]

where:
- $\|\mathbf{q}\|_2 = \sqrt{\sum_j q_j^2}$ is the L2 norm of the query fingerprint.
- $\|\mathbf{d}_i\|_0 = \text{nnz}(\mathbf{d}_i)$ is the number of non-zero (active) bits in the document fingerprint.
- The denominator uses $\sqrt{\|\mathbf{d}_i\|_0}$ rather than $\|\mathbf{d}_i\|_2$ because document vectors are **binary**; for a binary vector, $\|\mathbf{d}_i\|_2 = \sqrt{\|\mathbf{d}_i\|_0}$.

After L2 normalisation of the query, $\|\mathbf{q}\|_2 = 1.0$, simplifying to:

\[
\text{score}(Q, D_i) = \frac{ \text{raw\_dot} }{ \sqrt{ \text{nnz}(\mathbf{d}_i) } }
\tag{2}
\]

**Implementation:** `query_processor.py:1539-1541`

```python
raw_dot = float(query_fp.dot(doc_fp.T).toarray()[0, 0])
score   = raw_dot / (query_norm * np.sqrt(doc_fp.nnz))
```

### 2.2 Normalization Diagnostic

Because each document has a distinct nnz (active bit count), the normalisation denominator varies per document. A **fixed-denominator bug** would occur if all documents used the same nnz value — this can be detected by checking the uniqueness of per-document nnz:

\[
\text{Normalization healthy} \iff |\{\text{nnz}(\mathbf{d}_i) \mid i = 1,\dots,D\}| = D
\]

The diagnostic in the evaluation report (`qa_evaluation_report.md`, Section 4) computes:

```
Unique doc_nnz values: [338, 357, 378, 556, 597, ...]
[OK] Healthy: 20 distinct doc_nnz values found
```

---

## 3. Query Construction Metrics

### 3.1 Query Fingerprint Sparsity

Let $\mathbf{q} \in \mathbb{R}^{g^2}$ be the query fingerprint before normalisation:

\[
\text{sparsity}(Q) = \frac{ \|\mathbf{q}\|_0 }{ g^2 } \times 100\%
\tag{3}
\]

**Importance:** Sparsity measures how much of the semantic grid the query activates. A very sparse query (< 1% active) may be too specific and miss relevant documents; a dense query (> 50% active) may be too generic and return many false positives. Typical values for well-formed queries on a $128 \times 128$ grid are 8–16%.

### 3.2 Spreading Gain

Topological bit spreading (radius $r=1$, decay $\gamma=0.5$) dilates the query fingerprint to neighbouring Z-order cells:

\[
\text{spreading\_gain} = \frac{ \|\tilde{\mathbf{q}}\|_0 - \|\mathbf{q}\|_0 }{ \|\mathbf{q}\|_0 } \times 100\%
\tag{4}
\]

where $\tilde{\mathbf{q}}$ is the spread fingerprint. A consistent gain of ~32% (as observed across all test queries) indicates the spreading mechanism is operating correctly and uniformly.

### 3.3 Phrase Coverage

\[
\text{phrase\_coverage} = \frac{ \text{num\_matched} }{ \text{num\_phrases} } \times 100\%
\tag{5}
\]

Measures what fraction of the query's extracted phrases exist in the vocabulary. Low coverage (< 50%) indicates a vocabulary mismatch — the corpus lacks terms needed to represent the query.

### 3.4 OOV Expansion Statistics

For out-of-vocabulary terms, the system expands the query by finding semantically similar in-vocabulary phrases via fingerprint cosine similarity:

\[
\text{OOV expansion count} = \sum_{\text{OOV term } t} |\text{matches}(t)|
\tag{6}
\]

Each match is scored by cosine similarity $\text{sim}(t, v) \in [0,1]$ and weighted by $w = \text{sim}^2$. Excessive OOV expansion (> 20 terms) can introduce noise and degrade precision.

---

## 4. Retrieval Metrics

### 4.1 Precision@K (P@K)

\[
P@K = \frac{ |\mathcal{R} \cap \text{top-}K| }{ K }
\tag{7}
\]

where $\mathcal{R}$ is the set of relevant documents (ground truth) and $\text{top-}K$ is the set of the $K$ highest-scoring retrieved documents.

**Interpretation:** The fraction of top-$K$ results that are relevant. High P@K means the system places relevant documents at the very top of the ranking.

**Importance for PhD:** Precision measures the **purity** of the retrieval. In closed-domain QA, a high P@5 (> 0.600) indicates that the SDR representation concentrates semantic signal in the correct grid regions.

### 4.2 Recall@K (R@K)

\[
R@K = \frac{ |\mathcal{R} \cap \text{top-}K| }{ |\mathcal{R}| }
\tag{8}
\]

**Interpretation:** The fraction of all relevant documents that appear in the top-$K$. High R@K means the system does not miss relevant documents.

**Importance for PhD:** Recall measures the **completeness** of the retrieval. Perfect R@5 (1.000) on the test corpus means the SDR neighbourhood of every query correctly encompasses all expected relevant documents.

### 4.3 Mean Reciprocal Rank (MRR)

\[
\text{MRR} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{1}{\text{rank}_q}
\tag{9}
\]

where $\text{rank}_q$ is the rank position of the **first** relevant document for query $q$.

**Interpretation:** MRR rewards systems that place the first relevant result near the top. An MRR of 1.000 means every query has a relevant document at rank 1.

**Importance for PhD:** MRR is the standard metric for question answering systems where the user typically examines only the top result. It captures the user's "first look" experience.

### 4.4 Normalized Discounted Cumulative Gain (NDCG@K)

\[
\text{DCG@K} = \sum_{i=1}^{K} \frac{ \text{rel}_i }{ \log_2(i+1) }
\tag{10}
\]

\[
\text{IDCG@K} = \sum_{i=1}^{K} \frac{ \text{rel}_i^{\text{ideal}} }{ \log_2(i+1) }
\tag{11}
\]

\[
\text{NDCG@K} = \frac{ \text{DCG@K} }{ \text{IDCG@K} }
\tag{12}
\]

where $\text{rel}_i \in \{0, 1\}$ is the binary relevance of the document at rank $i$, and $\text{rel}_i^{\text{ideal}}$ is the relevance in the ideal ranking (all relevant documents first, in any order).

**Interpretation:** NDCG measures the **ranking quality** against the optimal possible ranking. A value of 1.000 means the top-$K$ results are perfectly ordered.

**Importance for PhD:** NDCG is more informative than P@K because it penalises relevant documents appearing at lower ranks. It is the preferred metric for evaluating ranked retrieval in information retrieval research.

### 4.5 Average Precision (AP) and Mean Average Precision (MAP)

\[
\text{AP}(q) = \frac{1}{|\mathcal{R}_q|} \sum_{i=1}^{N} P@i \cdot \mathbb{1}[\text{doc}_i \in \mathcal{R}_q]
\tag{13}
\]

where $\mathbb{1}[\cdot]$ is the indicator function. MAP is the mean of AP across all queries:

\[
\text{MAP} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{AP}(q)
\tag{14}
\]

**Interpretation:** AP summarises the precision-recall curve as a single value. It rewards systems that rank relevant documents higher.

**Importance for PhD:** MAP is the standard for evaluating ranked retrieval in TREC-style evaluations. It is sensitive to the entire ranking, not just the top-$K$.

---

## 5. Score Distribution Metrics

### 5.1 Top Score / Mean Score Ratio

\[
\text{score\_ratio} = \frac{ \max_i \text{score}_i }{ \frac{1}{D} \sum_i \text{score}_i }
\tag{15}
\]

**Interpretation:** A large ratio (> 5.0) indicates one document strongly dominates the ranking. A ratio near 1.0 means all documents are scored similarly (poor discrimination).

### 5.2 Score Gap Between Ranks

\[
\Delta_{i \to i+1} = \text{score}_i - \text{score}_{i+1}
\tag{16}
\]

**Interpretation:** Large gaps between top ranks indicate confident discrimination. Small or negative gaps suggest the system is uncertain between alternatives.

### 5.3 Documents Above Threshold

\[
D_{\text{above}} = \sum_{i=1}^{D} \mathbb{1}[\text{score}_i \geq \tau]
\tag{17}
\]

where $\tau$ is the minimum similarity threshold (default 0.0).

**Interpretation:** When $D_{\text{above}} = D$ (all documents scored above threshold), the threshold is not filtering any results. A stricter threshold may improve precision at the cost of recall.

---

## 6. Expected vs. Actual Ranking Analysis

### 6.1 Precision and Recall at the Set Level

For evaluating whether the expected relevant set appears in the top-$K$:

\[
\text{Top-}K\ \text{Match Count} = |\mathcal{R}_{\text{expected}} \cap \text{top-}K|
\tag{18}
\]

\[
\text{Top-}K\ \text{Match Rate} = \frac{|\mathcal{R}_{\text{expected}} \cap \text{top-}K|}{|\mathcal{R}_{\text{expected}}|}
\tag{19}
\]

**Interpretation:** A match rate of 1.000 means all expected documents appear in the top-$K$. This is a binary sanity check — if the SDR representation is semantically faithful, expected documents should consistently appear near the top.

---

## 7. Summary of Required Metrics for PhD Evaluation

For a rigorous evaluation of the Semantic Folding retrieval system, report the following set of metrics:

| Category | Metric | Symbol | Range | Target | When to Report |
|---|---|---|---|---|---|
| **Scoring** | Score formula | Eq. (1) | — | Correct normalisation | Always |
| **Scoring** | Norm diagnostic | Eq. (2) | healthy / bug | Healthy | Always |
| **Query** | Query sparsity | Eq. (3) | [0%, 100%] | 8–16% | Per query |
| **Query** | Spreading gain | Eq. (4) | [0%, ∞) | ~32% | Per experiment |
| **Query** | Phrase coverage | Eq. (5) | [0%, 100%] | > 80% | Per query |
| **Query** | OOV expansion count | Eq. (6) | [0, ∞) | < 15 | Per query |
| **Retrieval** | Precision@5 | Eq. (7) | [0, 1] | > 0.500 | Per query + avg |
| **Retrieval** | Recall@5 | Eq. (8) | [0, 1] | > 0.800 | Per query + avg |
| **Retrieval** | MRR | Eq. (9) | [0, 1] | > 0.800 | Overall |
| **Retrieval** | NDCG@5 | Eq. (12) | [0, 1] | > 0.700 | Per query + avg |
| **Retrieval** | MAP | Eq. (14) | [0, 1] | > 0.600 | Overall |
| **Distribution** | Score ratio | Eq. (15) | [1, ∞) | > 2.0 | Per query |
| **Distribution** | Score gap | Eq. (16) | [0, ∞) | > 0.5 | Per query |
| **Validation** | Expected match rate | Eq. (19) | [0, 1] | 1.000 | Overall |

---

## 8. Implementation Notes

### 8.1 Computing Metrics from Pipeline Output

All metrics can be computed from the `query_results_all.json` file produced by `query_processor.py --query-file`. The JSON structure contains:

```json
{
  "query": "What role do sentence structure...",
  "results": [["16", 6.1678], ["18", 3.8230], ...],
  "metadata": {
    "query_construction": {
      "num_phrases": 17,
      "num_matched": 17,
      "active_bits": 2561,
      "sparsity": 0.1563,
      "phrase_weights": {"sentence": 1.897, ...}
    },
    "spreading": {
      "active_bits_before": 2561,
      "active_bits_after": 3390
    },
    "ranking": {
      "total_documents": 20,
      "documents_above_threshold": 20,
      "mean_similarity": 1.3400,
      "max_similarity": 6.1678
    }
  }
}
```

### 8.2 Ground Truth Definition

Ground truth relevance sets ($\mathcal{R}_q$) must be defined **before** running the evaluation. For each query, identify the document IDs that are topically relevant based on the corpus content. In the Semantic Folding pipeline, relevance is typically **binary**:
- $\text{rel}_i = 1$ if document $i$ covers the topic of query $q$.
- $\text{rel}_i = 0$ otherwise.

For graded relevance (e.g., highly relevant, partially relevant, irrelevant), NDCG supports graded relevance values directly.

### 8.3 Statistical Significance

When comparing two retrieval configurations (e.g., with and without spreading, different grid sizes), report:
- Mean and standard deviation of each metric across all queries.
- Pairwise comparisons using the **paired t-test** or **Wilcoxon signed-rank test** ($\alpha = 0.05$).

---

## 9. References

1. **Morton, G. M.** (1966). A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing. IBM Technical Report.
2. **Hawkins, J., Ahmad, S., & Cui, Y.** (2017). A Theory of How Columns in the Neocortex Enable Learning the Structure of the World. *Frontiers in Neural Circuits*, 11, 81.
3. **Lai, T., & Ahmad, S.** (2021). Sparse Distributed Representations for Semantic Folding. *Numenta Research Report*.
4. **Järvelin, K., & Kekäläinen, J.** (2002). Cumulated Gain-Based Evaluation of IR Techniques. *ACM TOIS*, 20(4), 422–446.
5. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval*. Cambridge University Press.
