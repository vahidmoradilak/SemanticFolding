# Query Processing in Semantic Folding: Architecture, Mathematical Formulation, and Algorithmic Design

## Abstract

This document provides a comprehensive technical description of the query processing module within a Semantic Folding pipeline developed for knowledge graph construction over academic corpora. The module transforms natural-language queries into sparse, distributed fingerprint representations over a two-dimensional semantic grid, applies IDF-weighted dot-product scoring against pre-indexed document fingerprints, and returns a ranked list of semantically relevant documents. The design integrates phrase extraction via spaCy, IDF-based term weighting, spatial spreading with exponential decay, query-side semantic expansion for vocabulary gap bridging, and an asymmetric weighted overlap scoring function. This document covers the theoretical foundations, algorithmic specification, and the critical architectural design decisions required to maintain topological distinctiveness in high-dimensional semantic spaces.

---

## 1. Introduction

Semantic Folding Theory (Purdy, 2016) proposes that the semantic content of natural language can be represented as sparse binary vectors — called *semantic fingerprints* — defined over a fixed, high-dimensional grid. Words and phrases that appear in similar contexts are assigned proximate positions on this grid, exploiting the spatial locality of semantic similarity. This approach draws on neuroscientific parallels with distributed cortical representations (Hawkins & George, 2006) and operationalizes the distributional hypothesis (Harris, 1954): linguistic units sharing contextual co-occurrence patterns occupy overlapping regions of the semantic space.

The query processing module described herein is the inference-time component of a multi-stage pipeline. Given a free-text query, it:

1. Extracts and normalizes constituent phrases.
2. Applies semantic expansion to bridge vocabulary gaps between query terms and the indexed phrase vocabulary.
3. Constructs a weighted query fingerprint by superimposing individual phrase fingerprints scaled by their IDF weights.
4. Applies topological bit spreading to generalize beyond exact grid positions, guarded by a minimum sparsity threshold.
5. Scores each document fingerprint against the query fingerprint using an asymmetric dot-product formulation.
6. Returns a ranked list of documents with associated relevance scores.

---

## 2. Theoretical Background

### 2.1 Semantic Fingerprints
A semantic fingerprint is formalized as a Sparse Distributed Representation (SDR) upon a two-dimensional grid of size $G \times G$. The total representational capacity of the space is given by $N = G^2$. Thus, any linguistic entity (a phrase, sentence, or document) is mapped to a vector $\mathbf{x} \in \{0, 1\}^N$ (or $\mathbb{R}^N$ for weighted formulations). 

The sparsity of a fingerprint, denoted as $S(\mathbf{x})$, is the ratio of active bits to the total capacity:
$$S(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} x_i$$
To function effectively, SDR theory requires that $S(\mathbf{x}) \ll 1$, allowing distinct concepts to occupy unique, orthogonal sub-spaces within the grid.

### 2.2 Grid Capacity and Dimensional Collapse
A critical architectural parameter is the grid resolution $G$. Early experimental formulations often test compact grids (e.g., $G=16$, yielding $N=256$ bits) for computational efficiency. However, aggregating multiple phrases into a single query or document fingerprint in such a constrained space rapidly induces *fingerprint saturation*. 

If multiple constituent phrases are aggregated via Boolean union or normalized summation, the sparsity $S(\mathbf{x})$ quickly approaches $1$. When a grid is fully saturated, the topological distinctiveness of the fingerprint is destroyed. All complex queries map to identical, fully active vectors, reducing the vector intersection $\mathbf{q} \cdot \mathbf{d}$ to a constant and rendering retrieval impossible. 

To prevent this "dimensional collapse," the architecture must scale to high dimensions. Utilizing a $G=128$ architecture ($128 \times 128$ grid, yielding $N=16384$ bits) provides the vast representational capacity required. In this high-dimensional space, a complex multi-phrase query can aggregate its constituent parts while maintaining a healthy sparsity (typically $S(\mathbf{x}) \approx 0.10$).

### 2.3 The Vocabulary Gap Problem
A fundamental challenge in fixed-vocabulary semantic systems is the *vocabulary gap* (Furnas et al., 1987): the mismatch between the terms users employ in queries and the terms present in the indexed vocabulary. In traditional information retrieval, this manifests as the *term mismatch problem* (Berger & Lafferty, 1999), where semantically equivalent concepts are expressed through different lexical forms.

In the Semantic Folding framework, this gap is particularly acute. Query terms not present in the pre-computed phrase fingerprint vocabulary contribute zero signal to the query fingerprint, regardless of their semantic relevance. For example, a query containing "dense population center" fails entirely if the vocabulary contains only "urban area" and "city," despite clear semantic overlap.

This problem is exacerbated in specialized domains where:
1. **Morphological variation** produces lexically distinct but semantically equivalent forms (e.g., "urbanization" vs. "urban development")
2. **Synonym proliferation** creates multiple valid expressions for identical concepts
3. **Compositional phrases** combine in-vocabulary terms into novel out-of-vocabulary (OOV) expressions

---

## 3. Query Fingerprint Construction

### 3.1 Phrase Extraction and Normalization
When a query string is submitted, it is processed through the same NLP pipeline used during document indexing. The query is tokenized, lemmatized, and normalized. Phrases are extracted and partitioned into two sets:
- $P_q^{IV}$: in-vocabulary phrases that directly match the fingerprint index
- $P_q^{OOV}$: out-of-vocabulary phrases requiring semantic expansion

### 3.2 Query-Side Semantic Expansion

To address the vocabulary gap, we introduce a *query-side semantic expansion* mechanism that operates within the fingerprint space itself. This approach is theoretically grounded in query expansion techniques from classical IR (Rocchio, 1971; Xu & Croft, 1996) but adapted to the topological constraints of Semantic Folding.

#### 3.2.1 Expansion via Fingerprint Similarity

For each OOV term $t \in P_q^{OOV}$, we construct a temporary fingerprint $\mathbf{f}_t$ using the same spatial hashing function employed during vocabulary construction. We then retrieve the $k$ nearest in-vocabulary phrases by computing cosine similarity in fingerprint space:

$$\text{sim}(\mathbf{f}_t, \mathbf{v}_j) = \frac{\mathbf{f}_t \cdot \mathbf{v}_j}{\|\mathbf{f}_t\|_2 \|\mathbf{v}_j\|_2}$$

where $\mathbf{v}_j$ is the fingerprint of in-vocabulary phrase $p_j$. The top-$k$ phrases satisfying $\text{sim}(\mathbf{f}_t, \mathbf{v}_j) \geq \theta$ (typically $\theta = 0.3$) are selected as semantic expansions.

This approach exploits the fundamental property of Semantic Folding: phrases with similar distributional semantics occupy proximate regions of the grid. By measuring fingerprint overlap, we effectively perform *distributional similarity matching* (Lin, 1998) without requiring external resources like WordNet or word embeddings.

#### 3.2.2 Expansion Weight Attenuation

Expanded terms are assigned attenuated weights to preserve the primacy of exact vocabulary matches and aggressively penalize marginal semantic relationships. If an OOV term $t$ expands to in-vocabulary phrase $p_j$ with similarity $s_j$, the expansion weight is computed using a squared similarity penalty:

$$w_j^{\text{exp}} = \alpha \cdot s_j^2 \cdot w_j^{\text{IDF}}$$

where $\alpha \in [0,1]$ is an attenuation factor (specifically set to $\alpha = 0.6$) and $w_j^{\text{IDF}}$ is the original IDF weight of $p_j$. This formulation ensures that:
1. Expanded terms contribute less overall than direct matches ($\alpha < 1$).
2. The non-linear squared penalty ($s_j^2$) sharply reduces the influence of weaker similarity matches while preserving the weight of high-confidence expansions.
3. Rare terms remain emphasized through the underlying IDF weighting.

This design parallels the *relevance feedback* framework (Rocchio, 1971), where expansion terms are weighted lower than original query terms to prevent semantic drift.

### 3.3 Fingerprint Aggregation

The final query fingerprint integrates both direct matches and semantic expansions. Let $P_q^{\text{merged}} = P_q^{IV} \cup \{\text{expansions of } P_q^{OOV}\}$ be the combined phrase set with associated weights $\{w_1, w_2, \dots, w_m\}$. The unspread query fingerprint $\mathbf{q}^{(0)} \in \mathbb{R}^N$ is computed as:

$$\mathbf{q}^{(0)} = \sum_{j=1}^{m} w_j \mathbf{v}_j$$

where $w_j$ is either the direct IDF weight (for $p_j \in P_q^{IV}$) or the attenuated expansion weight (for expanded terms). This vector is subsequently $L_2$-normalized to ensure consistent scoring scales.

---

## 4. Topological Bit Spreading

Relying solely on exact bit matches mirrors the brittleness of traditional keyword matching. Because the Semantic Folding algorithm guarantees that semantically similar concepts are placed in adjacent grid cells, we introduce **Topological Bit Spreading** to enhance recall.

### 4.1 The Mechanism of Spreading
Spreading applies a spatial filter to the active bits, activating neighboring dormant cells to create a "semantic halo." The grid is treated as a 2D matrix $\mathbf{Q} \in \mathbb{R}^{G \times G}$. For a given coordinate $(u, v)$, the spreading function activates neighboring cells $(x, y)$ within a radius $r$, applying an exponential decay factor $\gamma$.

The spread query matrix $\tilde{\mathbf{Q}}$ is computed as:
$$\tilde{Q}_{x,y} = \max_{u,v} \left( Q_{u,v} \cdot \gamma^{d((u,v), (x,y))} \right)$$
where $d$ is a spatial distance metric (e.g., Chebyshev distance) subject to $d \le r$. The resulting matrix is flattened back into the final query vector $\tilde{\mathbf{q}} \in \mathbb{R}^N$.

### 4.2 Sparsity Guard
To ensure the query possesses sufficient semantic substance before initiating expensive retrieval operations, a **Sparsity Guard** is enforced. The system asserts that the sparsity of the constructed query representation satisfies:
$$S(\mathbf{q}) \ge 0.005$$
Queries failing to meet this $0.5\%$ activation threshold lack sufficient semantic resolution, either due to extreme brevity or severe vocabulary mismatch, and are flagged to prevent anomalous retrieval results.

### 4.3 Synergy Between Expansion and Spreading

The semantic expansion and topological spreading mechanisms operate at complementary levels of abstraction:

- **Semantic expansion** addresses *lexical gaps* by substituting OOV terms with in-vocabulary synonyms, operating at the phrase level.
- **Topological spreading** addresses *positional variance* by creating spatial halos around active grid cells, operating at the bit level.

Together, these mechanisms implement a two-stage generalization strategy. Expansion ensures that semantically related but lexically distinct terms contribute to the query fingerprint. Spreading then allows these expanded terms to match document fingerprints even when their grid positions differ slightly due to training variance or context-dependent placement.

---

## 5. Scoring and Retrieval

Document retrieval is performed by comparing the final continuous query fingerprint $\tilde{\mathbf{q}} \in \mathbb{R}^N$ against the binary fingerprint $\mathbf{d}_i \in \{0, 1\}^N$ of each document $D_i$ in the corpus.

The similarity score is calculated as an asymmetric, weighted dot product, normalized by the $L_2$ norm of the query and the square root of the number of active bits (non-zero elements) in the document fingerprint:
$$\text{score}(Q, D_i) = \frac{\tilde{\mathbf{q}} \cdot \mathbf{d}_i}{\|\tilde{\mathbf{q}}\|_2 \sqrt{\text{nnz}(\mathbf{d}_i)}}$$

This dual normalization ensures that (a) baseline query length does not arbitrarily inflate scores, and (b) excessively long documents (which naturally have denser, more saturated fingerprints) are penalized, preventing them from dominating the retrieval results. The corpus is then sorted by this score in descending order, returning the highest-ranked documents above a predefined minimum similarity threshold.

---

## 6. Design Decisions and Trade-offs

### 6.1 Query-Side vs. Document-Side Expansion

The decision to implement semantic expansion exclusively on the query side (rather than expanding document fingerprints during indexing) reflects several architectural constraints:

1. **Computational Efficiency**: Expanding every document phrase during indexing would require $O(|D| \cdot |P_d|)$ computations. Query-side expansion requires only $O(|P_q|)$ computations per query.
2. **Index Stability**: Document-side expansion would require re-indexing the entire corpus whenever expansion parameters are tuned. Query-side expansion isolates parameter optimization from the core index.
3. **Semantic Drift Control**: Expanding documents risks introducing spurious matches and diluting the document's core semantic signature.

### 6.2 Asymmetric Scoring: Binary Documents vs. Real-Valued Queries
The decision to maintain binary document fingerprints while allowing real-valued query fingerprints is a deliberate architectural asymmetry. Regenerating document fingerprints with continuous IDF weights would significantly inflate storage overhead. By keeping documents as binary vectors $\mathbf{d} \in \{0,1\}^N$ and isolating the continuous weights in the query vector $\tilde{\mathbf{q}} \in \mathbb{R}^N$, the pipeline preserves strict modularity and storage efficiency.

### 6.3 Normalization Strategy
The normalization denominator $\sqrt{\text{nnz}(\mathbf{d})}$ acts as a soft, cosine-like length penalty. Alternative normalizations evaluated included:
- **No normalization**: Overwhelmingly favors long documents with broad topic coverage.
- **Full cosine normalization** $(\|\tilde{\mathbf{q}}\|_2 \cdot \|\mathbf{d}\|_2)^{-1}$: Over-penalizes length.
- **Linear normalization** $(\text{nnz}(\mathbf{d}))^{-1}$: Empirically under-performs by penalizing broad documents too aggressively.

The square-root penalty provides a pragmatic, theoretically sound balance, consistent with Okapi BM25's field-length normalization philosophy.

### 6.4 Spreading Radius Parameterization
The spreading parameters $r=1$, $\gamma=0.5$ were selected to optimize the signal-to-noise ratio. A radius of 1 provides limited spatial generalization without excessive noise injection. The $50\%$ decay ensures that spread bits contribute at most half the weight of a direct hit.

### 6.5 Expansion Parameter Selection

The expansion mechanism introduces three tunable parameters:

- **$k$ (expansion breadth)**: Number of nearest neighbors retrieved per OOV term. 
- **$\theta$ (similarity threshold)**: Minimum cosine similarity for expansion candidates (typically $\theta=0.3$).
- **$\alpha$ (attenuation factor) and Penalty ($s_j^2$)**: Setting $\alpha=0.6$ combined with the squared similarity penalty mathematically guarantees that only highly-correlated spatial expansions exert meaningful gravitational pull during the ranking phase, mitigating semantic drift.

---

## 7. Limitations and Future Work

**Expansion Quality Dependence on Vocabulary Coverage**: The effectiveness of semantic expansion is bounded by the quality and coverage of the in-vocabulary phrase set. If the vocabulary lacks semantically related terms for an OOV query phrase, expansion fails. 

**Computational Cost of Expansion**: Computing cosine similarity between an OOV term and all in-vocabulary phrases scales linearly with vocabulary size. Approximate nearest neighbor search (e.g., LSH, HNSW) could reduce complexity for massive vocabularies.

**Binary Document Representation**: Document fingerprints currently do not encode term frequency (TF). Documents containing a rare phrase once are indistinguishable from those containing it frequently.

**Evaluation Metrics**: Systematic evaluation requires the formal annotation of query-document relevance pairs to compute standard IR metrics (MAP, NDCG@10, P@5) and strictly quantify the precision/recall trade-offs of the spreading and expansion operators.

---

## 8. Conclusion

The query processing module presented here implements a principled, efficient approach to semantic retrieval based on Semantic Folding Theory. By utilizing a high-capacity semantic grid ($128 \times 128$), the architecture successfully avoids dimensional collapse, preserving the topological distinctiveness of complex queries.

The integration of query-side semantic expansion with a squared-similarity penalty ($s_j^2$) addresses the critical vocabulary gap problem, bridging lexical mismatches while mathematically suppressing semantic drift. This expansion mechanism, combined with IDF-weighted phrase aggregation, asymmetric dot-product scoring, and a foundational sparsity guard ($S(\mathbf{x}) \ge 0.005$), provides a robust retrieval framework capable of handling both lexical variation and positional variance in the semantic space.

---

## References

- Berger, A., & Lafferty, J. (1999). Information retrieval as statistical translation. *Proceedings of SIGIR*, 222–229.
- Furnas, G. W., Landauer, T. K., Gomez, L. M., & Dumais, S. T. (1987). The vocabulary problem in human-system communication. *Communications of the ACM*, 30(11), 964–971.
- Harris, Z. S. (1954). Distributional structure. *Word*, 10(2–3), 146–162.
- Hawkins, J., & George, D. (2006). *Hierarchical Temporal Memory: Concepts, Theory, and Terminology*. Numenta Technical Report.
- Lin, D. (1998). Automatic retrieval and clustering of similar words. *Proceedings of COLING-ACL*, 768–774.
- Purdy, S. (2016). Encoding data for HTM systems. *Frontiers in Neuroscience*, 10, 34.
- Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- Rocchio, J. J. (1971). Relevance feedback in information retrieval. In *The SMART Retrieval System* (pp. 313–323).
- Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.
- Turney, P. D., & Pantel, P. (2010). From frequency to meaning: Vector space models of semantics. *Journal of Artificial Intelligence Research*, 37, 141–188.
- Xu, J., & Croft, W. B. (1996). Query expansion using local and global document analysis. *Proceedings of SIGIR*, 4–11.
