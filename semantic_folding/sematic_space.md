# Semantic Space Construction: Technical and Academic Foundations

## Abstract

Semantic space construction formalises the intuition that linguistic meaning
is distributed across contexts of use. This document establishes the
mathematical, algorithmic, and implementational foundations of the
semantic-space pipeline step, which maps a high-dimensional sparse
term-context matrix onto a discrete two-dimensional integer grid. The
resulting spatial arrangement encodes distributional similarity as
geometric proximity and serves as the coordinate substrate for phrase
fingerprint generation. We derive the full pipeline from first principles,
situating each design decision within the relevant literature.

---

## 1. Theoretical Foundations

### 1.1 The Distributional Hypothesis

The distributional hypothesis, articulated by Harris (1954) and later
refined by Firth (1957), asserts that the meaning of a linguistic unit is
a function of the contexts in which it characteristically occurs:

> *"You shall know a word by the company it keeps."* — Firth (1957, p. 11)

Formally, let $\mathcal{V} = \{w_1, w_2, \ldots, w_n\}$ be a vocabulary of
$n$ phrase types and $\mathcal{C} = \{c_1, c_2, \ldots, c_m\}$ a set of $m$
context types. The distributional profile of a phrase $w_i$ is the vector:

$$\mathbf{v}_i = \bigl[f(w_i, c_1),\ f(w_i, c_2),\ \ldots,\ f(w_i, c_m)\bigr]
  \;\in\; \mathbb{R}^m$$

where $f(w_i, c_j)$ is a weighting of co-occurrence frequency. The
hypothesis implies that if two phrases $w_i$ and $w_k$ are semantically
similar, then $\mathbf{v}_i \approx \mathbf{v}_k$ under some appropriate
metric on $\mathbb{R}^m$.

### 1.2 The Term-Context Matrix

The full distributional structure of a corpus is captured by the
term-context matrix $\mathbf{M} \in \mathbb{R}^{n \times m}$ where entry
$M_{ij} = f(w_i, c_j)$. Raw co-occurrence counts are a poor weighting
because they are dominated by high-frequency function words. Two
alternatives are standard in the literature:

**Positive Pointwise Mutual Information (PPMI)**

$$\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w)\, P(c)}, \qquad
  \text{PPMI}(w, c) = \max\!\bigl(0,\, \text{PMI}(w, c)\bigr)$$

PPMI is the de facto standard for count-based distributional models
(Bullinaria & Levy, 2007; Levy & Goldberg, 2014) because it strongly
penalises coincidental co-occurrence without introducing negative values
that complicate downstream processing.

**Term Frequency–Inverse Document Frequency (TF-IDF)**

$$\text{TF-IDF}(w, c) = \text{tf}(w, c) \cdot \log \frac{N}{1 + \text{df}(w)}$$

where $N$ is the total number of contexts and $\text{df}(w)$ is the
document frequency of $w$. TF-IDF is appropriate when contexts are
document-sized windows or when sparse-context selection is preferable.

### 1.3 Distributional Semantics as Geometry

The term-context matrix $\mathbf{M}$ can be interpreted geometrically: each
context $c_j$ is a direction in phrase-space and each phrase $w_i$ is a
point. Alternatively — and crucially for this pipeline — each phrase $w_i$
is a direction in context-space and each **context** $c_j$ is a point in
$\mathbb{R}^n$. It is this transposed interpretation that motivates the
pipeline: contexts that consistently co-occur with the same phrases cluster
together in $\mathbb{R}^n$, and this high-dimensional clustering should be
preserved by the subsequent dimensionality reduction.

---

## 2. Vector Space Preprocessing

### 2.1 Transpose Orientation

The matrix loaded from disk has shape $(n, m)$ — phrases as rows, contexts
as columns. The pipeline transposes it to obtain:

$$\mathbf{M}^\top \in \mathbb{R}^{m \times n}$$

so that row $j$ is the context vector $\mathbf{c}_j \in \mathbb{R}^n$,
expressing context $c_j$ in terms of the phrase weights that characterise
it. This is the input to dimensionality reduction.

### 2.2 L2 Normalisation

Each context vector is normalised to unit length:

$$\hat{\mathbf{c}}_j = \frac{\mathbf{c}_j}{\|\mathbf{c}_j\|_2}$$

This normalisation serves two purposes:

1. It converts Euclidean dot products into cosine similarities, which is
   theoretically preferable for distributional vectors because it removes
   length bias (longer texts produce larger raw vectors):

$$\cos(\mathbf{c}_j, \mathbf{c}_k) = \hat{\mathbf{c}}_j \cdot \hat{\mathbf{c}}_k$$

2. It provides UMAP with a well-defined unit-sphere geometry when the
   ``cosine`` metric is selected, aligning the UMAP objective with the
   standard semantic similarity measure used throughout the literature
   (Turney & Pantel, 2010).

---

## 3. Dimensionality Reduction

### 3.1 The Curse of Dimensionality

The context vectors live in $\mathbb{R}^n$ where $n$ may be in the tens of
thousands. In high-dimensional spaces, the ratio of the maximum to minimum
pairwise distance approaches 1 as dimensionality grows (Beyer et al., 1999),
making neighbourhood relationships unstable and visualisation impossible.
Dimensionality reduction projects the data into $\mathbb{R}^2$ while
attempting to preserve the neighbourhood structure that encodes semantic
similarity.

### 3.2 t-SNE

t-distributed Stochastic Neighbour Embedding (van der Maaten & Hinton, 2008)
defines a probability distribution over pairs of high-dimensional points:

$$p_{j \mid i} = \frac{\exp\!\bigl(-\|\mathbf{c}_i - \mathbf{c}_j\|^2
  / 2\sigma_i^2\bigr)}
  {\sum_{k \neq i} \exp\!\bigl(-\|\mathbf{c}_i - \mathbf{c}_k\|^2
  / 2\sigma_i^2\bigr)}, \qquad
  p_{ij} = \frac{p_{j \mid i} + p_{i \mid j}}{2m}$$

and a corresponding distribution in the 2-D embedding using a Student-$t$
kernel with one degree of freedom:

$$q_{ij} = \frac{\bigl(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2\bigr)^{-1}}
  {\sum_{k \neq l}\bigl(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2\bigr)^{-1}}$$

The embedding is obtained by minimising the Kullback-Leibler divergence:

$$\mathcal{L}_{\text{t-SNE}} = \text{KL}(P \| Q)
  = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Perplexity.** The bandwidth $\sigma_i$ is determined by the
perplexity parameter, defined as:

$$\text{Perp}(P_i) = 2^{H(P_i)}, \qquad H(P_i) = -\sum_j p_{j \mid i}
\log_2 p_{j \mid i}$$

Perplexity can be interpreted as a smooth estimate of the number of
effective neighbours. For a corpus of $m$ contexts the perplexity is
automatically clamped to $\min(\text{perplexity},\ \max(5, \lfloor m/3 \rfloor))$
to maintain numerical stability.

**Properties for semantic space.** t-SNE excels at revealing local cluster
structure — semantically coherent context groups are compressed into tight
clusters separated by empty regions. However, inter-cluster distances are
not interpretable (van der Maaten & Hinton, 2008), and the method does not
scale gracefully beyond $\sim$10,000 points without approximations such as
Barnes-Hut (van der Maaten, 2014).

### 3.3 UMAP

Uniform Manifold Approximation and Projection (McInnes et al., 2018)
is grounded in Riemannian geometry and algebraic topology. It assumes the
data lies on a Riemannian manifold with locally approximately uniform
density and seeks a low-dimensional representation that preserves the fuzzy
topological structure of the data.

The UMAP objective combines an attractive force for connected pairs and a
repulsive force for unconnected pairs:

$$\mathcal{L}_{\text{UMAP}} = \sum_{(i,j) \in E} \Bigl[
  w_{ij} \log \frac{w_{ij}}{v_{ij}}
  + (1 - w_{ij}) \log \frac{1 - w_{ij}}{1 - v_{ij}}
\Bigr]$$

where $w_{ij}$ is the fuzzy membership strength in the high-dimensional
graph and $v_{ij}$ is the corresponding low-dimensional membership
computed from the embedding coordinates.

**$n$-neighbors.** Controls the size of the local neighbourhood used to
construct the high-dimensional graph. Larger values emphasise global
manifold topology; smaller values emphasise local structure.

**min_dist.** Controls how tightly points may cluster in the 2-D
embedding. Smaller values allow contexts to pack into tight clusters;
larger values produce a more diffuse, globally uniform layout.

**Properties for semantic space.** UMAP is significantly faster than
t-SNE for large corpora and better preserves global distances at the cost
of less dramatic local clustering. The cosine metric on L2-normalised
vectors aligns UMAP's neighbourhood construction with semantic similarity
theory (McInnes et al., 2018).

### 3.4 PCA

Principal Component Analysis (Pearson, 1901; Hotelling, 1933) finds the
orthogonal directions of maximum variance in the data:

$$\mathbf{P} = \underset{\mathbf{W}^\top \mathbf{W} = \mathbf{I}}{\arg\max}
  \; \|\mathbf{X}\mathbf{W}\|_F^2$$

The first two principal components $\mathbf{p}_1, \mathbf{p}_2$ define the
2-D projection $\mathbf{Y} = \mathbf{X}\mathbf{W}_{1:2}$.

For sparse context matrices the full eigenvector decomposition is
avoided; instead, Truncated SVD (Halko et al., 2011) is applied:

$$\mathbf{M}^\top \approx \mathbf{U}_{:,1:2}\, \boldsymbol{\Sigma}_{1:2}\,
  \mathbf{V}_{1:2,:}^\top$$

**Properties for semantic space.** PCA is deterministic and computationally
inexpensive but captures only linear structure. It is appropriate as a
baseline or for debugging when reproducibility and speed take priority over
embedding quality.

### 3.5 Method Selection Guidance

| Criterion | t-SNE | UMAP | PCA |
|-----------|-------|------|-----|
| Local cluster separation | Excellent | Good | Poor |
| Global distance preservation | Poor | Good | Good |
| Scalability ($m > 5{,}000$) | Poor | Excellent | Excellent |
| Determinism | No (seed-fixed) | No (seed-fixed) | Yes |
| Sparse input | No | Yes | Yes (TruncatedSVD) |
| Interpretability of axes | None | None | Variance-ranked |

---

## 4. Grid Quantisation

### 4.1 Motivation

The fingerprint construction step (``phrase_fingerprints.py``) requires
each context to occupy a unique integer cell in a bounded grid so that its
position can be encoded as a 1-D index via Morton coding. The continuous
2-D embedding must therefore be discretised onto an $N \times N$ integer
grid.

### 4.2 Linear Scaling

Let $\mathbf{y}_j = (x_j, y_j) \in \mathbb{R}^2$ be the continuous
embedding of context $c_j$. Define per-axis minimum and maximum:

$$x_{\min} = \min_j x_j, \quad x_{\max} = \max_j x_j, \quad
  y_{\min} = \min_j y_j, \quad y_{\max} = \max_j y_j$$

The normalised position in the unit square is:

$$\tilde{x}_j = \frac{x_j - x_{\min}}{x_{\max} - x_{\min} + \varepsilon},
\qquad
\tilde{y}_j = \frac{y_j - y_{\min}}{y_{\max} - y_{\min} + \varepsilon}$$

where $\varepsilon = 10^{-10}$ prevents division by zero. The integer grid
coordinate with optional border padding $p$ is:

$$g^x_j = \text{clip}\!\Bigl(\text{round}\bigl(\tilde{x}_j (N - 2p - 1) + p\bigr),
  \;0,\;N-1\Bigr)$$

$$g^y_j = \text{clip}\!\Bigl(\text{round}\bigl(\tilde{y}_j (N - 2p - 1) + p\bigr),
  \;0,\;N-1\Bigr)$$

This maps the continuous unit square uniformly onto the discrete region
$[p,\, N-p-1]^2$.

### 4.3 Collision Analysis

Define the collision set:

$$\mathcal{K} = \bigl\{(i, j) : i \neq j,\ g_i = g_j \bigr\}$$

The collision rate is:

$$\rho = 1 - \frac{|\{\,g_j : j = 1,\ldots,m\,\}|}{m}$$

For a uniformly random placement of $m$ points into $N^2$ cells the
expected collision rate follows from the Birthday Problem:

$$\mathbb{E}[\rho] \approx 1 - e^{-m(m-1)/(2N^2)}$$

This gives guidance for grid sizing: to achieve $\rho < 0.05$ with $m$
contexts one should choose $N > \sqrt{m(m-1)/(2\ln(1/(1-0.05)))} \approx
\sqrt{10m}$.

---

## 5. Collision Resolution

### 5.1 Problem Statement

After quantisation, the function $\mathbf{g} : \mathcal{C} \to [0, N-1]^2$
may not be injective — two or more contexts may share a cell. The
fingerprint construction step requires injectivity. Collision resolution
finds a perturbation $\mathbf{g}' : \mathcal{C} \to [0, N-1]^2$ that is
injective and minimises displacement from the original quantised positions.

### 5.2 Chebyshev Spiral Search

The implemented algorithm performs a greedy sequential scan. Contexts are
processed in their original index order. For each context $c_j$ at
quantised position $\mathbf{g}_j = (g^x_j, g^y_j)$:

1. If $(g^x_j, g^y_j)$ is unoccupied, claim it: $\mathbf{g}'_j = \mathbf{g}_j$.
2. Otherwise, enumerate shells of increasing Chebyshev radius $r = 1, 2,
   \ldots, r_{\max}$. The shell at radius $r$ contains all cells
   $(g^x_j + \delta x,\; g^y_j + \delta y)$ satisfying:

$$\max(|\delta x|, |\delta y|) = r, \quad
  0 \leq g^x_j + \delta x < N, \quad
  0 \leq g^y_j + \delta y < N$$

3. Claim the first unoccupied in-bounds cell found: $\mathbf{g}'_j$.
4. If no free cell exists within $r_{\max}$, context $c_j$ remains at
   $\mathbf{g}_j$ (shared cell) and a warning is emitted.

The Chebyshev distance $d_\infty(\mathbf{a}, \mathbf{b}) = \max(|a_x -
b_x|, |a_y - b_y|)$ is used rather than Euclidean distance because the
shell enumeration is exact with integer arithmetic and every cell in the
grid is reachable without irrational step sizes.

**Complexity.** For $m$ contexts and maximum radius $r_{\max}$ the
worst-case time is $O(m \cdot r_{\max}^2)$ and the space overhead is
$O(N^2)$ for the occupancy hash map.

**Optimality.** The greedy algorithm is not globally optimal — the order
in which contexts are processed influences which cells are displaced.
Global optimality would require solving an assignment problem of size
$m \times N^2$, which is computationally intractable for large $m$.
The greedy approach is justified because: (i) the collision rate after
quantisation is typically low ($\rho \ll 0.1$ for $N \geq \sqrt{10m}$);
(ii) the displacement magnitude is bounded by $r_{\max}$, which is small
relative to the grid size; and (iii) the relative semantic ordering is
preserved for the vast majority of contexts.

---

## 6. Morton Code Linearisation (Downstream)

Morton codes (Morton, 1966), also known as Z-order curves, are used in
the downstream ``phrase_fingerprints.py`` step — **not** in semantic space
construction — and are mentioned here to clarify the architectural
separation of concerns.

A Morton code interleaves the binary representations of a 2-D integer
coordinate $(x, y)$ to produce a 1-D index that preserves 2-D locality:

$$z = \mathcal{M}(x, y) = \sum_{k=0}^{b-1}
  \Bigl[ \bigl((x \gg k) \mathbin{\&} 1\bigr) \ll 2k \;+\;
         \bigl((y \gg k) \mathbin{\&} 1\bigr) \ll (2k+1) \Bigr]$$

where $b = \lceil \log_2 N \rceil$ is the bit width needed to represent
grid coordinates in $[0, N-1]$.

Morton linearisation is appropriate for fingerprint indexing because it
preserves the locality of the 2-D semantic arrangement in the 1-D
fingerprint vector: cells that are geometrically close in the grid tend to
have close 1-D indices, ensuring that the fingerprint Hamming distance
approximates the spatial semantic distance.

The decision to remove Morton code computation from ``semantic_space.py``
was taken because the semantic space step is responsible only for producing
the coordinate map; Morton encoding is a transformation that belongs to the
fingerprint representation layer.

---

## 7. Output Artefacts and Their Roles

### 7.1 `context_coordinates_continuous.csv`

Contains the raw floating-point 2-D coordinates $\{(x_j, y_j)\}_{j=1}^m$
produced by the dimensionality reducer. These preserve the full resolution
of the continuous embedding and are used for:

- Visual quality assessment of the embedding.
- Post-hoc analysis with external tools (e.g. R, Gephi).
- Debugging grid quantisation artefacts.

This file is not read by any downstream pipeline step.

### 7.2 `context_coordinates.csv`

Contains the integer grid positions $\{(g^x_j, g^y_j)\}_{j=1}^m$ after
collision resolution. Provided for human inspection only.

### 7.3 `context_coordinates.json`

The primary machine-readable artefact:

```json
{
  "context_0": {"x": 5,  "y": 12},
  "context_1": {"x": 14, "y": 3}
}
```

This JSON dictionary enables $O(1)$ lookup of any context's grid position
by key, eliminating the $O(m)$ CSV scan that would otherwise be required
at each lookup. It is the **only** coordinate file consumed by
``phrase_fingerprints.py``.

### 7.4 `coordinate_statistics.json`

Records summary statistics computed **after** all grid processing is
complete:

- `unique_positions` — number of injective grid assignments after resolution.
- `collision_rate` — fraction of contexts that required displacement, i.e.
  $\rho' = 1 - |\{g'_j\}|/m$ evaluated on the final resolved coordinates.
- Continuous embedding range (for sanity checking).

Computing statistics after finalisation guarantees that reported values
describe the actual outputs consumed by downstream steps, not an
intermediate state.

---

## 8. Complexity and Scalability

Let $n$ = number of phrases, $m$ = number of contexts.

| Step | Time Complexity | Space |
|------|-----------------|-------|
| Matrix transpose | $O(\text{nnz})$ | $O(\text{nnz})$ |
| L2 normalisation | $O(\text{nnz})$ | $O(m)$ |
| t-SNE (Barnes-Hut) | $O(m \log m)$ per iter | $O(m)$ |
| UMAP | $O(m^{1.14})$ approx. | $O(m)$ |
| PCA / TruncatedSVD | $O(\text{nnz} \cdot k)$ | $O(mk)$ |
| Grid quantisation | $O(m)$ | $O(m)$ |
| Collision resolution | $O(m \cdot r_{\max}^2)$ | $O(N^2)$ |
| JSON serialisation | $O(m)$ | $O(m)$ |

For a corpus with $m = 10{,}000$ contexts and grid size $N = 128$ the full
pipeline completes in approximately 2–5 minutes on a modern workstation
when UMAP is selected, and 10–30 minutes when t-SNE is selected (without
GPU acceleration).

---

## 9. Validity and Evaluation

### 9.1 Neighbourhood Preservation

The quality of the embedding can be assessed by the Trustworthiness metric
(Venna & Kaski, 2006):

$$T(k) = 1 - \frac{2}{mk(2m - 3k - 1)}
  \sum_{i=1}^{m} \sum_{j \in \mathcal{U}_k(i)} \bigl(r(i,j) - k\bigr)$$

where $r(i, j)$ is the rank of context $j$ in the high-dimensional
neighbourhood of context $i$, and $\mathcal{U}_k(i)$ is the set of
contexts that are in the $k$-nearest neighbourhood in the 2-D embedding
but not in the high-dimensional space. $T(k) = 1$ indicates perfect local
neighbourhood preservation; $T(k) < 0.9$ indicates significant distortion.

### 9.2 Grid Coverage

Grid coverage $\gamma$ is:

$$\gamma = \frac{|\{\,g'_j : j = 1,\ldots,m\,\}|}{N^2}$$

Coverage significantly below 0.5 suggests that the grid is over-sized
relative to the corpus and that the fingerprint bit-vector will be
dominated by zeros, reducing discriminability. Coverage approaching 1
suggests the grid is under-sized and collision resolution quality will
degrade.

### 9.3 Intrinsic Dimensionality

Before applying any reduction, the intrinsic dimensionality $d^*$ of the
context vectors can be estimated using the Two-Nearest-Neighbours estimator
(Facco et al., 2017):

$$d^* \approx \frac{\ln m}{\ln(\bar{r}_2 / \bar{r}_1)}$$

where $\bar{r}_1$ and $\bar{r}_2$ are mean first and second nearest-neighbour
distances. When $d^*$ is much larger than 2, the 2-D projection is a strong
approximation and the collision rate after quantisation is expected to be
higher than for intrinsically low-dimensional data.

---

## 10. References

- Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? *International Conference on Database Theory*, 217–235.
- Bullinaria, J. A., & Levy, J. P. (2007). Extracting semantic representations from word co-occurrence statistics. *Behavior Research Methods*, 39(3), 510–526.
- Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7, 12140.
- Firth, J. R. (1957). A synopsis of linguistic theory 1930–55. *Studies in Linguistic Analysis*, 1–32.
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217–288.
- Harris, Z. S. (1954). Distributional structure. *Word*, 10(2–3), 146–162.
- Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417–441.
- Levy, O., & Goldberg, Y. (2014). Neural word embedding as implicit matrix factorization. *Advances in Neural Information Processing Systems*, 27.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv:1802.03426*.
- Morton, G. M. (1966). *A computer oriented geodetic data base and a new technique in file sequencing*. IBM Technical Report.
- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559–572.
- Turney, P. D., & Pantel, P. (2010). From frequency to meaning: Vector space models of semantics. *Journal of Artificial Intelligence Research*, 37, 141–188.
- van der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. *Journal of Machine Learning Research*, 15(1), 3221–3245.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
- Venna, J., & Kaski, S. (2006). Visualizing gene interaction graphs with local multidimensional scaling. *European Symposium on Artificial Neural Networks*, 557–562.