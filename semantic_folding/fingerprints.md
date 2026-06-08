# Semantic Fingerprints: Word‑to‑Document SDR Encoding with Topology Preservation

**Modules:** `phrase_fingerprints.py` (Step 4) · `doc_fingerprints.py` (Step 5)
  
**Version:** 3.1 — Corrected Morton mapping, metadata propagation, and visualisation alignment

---

## 1. Introduction

The Semantic Folding framework requires that words, phrases, and documents be represented as high‑dimensional sparse binary vectors (Sparse Distributed Representations, SDRs) that preserve the metric of the underlying semantic space. The present chapter describes Steps 4 and 5 of the pipeline, which translate the grid coordinates produced by the semantic space mapping (Step 3) into fixed‑length, topologically faithful SDRs for every phrase, and subsequently aggregate phrase‑level fingerprints into document‑level representations. The key challenge addressed here is two‑fold: (i) constructing phrase SDRs that maintain the spatial locality of the 2‑D semantic grid after linearisation, and (ii) combining hundreds or thousands of phrase vectors into a single document SDR without destroying the semantic topology.

We first present the phrase fingerprinting algorithm (Step 4), which uses centroid‑based placement, 2‑D Gaussian smoothing, and Morton (Z‑order) encoding to produce locality‑preserving vectors. Then we detail the document fingerprinting algorithm (Step 5), which accumulates phrase contributions on a properly reconstructed 2‑D grid and employs a topology‑preserving sparsifier to retain only the most semantically salient activations. A critical correction — the use of an exact inverse Morton mapping rather than naïve reshape — is proved to guarantee that the reconstructed document grid is identical to the weighted sum of the original phrase grids. The chapter concludes with complexity analysis, parameter guidance, and an integration roadmap for query processing (Step 6).

---

## 2. Phrase Fingerprinting (Step 4)

### 2.1 Algorithm Overview

The input to Step 4 is a set of contexts whose semantic coordinates $(x,y)$ on a square grid of side length $g$ have been determined by Step 3. For each phrase $p$, let $\mathcal{C}_p$ be the collection of contexts in which $p$ occurs. The phrase fingerprint $\mathbf{f}_p \in \mathbb{R}^{g^2}$ is constructed in three stages:

1. **Centroid computation**: the 2‑D centroid of the context coordinates is computed and snapped to the nearest integer cell.
2. **2‑D Gaussian smoothing**: a multi‑hot activation at the centroid cell is convolved with a 2‑D Gaussian kernel to create a soft receptive field.
3. **Morton linearisation**: the smoothed 2‑D grid is scanned in Z‑order to produce the final 1‑D vector.

### 2.2 Centroid Placement

The unweighted centroid $(\bar{x}_p, \bar{y}_p)$ of the contexts is given by

\[
\bar{x}_p = \frac{1}{|\mathcal{C}_p|} \sum_{c \in \mathcal{C}_p} x_c, \qquad
\bar{y}_p = \frac{1}{|\mathcal{C}_p|} \sum_{c \in \mathcal{C}_p} y_c .
\]

The discrete grid position is obtained by rounding to the nearest integer:

\[
\hat{x}_p = \lfloor \bar{x}_p + 0.5 \rfloor, \qquad
\hat{y}_p = \lfloor \bar{y}_p + 0.5 \rfloor .
\]

This procedure ensures that phrases with distinct co‑occurrence patterns map to different cells, even if they share a dominant context.

### 2.3 2‑D Gaussian Smoothing

A null‑initialised grid $G \in \mathbb{R}^{g \times g}$ receives a unit activation at $(\hat{x}_p, \hat{y}_p)$:

\[
G_{ij} = \begin{cases}
1, & (i,j) = (\hat{y}_p, \hat{x}_p) \\
0, & \text{otherwise}.
\end{cases}
\]

No accumulation of multiple contexts is performed at this stage—the centroid captures the distributional central tendency. The grid is then smoothed with an isotropic 2‑D Gaussian kernel of radius $\sigma$:

\[
\tilde{G} = G * K_{\sigma}, \qquad
K_{\sigma}(u,v) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{u^2+v^2}{2\sigma^2}\right).
\]

The discrete convolution is implemented via `scipy.ndimage.gaussian_filter`, which approximates the continuous kernel on the Cartesian grid. The resulting smoothed grid $\tilde{G}$ contains a central peak that decays smoothly to zero, forming a circular activation region. This soft assignment allows neighbouring grid cells to carry partial semantic signal, making fingerprints robust to small shifts in coordinate placement.

### 2.4 Morton (Z‑order) Linearisation

The 2‑D grid $\tilde{G}$ is converted to a 1‑D vector $\mathbf{f}_p$ of length $g^2$ by traversing its cells in Morton order (Z‑order curve). Formally, the Morton index $z(x,y)$ is the interleaving of the binary representations of $x$ and $y$:

\[
z(x,y) = \sum_{k=0}^{b-1} 2^{2k} \,\text{bit}_k(x) + 2^{2k+1} \,\text{bit}_k(y),
\]

where $b = \log_2 g$ (assuming $g$ is a power of two). The fingerprint is then

\[
\mathbf{f}_p[\,z(x,y)\,] = \tilde{G}_{y,x}, \qquad \forall (x,y) \in [0,g)\times[0,g).
\]

Morton encoding preserves 2‑D spatial locality in the 1‑D index: cells that are close in Euclidean distance tend to have Morton indices that are close, with the distance bounded by $O(\max(|x_1-x_2|,|y_1-y_2|)^2)$ for cells within the same Z‑order quadrant.

### 2.5 Output and Metadata

The step writes three files:

- `phrase_fingerprints.npz`: a $(P \times g^2)$ float32 array, where $P$ is the number of phrases.
- `phrase_fingerprints_meta.json`: a JSON object containing the phrase‑to‑row index map, together with the boolean `use_morton` and integer `grid_size`. Example:
  ```json
  {
    "phrase_to_row": {"neural plasticity": 42, …},
    "use_morton": true,
    "grid_size": 64
  }
  ```
  The metadata fields are essential for downstream steps that must unflatten the vectors correctly.
- `phrase_fingerprints_stats.json`: run‑level statistics.

The overall time complexity of Step 4 is $O(P \cdot g^2 \cdot \sigma^2)$, dominated by the Gaussian convolutions. The space requirement is $O(P \cdot g^2)$ for the output matrix.

---

## 3. Document Fingerprinting (Step 5)

### 3.1 Motivation and Challenge

A document fingerprint is obtained by aggregating the phrase fingerprints of all vocabulary‑matched phrases it contains. The naïve approach — summing the 1‑D vectors directly — appears straightforward. However, when phrase fingerprints are Morton‑encoded, a direct 1‑D sum loses the 2‑D spatial structure, because the activation of each phrase is spread along the Z‑order curve, not in a row‑major arrangement. Moreover, even if the aggregation is performed correctly in 2‑D, the un‑sparsified result is far too dense (often 40–60% active cells) to serve as a useful SDR. Therefore, the algorithm must (a) **correctly reconstruct the 2‑D semantic grid** from the Morton‑encoded phrase vectors, and (b) **sparsify the grid while preserving the semantic hotspots** that emerge from overlapping phrase activations.

### 3.2 Consistency Contract

Every document is subject to the identical phrase extraction pipeline used in Step 1: raw text → normalisation → sub‑phrase expansion → vocabulary filter. Any deviation would produce mismatched phrase representations and invalid fingerprints. Additionally, Step 5 must respect the binary encoding flag (`use_morton`) inherited from Step 4. If Morton encoding was used, the inverse mapping must be applied exactly; a naïve reshape would scramble the activation pattern.

### 3.3 Phase 1: 2‑D Grid Reconstruction via Exact Inverse Morton Mapping

Let the phrase fingerprints be given as a set $\{ \mathbf{f}_p \}_{p=1}^P$, where each $\mathbf{f}_p \in \mathbb{R}^{g^2}$ was linearised via the Morton function $z$. The document grid $G_d \in \mathbb{R}^{g \times g}$ is constructed by adding the weighted contributions of all phrases found in document $d$.

**Definition 1 (Inverse Morton lookup table).**  
For a given grid size $g$, define the table $\mathbf{T} \in \mathbb{N}^{g^2 \times 2}$ as

\[
\mathbf{T}[i] = \big( y(i), x(i) \big) =
\begin{cases}
z^{-1}(i), & \text{if use\_morton} = \text{True},\\[4pt]
\big( \lfloor i/g \rfloor,\; i \bmod g \big), & \text{otherwise}.
\end{cases}
\]

Here $z^{-1}(i)$ returns the unique $(x,y)$ such that $z(x,y)=i$, which exists and is bijective when $g$ is a power of two. The table is computed once during initialisation.

**Accumulation procedure.**  
For each phrase $p$ detected in $d$, let $w_{p,d} = \text{tf}(p,d) \cdot \text{idf}(p)$ be its TF‑IDF weight. The 1‑D fingerprint $\mathbf{f}_p$ is scattered into the 2‑D grid using advanced indexing:

\[
G_d\big[ \,\mathbf{T}[:,0],\; \mathbf{T}[:,1]\, \big] \;\mathrel{+}=\; w_{p,d} \cdot \mathbf{f}_p .
\tag{1}
\]

Equation (1) places every element $\mathbf{f}_p[i]$ into the cell $(y(i), x(i))$ that originally held that value before Morton flattening. This operation is vectorised, thus efficient, and works identically for both Morton and row‑major encodings.

**Theorem 1 (Exact Grid Reconstruction).**  
Let $\tilde{G}_p$ be the 2‑D smoothed grid of phrase $p$ produced by Step 4. After linearisation $\mathbf{f}_p[z(x,y)] = \tilde{G}_p[y,x]$ and subsequent accumulation via Equation (1), the document grid satisfies

\[
G_d = \sum_{p \in P(d)} w_{p,d} \cdot \tilde{G}_p ,
\]

where $P(d)$ is the multiset of matched phrases in document $d$. No spatial distortion is introduced.

*Proof.*  
For each phrase $p$, the table $\mathbf{T}$ provides the exact inverse of the encoding function: $z^{-1}(i) = (x,y)$ implies $\mathbf{T}[i] = (y,x)$. Substituting into (1) yields $G_d[y,x] \mathrel{+}= w_{p,d} \cdot \tilde{G}_p[y,x]$ for every $(x,y)$. Because the scatter is element‑wise and the grid is initially zero, the final $G_d$ is exactly the weighted sum of the original 2‑D grids. $\square$

The theorem guarantees that the document grid faithfully represents the spatial overlap of its constituent phrases. Semantic hotspots — regions where many semantically related phrases intersect — appear as high‑activation clusters in $G_d$.

### 3.4 Phase 2: Topology‑Preserving Sparsification

The raw document grid $G_d$ is typically dense, containing non‑zero values in 40–60% of its cells. To obtain a sparse SDR with a target density $\rho$ (e.g., 10%), we apply a multi‑step procedure that identifies coherent semantic hotspots and allocates the bit budget proportionally.

**Step 1: Light smoothing.**  
A mild Gaussian filter ($\sigma_{\text{smooth}} \approx 0.5\text{–}1.2$) is applied to merge weak, nearby activations without destroying the overall structure:

\[
\hat{G}_d = G_d * K_{\sigma_{\text{smooth}}} .
\]

**Step 2: Peak detection.**  
Local maxima are detected using a morphological maximum filter with neighbourhood radius $r = \text{min\_peak\_distance}$:

\[
\mathcal{P} = \{(x,y) \mid \hat{G}_d(x,y) = \max_{(u,v)\in N_r(x,y)} \hat{G}_d(u,v) \;\text{and}\; \hat{G}_d(x,y) > 0\}.
\]

Each peak $(x_i,y_i)$ represents a distinct semantic theme.

**Step 3: Proportional bit allocation.**  
Let $k = \lfloor \rho \cdot g^2 \rfloor$ be the total number of active bits permitted. Bits are distributed among the $m = |\mathcal{P}|$ peaks according to their relative strength:

\[
k_i = \max\!\left(1,\; \Big\lfloor k \cdot \frac{\hat{G}_d(x_i,y_i)}{\sum_{j=1}^m \hat{G}_d(x_j,y_j)} \Big\rfloor \right),
\]
with any remainder assigned to the strongest peak.

**Step 4: Local top‑$k$ selection.**  
For each peak $i$, a square window of side $2w_i+1$ is defined, where $w_i = \max(1, \lceil\sqrt{k_i/\pi}\rceil)$. Within this window, the $k_i$ cells with the highest activation in the *original* grid $G_d$ are activated. The use of the original unsmoothed grid prevents the smoothing from blurring the fine‑grained pattern inside each hotspot.

**Step 5: Fallback.**  
If $\mathcal{P}$ is empty (e.g., for very short documents), a global top‑$k$ selection on $G_d$ is used as a fallback.

The resulting sparse grid $G_d^{\text{sparse}}$ retains the high‑activation cores of the semantic hotspots while discarding background noise. The final 1‑D fingerprint is obtained by re‑applying the Morton linearisation to $G_d^{\text{sparse}}$, followed by optional L2 normalisation.

### 3.5 Properties of the Sparsified SDR

- **Sparsity guarantee**: $||\mathbf{f}_d||_0 \le k = \rho \cdot g^2$.
- **Topology preservation**: The active bits are clustered in regions of genuine semantic overlap. For two documents sharing a common theme, their fingerprints will exhibit high overlap in the corresponding grid region, resulting in a larger dot product (and thus similarity) compared to a uniform top‑$k$ sparsification that scatters bits randomly.
- **Locality under Morton encoding**: The Z‑order flattening again ensures that Hamming distance between document SDRs approximates the spatial proximity of their active cells.

### 3.6 Output and Metadata

The step writes:

- `doc_fingerprints.npz`: a $(D \times g^2)$ float32 matrix, where $D$ is the number of documents.
- `doc_fingerprints_meta.json`: a JSON object containing the document‑to‑row index map, `use_morton`, and `grid_size`.
- `doc_fingerprints_stats.json`: run statistics including achieved sparsity and diversity metrics.

The propagation of `use_morton` to the step output metadata ensures that Step 6 and any visualisation tools can correctly unflatten the document vectors when needed (e.g., for spreading or display).

### 3.7 Complexity Analysis

Let $D$ be the number of documents, $L$ the average document length, $P$ the vocabulary size, $g$ the grid size, and $\bar{k}$ the average number of matched phrases per document. The per‑document operations are:

- Phrase extraction (spaCy or NLTK): $O(D \cdot L)$.
- Grid reconstruction: $O(D \cdot \bar{k} \cdot g^2)$, since each matched phrase contributes its entire flattened vector (sparse).
- Smoothing and peak detection: $O(D \cdot g^2)$ per document.
- Local top‑$k$ selection: $O(D \cdot m \cdot w^2 \log w^2)$, where $m$ is the number of peaks and $w$ the average window radius. Typically $m \ll g^2$ and $w$ is small, rendering this term negligible.

The overall time complexity is dominated by the grid reconstruction. For a corpus of $10^4$ documents, a 64 × 64 grid, and an average of 50 matched phrases, the algorithm completes in minutes on modern hardware. Memory requirement is $O(D \cdot g^2)$ for the output matrix and $O(g^2)$ per‑document working space.

---

## 4. Integration with Query Processing (Step 6)

The document SDRs produced by Step 5 are designed to be compared with a query fingerprint using dot‑product similarity. Step 6 constructs a query SDR in exactly the same manner: phrases are extracted, weighted with TF‑IDF (or a chosen scheme), and their Morton‑encoded phrase fingerprints are accumulated. Because the encoding is identical and the query vector is also optionally spread in Z‑order space, the similarity score

\[
\text{sim}(q,d) = \frac{ \mathbf{f}_q \cdot \mathbf{f}_d^\mathsf{T} }{ \|\mathbf{f}_q\|_2 \cdot \sqrt{\|\mathbf{f}_d\|_0} }
\]

(an asymmetric cosine that penalises very dense document vectors) accurately reflects the semantic overlap.

Importantly, the `use_morton` flag stored in the document metadata allows Step 6 to correctly unflatten the query vector if spatial spreading is enabled. The spreading operation expands each active bit to its Z‑order neighbours, which requires knowledge of the 2‑D layout. With the flag correctly set, the spreading proceeds on the true grid, further improving recall for partially matching documents.

---

## 5. Correctness and Impact of the Morton Mapping Fix

In the initial deployment, Step 5 used a simple `vec.reshape(g, g)` to reinterpret the 1‑D phrase fingerprint as a 2‑D grid. For Morton‑encoded vectors this is geometrically meaningless: the resulting grid is a scrambled version of the true activation pattern. Consequently, the document grids appeared diffuse, no meaningful peaks were detected, and retrieval accuracy suffered. The fix — the introduction of the inverse Morton lookup table $\mathbf{T}$ — restores the exact spatial correspondence.

**Visual evidence** (not shown here) demonstrates that a corrected document grid exhibits clear, localised hotspots corresponding to the document’s dominant themes, whereas the scrambled grid is uniform and featureless. This improvement propagates to higher overlap among semantically related documents and more discriminative retrieval rankings.

---

## 6. Conclusion

We have presented a complete, mathematically grounded implementation of the semantic fingerprinting stage of the Semantic Folding pipeline. The design combines centroid‑based placement, Gaussian smoothing, Morton encoding, and topology‑preserving sparsification to yield SDRs that are sparse, high‑dimensional, and locality‑sensitive. The correction of the inverse Morton mapping guarantees that the 2‑D semantic structure is faithfully carried through to the final document representations. The resulting fingerprints form a robust foundation for semantic retrieval and analogical reasoning in closed‑domain question answering.

