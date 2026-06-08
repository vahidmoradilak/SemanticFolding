# Phrase & Document Fingerprinting — Technical Documentation

**Modules:** `phrase_fingerprints.py` (Step 4) · `doc_fingerprints.py` (Step 5)  
**Pipeline Stage:** 4 of 6 and 5 of 6 — Semantic Folding Pipeline  
**Version:** 2.0 (Revised & Corrected)

---

## Pipeline Overview

```
Step 1  phrase_extractor.py    → phrases.txt
Step 2  term_context.py        → phrase_context_matrix.*
Step 3  semantic_space.py      → context_coordinates.json
Step 4  phrase_fingerprints.py → phrase_fingerprints/          ← §1 THIS DOC
Step 5  doc_fingerprints.py    → doc_fingerprints/             ← §2 THIS DOC
Step 6  query_processing.py    → query results
```
Both steps share a critical design contract: **every phrase extracted in Step 1 must traverse the identical normalisation and expansion path in Steps 4 and 5**. Any deviation produces silently incorrect fingerprints.

---

# Part 1 — `phrase_fingerprints.py` (Step 4)

## Overview

Step 4 converts the semantic grid positions produced by `semantic_space.py` into fixed-length binary fingerprint vectors. Each phrase is assigned a Sparse Distributed Representation (SDR) whose active bits encode the phrase's location on the 2-D semantic grid. These vectors serve as the atomic building blocks from which document-level SDRs are assembled in Step 5.

The module implements three configurable encoding strategies:

1. **Morton (Z-order) linearisation** — maps 2-D coordinates to a 1-D index that preserves spatial locality under Hamming distance (default).
2. **Row-major linearisation** — simple $y \cdot g + x$ mapping without locality guarantees.
3. **Gaussian smoothing** — convolves the binary fingerprint with a 1-D Gaussian kernel so neighbouring grid cells receive fractional signal.

---

## Theoretical Motivation

### Why Sparse Distributed Representations?

Semantic Folding theory (Kanerva, 1988; Numenta HTM framework) requires that semantic units be encoded as **sparse, high-dimensional binary vectors**. Two key properties make SDRs suitable for semantic similarity computation:

**Sparsity:** Only a small fraction $\rho$ of bits are active:

$$\rho = \frac{k}{n} \ll 1, \quad n = g^2, \quad k \approx \text{top\_percent} \cdot n$$

This ensures that the probability of random overlap is negligible:

$$P(\text{random match}) = \binom{n}{k}^{-1} \binom{n/2}{k/2}^2 \approx 0 \text{ for large } n$$

**Locality:** When Morton encoding is used, the Hamming distance between two fingerprints approximates their semantic distance on the grid:

$$d_H(\mathbf{f}_i, \mathbf{f}_j) \approx f(d_{\text{grid}}(p_i, p_j))$$

where $d_{\text{grid}}$ is the Euclidean distance between their grid positions and $f$ is a monotone increasing function.

### Why Morton (Z-order) Encoding?

Row-major linearisation maps the 2-D grid to 1-D by row: $i = y \cdot g + x$. While simple, this breaks 2-D locality: cells $(0, g-1)$ and $(1, 0)$ are spatially adjacent but their indices differ by $2g - 1$.

Morton encoding interleaves the bits of $x$ and $y$:

$$z(x, y) = \ldots b_3^y b_3^x b_2^y b_2^x b_1^y b_1^x b_0^y b_0^x$$

This guarantees that spatially close cells remain index-close along the Z-order curve, preserving 2-D locality in 1-D for cells within the same $2^k \times 2^k$ quadrant. The grid size should be a **power of two** for the Morton mapping to be bijective over $[0, g^2 - 1]$.

---

## Inputs

### `context_coordinates.json`
Primary output of `semantic_space.py`. Maps every `context_id` to its finalised integer grid position after spiral-search collision resolution:
```json
json
{
    "context_0": {"x": 5,  "y": 12},
    "context_1": {"x": 14, "y": 3}
}
```
> **Important:** The JSON file is the canonical machine-readable format. The companion `context_coordinates.csv` is for human inspection only and must **not** be used by downstream automation.

### `term_context_matrix.json`
Produced by `term_context.py` (Step 2). Contains the phrase list and frequency counts:

```json
{
    "phrases": ["neural plasticity", "synaptic pruning", ...],
    "phrase_frequencies": [42, 17, ...],
    "num_contexts": 500,
    "matrix_shape": [2000, 500]
}
```
---

## Outputs

### `phrase_fingerprints.npz`
Compressed NumPy archive. Contains one key `"fingerprints"` with shape `(n_phrases, g²)`, `dtype float32`. Row $i$ is the SDR for the phrase at index $i$ in the metadata map.

### `phrase_fingerprints_meta.json`
Maps each phrase token string to its row index in the fingerprint matrix:

```json
{
    "neural plasticity": 42,
    "synaptic pruning":  43
}
```
---

## Core Algorithm

### Centroid-Based Grid Position

A phrase does not belong to a single context — it co-occurs across many. Its grid position is computed as the **unweighted centroid** of all context positions:

$$\bar{x} = \frac{1}{|C_p|} \sum_{c \in C_p} x_c, \qquad \bar{y} = \frac{1}{|C_p|} \sum_{c \in C_p} y_c$$

The real-valued centroid is then snapped to the nearest integer grid cell:

$$\hat{x} = \text{round}(\bar{x}), \quad \hat{y} = \text{round}(\bar{y})$$

> **Design Note (Corrected):** The original implementation used a single context lookup per phrase, which produced identical fingerprints for many phrases that co-occurred in the same dominant context. The centroid approach is the corrected implementation — it uses the distributional evidence across *all* contexts, ensuring that phrases with different co-occurrence patterns receive distinct spatial positions even if they share a most-frequent context.

### Linearisation and Hot-Bit Placement

Given grid position $(\hat{x}, \hat{y})$ and grid size $g$:

**Morton encoding (default):**

$$i_{\text{morton}} = \text{spread\_bits}(\hat{x}) \;\big|\; (\text{spread\_bits}(\hat{y}) \ll 1)$$

where `spread_bits` inserts a zero between every pair of adjacent bits:

```python
def _spread_bits(value: int) -> int:
    value &= 0x0000FFFF
    value = (value | (value << 8))  & 0x00FF00FF
    value = (value | (value << 4))  & 0x0F0F0F0F
    value = (value | (value << 2))  & 0x33333333
    value = (value | (value << 1))  & 0x55555555
    return value
```
**Row-major encoding (`--no-morton`):**

$$i_{\text{row}} = \hat{y} \cdot g + \hat{x}$$

The fingerprint vector $\mathbf{f} \in \{0,1\}^{g^2}$ is then:

$$f_i = \begin{cases} 1.0 & i = i_{\text{morton or row}} \\ 0.0 & \text{otherwise} \end{cases}$$

### Gaussian Smoothing (`--smooth`)

When enabled, the 1-D fingerprint is convolved with a Gaussian kernel of standard deviation $\sigma$ (default: 1.5):

$$\tilde{\mathbf{f}} = \mathbf{f} * G_\sigma$$

where $G_\sigma$ is the discrete Gaussian kernel implemented via `scipy.ndimage.gaussian_filter1d`. The smoothing is applied **after** all hot bits are placed, so the result integrates signal from every context position before normalisation.

**Effect:** A phrase at grid position $(8, 8)$ with Morton index $i$ will activate not only bit $i$ but also neighbouring bits in proportion to their Z-order distance, creating a soft receptive field around the phrase's semantic locus.

**When to use:** Smoothing is recommended when the corpus is small (fewer than 500 contexts) or when phrases have sparse co-occurrence patterns. For large, dense corpora, binary fingerprints with Morton encoding provide sufficient locality.

---

## `build_fingerprint` — Corrected Implementation

```python
def build_fingerprint(
    phrase_text : str,
    freq        : int,
    coordinates : Dict[str, Tuple[int, int]],
    grid_size   : int,
    use_morton  : bool,
) -> np.ndarray:
    """
    Build a single raw binary fingerprint for one phrase.
    Grid position = unweighted centroid of ALL context coordinates.
    """
    if not coordinates:
        raise ValueError(
            f"Phrase '{phrase_text}': coordinates dict is empty — "
            f"cannot compute centroid."
        )

    xs = [x for (x, _) in coordinates.values()]
    ys = [y for (_, y) in coordinates.values()]
    cx = int(round(sum(xs) / len(xs)))
    cy = int(round(sum(ys) / len(ys)))

    cx = max(0, min(cx, grid_size - 1))
    cy = max(0, min(cy, grid_size - 1))

    fp = np.zeros(grid_size * grid_size, dtype=np.float32)
    idx = xy_to_morton(cx, cy, grid_size) if use_morton else cy * grid_size + cx
    fp[idx] = 1.0
    return fp
```
> **Bug Fixed:** The original `build_fingerprint` stub passed the full `coordinates` dict (all contexts) without computing the centroid, leaving the implementation incomplete. The corrected version averages all context positions before snapping to the nearest grid cell.

---

## `load_phrase_metadata` — Corrected Signature

The function loads from `term_context_matrix.json`, **not** from a hypothetical `phrase_metadata.json` as originally named. Required keys are `"phrases"` and `"phrase_frequencies"`:

```python
def load_phrase_metadata(path: str) -> Tuple[List[str], List[int]]:
    data        = json.load(open(path))
    phrases     = data["phrases"]
    frequencies = data["phrase_frequencies"]
    assert len(phrases) == len(frequencies)
    return phrases, frequencies
```
---

## Morton Encoding — Correctness Constraints

`xy_to_morton` raises `ValueError` if either coordinate is outside $[0, g-1]$. Callers must clamp centroid values before passing:

```python
cx = max(0, min(cx, grid_size - 1))
cy = max(0, min(cy, grid_size - 1))
```
For grid sizes that are **not** powers of two, the Morton mapping is injective but not surjective (some indices in $[0, g^2-1]$ are unreachable). This is harmless for fingerprinting but means some vector dimensions are structurally always zero.

---

## CLI Reference

```bash
python phrase_fingerprints.py \
    --coordinates   runs/run_001/context_coordinates.json \
    --metadata      runs/run_001/term_context_matrix.json \
    --output-dir    runs/run_001/ \
    --grid-size     64 \
    --use-morton \
    --smooth \
    --sigma         1.5
```
| Flag | Default | Description |
|------|---------|-------------|
| `--coordinates` | required | Path to `context_coordinates.json` |
| `--metadata` | required | Path to `term_context_matrix.json` |
| `--output-dir` | required | Output directory |
| `--grid-size` | `64` | Side length of semantic grid (power of 2 recommended) |
| `--use-morton` / `--no-morton` | Morton enabled | Linearisation strategy |
| `--smooth` / `--no-smooth` | disabled | Apply Gaussian smoothing |
| `--sigma` | `1.5` | Gaussian kernel standard deviation |

**Exit codes:** 0 success · 1 file not found · 2 malformed JSON · 3 coordinate out of bounds · 4 runtime error.

---

## Computational Complexity

| Stage | Complexity | Notes |
|-------|-----------|-------|
| Load coordinates | $O(N_c)$ | $N_c$ = number of contexts |
| Load metadata | $O(N_p)$ | $N_p$ = number of phrases |
| Centroid computation | $O(N_p \cdot \bar{C})$ | $\bar{C}$ = average contexts per phrase |
| Fingerprint construction | $O(N_p \cdot g^2)$ | Gaussian convolution dominates if smoothing enabled |
| Write `.npz` | $O(N_p \cdot g^2)$ | Compression reduces disk footprint |

**Total:** $O(N_p \cdot g^2)$ for large grids.

---

---

# Part 2 — `doc_fingerprints.py` (Step 5)

## Overview

Step 5 aggregates phrase-level SDRs (from Step 4) into document-level representations. For each document, vocabulary-matched phrases are extracted (using the same normalisation pipeline as Step 1), their fingerprint vectors are combined via TF-IDF weighted summation, and the result is sparsified to a fixed-density SDR using Morton-ordered thresholding.

The module is the "read" counterpart of Step 1: it must mirror every extraction, normalisation, and expansion decision made during corpus indexing. Inconsistency between Steps 1 and 5 is the most common source of silent quality degradation in the pipeline.

---

## Consistency Contract

Every phrase extracted from a document in Step 5 traverses the **identical** three-stage pipeline as Step 1:


raw text
    └─ extract_and_normalize_phrases()     # spaCy / NLTK + normalize_phrase()
            └─ expand_phrases()            # sub-phrase generation
                    └─ vocab filter        # keep only phrase_fps keys

This guarantees that a document containing `"deep neural network"` activates fingerprints for `"deep neural"`, `"neural network"`, and `"neural"`, exactly as during vocabulary construction.

**Parameters that must match Step 1:**

| Parameter | CLI Flag | Step 1 equivalent |
|-----------|----------|-------------------|
| `use_spacy` | `--no-spacy` | `--no-spacy` |
| `remove_verbs` | `--keep-verbs` | `--keep-verbs` |
| `filter_generic` | `--no-filter-generic` | `--no-filter-generic` |
| `min_word_length` | `--min-word-length` | `--min-word-length` |

---

## Inputs

### `phrases.txt` (Step 1 output)
Format: `phrase:count` per line, sorted by descending frequency.

### `phrase_fingerprints/` (Step 4 output)
Directory containing `phrase_fingerprints.npz` and `phrase_fingerprints_meta.json`.

### Corpus CSV
Same `context_id,context_text` file used in Step 1.

---

## Outputs

### `doc_fingerprints.npz`
Shape `(n_docs, g²)`, `dtype float32`. Row $i$ is the SDR for the document mapped to index $i$ in the metadata file.

### `doc_fingerprints_meta.json`
Maps `doc_id → row_index` for $O(1)$ lookup without loading the full matrix.

### `doc_fingerprints_stats.json`
Run-level provenance statistics:

```json
{
    "total_documents"    : 500,
    "fingerprinted_docs" : 487,
    "skipped_docs"       : 13,
    "skip_rate_pct"      : 2.6,
    "vector_size"        : 256,
    "avg_active_bits"    : 24.3,
    "grid_size"          : 16,
    "top_percent"        : 0.1
}
```
---

## Core Algorithm

### Per-Document Fingerprint Construction

The document fingerprint is the TF-IDF weighted sum of matched phrase vectors:

$$\mathbf{f}_d = \sum_{p \in P(d)} \text{tf}(p, d) \cdot \text{idf}(p) \cdot \mathbf{f}_p$$

where:

- $P(d)$ is the multiset of vocabulary-matched phrases extracted from document $d$ (including sub-phrase expansion paths; duplicates preserved for natural TF boosting)
- $\mathbf{f}_p \in \mathbb{R}^{g^2}$ is the phrase SDR from Step 4
- $\text{tf}(p, d)$ is the within-document occurrence count
- $\text{idf}(p) = \log\!\left(\tfrac{N}{|\{d : p \in d\}|}\right)$ is computed over the full corpus

Phrases absent from the IDF dictionary receive a default weight of $1.0$.

### SDR Sparsification

The weighted accumulator is sparsified to a target density using Morton-ordered thresholding:

$$k = \max\!\left(1,\, \text{round}(\rho \cdot g^2)\right), \quad \rho = \text{top\_percent}$$

`lib.sparsify_fingerprint` retains only the top-$k$ cells by activation value. Morton ordering is used as a tie-breaking criterion, preserving the topographic structure of the grid.

**Mathematical guarantee:**

$$\|\mathbf{f}_d\|_0 \leq k$$

where $\|\cdot\|_0$ counts non-zero elements.

### Optional Normalisation

After sparsification, each SDR is normalised:

| Method | Operation |
|--------|-----------|
| `l2` (default) | $\hat{\mathbf{f}} = \mathbf{f} / \|\mathbf{f}\|_2$ |
| `l1` | $\hat{\mathbf{f}} = \mathbf{f} / \|\mathbf{f}\|_1$ |
| `max` | $\hat{\mathbf{f}} = \mathbf{f} / \max(\mathbf{f})$ |

Normalisation enables cosine similarity comparisons between documents of varying length without explicit length normalisation in Step 6.

---

## `extract_phrases_from_doc` — Three-Stage Pipeline

```python
def extract_phrases_from_doc(
    text, phrase_fps, use_spacy=True, remove_verbs=False,
    filter_generic=True, min_word_length=3,
) -> List[str]:

    # Stage 1: extraction + normalisation
    candidates = extract_and_normalize_phrases(
        text, use_spacy=use_spacy, remove_verbs=remove_verbs
    )

    # Stage 2: sub-phrase expansion
    expanded = expand_phrases(
        list(candidates),
        filter_generic=filter_generic,
        min_word_length=min_word_length,
    )

    # Stage 3: vocabulary filter
    return [p for p in expanded if p in phrase_fps]
```
**Design note — duplicates preserved:** `expand_phrases` returns a sorted, deduplicated list. However, duplicates may still arise if a phrase is reachable via multiple expansion paths from distinct parent phrases. These duplicates are intentionally preserved in the returned list so that the TF counter in `build_document_fingerprint` accumulates them as natural term-frequency boosts.

---

## `build_document_fingerprint` — Implementation

```python
def build_document_fingerprint(doc_text, phrase_fingerprints,
                                idf_weights, grid_size, ...) -> Optional[csr_matrix]:
    n   = grid_size * grid_size
    acc = lil_matrix((1, n), dtype=np.float32)

    matched_phrases = extract_phrases_from_doc(doc_text, phrase_fingerprints, ...)

    if not matched_phrases:
        return None

    # TF count
    tf = {}
    for phrase in matched_phrases:
        tf[phrase] = tf.get(phrase, 0) + 1

    # Weighted accumulation
    for phrase, term_freq in tf.items():
        vec    = phrase_fingerprints.get(phrase)
        weight = term_freq * idf_weights.get(phrase, 1.0)

        if isinstance(vec, np.ndarray):
            flat = vec.flatten()[:n]
            acc[0, :len(flat)] += weight * flat
        else:
            acc += weight * vec   # csr_matrix path

    return acc.tocsr()
```
Both `csr_matrix` and 1-D `np.ndarray` phrase vectors are handled transparently, maintaining compatibility with both storage formats produced by Step 4.

---

## `sparsify_to_sdr` — Thin Wrapper

```python
def sparsify_to_sdr(fingerprint, top_percent, grid_size) -> csr_matrix:
    top_k = max(1, int(round(top_percent * grid_size * grid_size)))
    return sparsify_fingerprint(
        fingerprint, top_k=top_k, use_zorder=True, grid_size=grid_size
    )
```
The function translates the fractional density parameter into an absolute bit count before delegating to `lib.sparsify_fingerprint`. The `use_zorder=True` flag ensures Morton ordering is used for tie-breaking, consistent with Step 4's encoding.

---

## Full Pipeline Architecture

```bash
Input: corpus.txt (context_id, context_text CSV)
    │
    ▼
[1] Load phrase inventory (phrases.txt, min_freq filter)
    │
    ▼
[2] Load phrase fingerprints (Step 4 .npz + _meta.json)
    │  Filter to loaded inventory
    │
    ▼
[3] Load corpus via load_contexts_dict()
    │
    ▼
[4] Compute IDF weights over full corpus vocabulary
    │
    ▼
[5] Per-document loop:
    │
    ├── extract_phrases_from_doc()      ← mirrors Step 1 exactly
    │       ├─ extract_and_normalize_phrases()
    │       ├─ expand_phrases()
    │       └─ vocab filter
    │
    ├── build_document_fingerprint()    ← TF-IDF weighted sum
    │
    ├── sparsify_to_sdr()               ← Morton-ordered top-k
    │
    └── normalize_fingerprint()         ← optional L1/L2/max
    │
    ▼
[6] Optional: compute_fingerprint_diversity() on random sample
    │
    ▼
[7] Stack sparse dict → dense (n_docs, g²) float32 matrix
    │
    ▼
Output: doc_fingerprints.npz + _meta.json + _stats.json
```
---

## Computational Complexity

Let $N$ = documents, $L$ = average document length, $P$ = vocabulary size, $g$ = grid size.

| Stage | Complexity | Notes |
|-------|-----------|-------|
| Phrase extraction (spaCy) | $O(N \cdot L)$ | Dominates for large corpora |
| IDF computation | $O(N \cdot P)$ | Single pass over all (doc, phrase) pairs |
| TF-IDF accumulation | $O(N \cdot \bar{k} \cdot g^2)$ | $\bar{k}$ = avg matched phrases per doc |
| Sparsification | $O(N \cdot g^2 \log g^2)$ | Sort per document |
| Stacking | $O(N \cdot g^2)$ | Dense matrix assembly |
| **Total** | $O(N \cdot L + N \cdot g^2 \log g^2)$ | spaCy or stacking dominates |

**Space:** $O(N \cdot g^2)$ for the output matrix; $O(P \cdot N)$ for IDF computation (can be streamed).

---

## Configuration Parameters

| Parameter | CLI Flag | Default | Notes |
|-----------|----------|---------|-------|
| `grid_size` | `--grid-size` | `16` | Must match Steps 3 & 4 |
| `top_percent` | `--top-percent` | `0.1` | Fraction of bits kept active |
| `min_freq` | `--min-freq` | `1` | Phrase inventory filter |
| `normalize` | `--normalize` / `--no-normalize` | `True` | Enable L2 normalisation |
| `normalize_method` | `--normalize-method` | `l2` | `l1`, `l2`, or `max` |
| `use_spacy` | `--no-spacy` | `True` | Must mirror Step 1 |
| `remove_verbs` | `--keep-verbs` | `True` | Must mirror Step 1 |
| `filter_generic` | `--no-filter-generic` | `True` | Must mirror Step 1 |
| `min_word_length` | `--min-word-length` | `3` | Must mirror Step 1 |
| `compute_diversity` | `--compute-diversity` | `False` | Expensive; use on small samples |
| `diversity_sample` | `--diversity-sample` | `100` | Max docs for diversity stats |

> **CLI Inversion Note:** `--keep-verbs` sets `keep_verbs=True`, which the implementation maps to `remove_verbs=not keep_verbs = False`. This inversion exists for CLI readability (the flag describes what is *kept*, not what is *removed*), but can be confusing when reading function signatures directly.

---

## `write_outputs` — Output Contract

Three files are always written atomically (either all succeed or an `OSError` is raised):

```python
def write_outputs(fingerprints, doc_index_map, stats, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "doc_fingerprints.npz",
                        fingerprints=fingerprints)
    json.dump(doc_index_map, open(output_dir / "doc_fingerprints_meta.json","w"))
    json.dump(stats,         open(output_dir / "doc_fingerprints_stats.json","w"))

The function is intentionally decoupled from `build_doc_fingerprints` to allow unit testing of the main pipeline without filesystem side effects.
```
---

## `coords_to_csr` — Coordinate Set Conversion

A utility function used when phrase fingerprints are assembled from raw coordinate sets rather than pre-built vectors:

```python
def coords_to_csr(coords: Set[Tuple[int,int]], grid_size: int) -> csr_matrix:
    n   = grid_size * grid_size
    mat = lil_matrix((1, n), dtype=np.float32)
    for (r, c) in coords:
        if 0 <= r < grid_size and 0 <= c < grid_size:
            mat[0, r * grid_size + c] = 1.0
    return mat.tocsr()

Out-of-bounds coordinates are silently ignored, guarding against off-by-one errors from upstream grid mapping. Shape: `(1, g²)`, compatible with the accumulator in `build_document_fingerprint`.
```
---

## Integration with Step 6 — Query Processing

Document fingerprints produced here are consumed by `query_processing.py` (Step 6). Two compatibility requirements apply:

**Binary vs. floating-point SDRs:** Step 5 outputs normalised floating-point SDRs. Step 6 (Approach A) retains these as binary matrices and computes similarity using floating-point query vectors via dot-product (weighted overlap):

$$\text{sim}(q, d) = \mathbf{q}_{\text{float}} \cdot \mathbf{d}_{\text{binary}}^T$$

This allows IDF weighting in the query without modifying stored document SDRs, preserving backward compatibility with any existing index.

**Grid parameter consistency:** `grid_size` must be identical across Steps 3, 4, 5, and 6. A mismatch causes silent dimension errors or incorrect similarity scores.

---

## Known Issues

### Token Map Misalignment Warning


WARNING | token_map has 831 entries but matrix has 862 rows —
          index map and matrix may be misaligned.

**Cause:** Phrases are fingerprinted and stored in the sparse matrix during Step 4, but subsequently deduplicated or filtered out of `phrase_fingerprints_meta.json`. The result is 31 orphaned rows in the matrix with no corresponding phrase label.

**Impact:** No functional impact on query processing. Orphaned rows are never matched during vocabulary lookup.

**Remediation:** Re-run Step 4 with consistent filtering parameters, or implement a post-processing step to prune orphaned matrix rows by aligning the sparse matrix to the metadata index after construction.

### spaCy Unavailability Warning


WARNING | spaCy requested but unavailable — falling back to NLTK.
         Ensure this matches the setting used in Step 1.

If Step 1 used spaCy and Step 5 falls back to NLTK (or vice versa), phrase extraction will differ, producing lower-quality document fingerprints. Always verify that the same extraction backend is available in both steps.

---

## Limitations and Future Work

1. **Language dependency** — Both modules assume English text and the `en_core_web_sm` spaCy model.
2. **Single-pass IDF** — IDF weights are computed over the corpus at fingerprint-build time. For incremental indexing, a streaming IDF update mechanism is needed.
3. **Dense output matrix** — Step 5 stacks all SDRs into a dense float32 matrix. For corpora exceeding $10^4$ documents on a 64×64 grid, this requires 1.6 GB of RAM. A sparse output format would be more scalable.
4. **No incremental update** — Adding new documents requires a full rebuild. An append-only mode would support live indexing.
5. **Frequency-weighted centroid** — Step 4's centroid computation is currently unweighted. A frequency-weighted variant (using `phrase_frequencies` from `term_context_matrix.json`) may produce better-localised grid positions for high-frequency phrases.

---

## Conclusion

Steps 4 and 5 form the representation bridge between linguistic analysis (Steps 1–3) and semantic retrieval (Step 6). The key design decisions are:

- **Centroid-based grid placement** (Step 4) — uses the full distributional evidence across all contexts to compute a phrase's semantic locus, avoiding the identical-fingerprint bug of single-context lookup.
- **Morton encoding** — preserves 2-D spatial locality under 1-D Hamming distance, making the fingerprint index semantically meaningful.
- **Three-stage extraction mirroring** (Step 5) — strict reuse of Step 1's normalisation, expansion, and vocabulary filter pipeline ensures representational consistency throughout the index.
- **TF-IDF weighted accumulation** — combines term importance (IDF) with within-document frequency (TF) before sparsification, producing document SDRs that reflect both the breadth and the discriminative weight of their constituent phrases.
- **Morton-ordered sparsification** — fixes SDR density at a target fraction while preserving topographic structure during tie-breaking, yielding compact representations with controlled overlap statistics.

