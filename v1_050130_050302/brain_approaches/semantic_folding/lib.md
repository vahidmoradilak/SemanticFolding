# lib.py — Quick Reference

## Table of Contents
1. [Core NLP Utilities](#core-nlp-utilities)
2. [Text Normalization](#text-normalization)
3. [Phrase Expansion](#phrase-expansion)
4. [File I/O Utilities](#file-io-utilities)
5. [Fingerprint Loading & Saving](#fingerprint-loading--saving)
6. [Coordinate Utilities](#coordinate-utilities)
7. [IDF Computation](#idf-computation)
8. [Similarity Computation](#similarity-computation)
9. [Z-Order Curve Utilities](#z-order-curve-utilities)
10. [Fingerprint Manipulation](#fingerprint-manipulation)
11. [Validation & Stats](#validation--stats)
12. [Extended Utilities](#extended-utilities)

---

## Core NLP Utilities

| Function | Parameters | Returns |
|---|---|---|
| `get_wordnet_pos` | `treebank_tag: str` | `str` — WordNet POS constant |
| `lemmatize_token` | `word: str, pos_tag: str` | `str` — lowercased lemma *(cached × 10k)* |
| `is_generic_word` | `word: str, min_length: int = 3` | `bool` — True if too generic |
| `is_valid_phrase_structure` | `tagged_tokens: List[Tuple[str,str]]` | `bool` — True if has noun/adj |

---

## Text Normalization

| Function | Parameters | Returns |
|---|---|---|
| `normalize_phrase` | `text: str, remove_verbs: bool = True` | `Optional[str]` — normalized phrase or `None` |
| `remove_edge_stop_words` | `text: str` | `str` — trimmed phrase |

**normalize_phrase pipeline:**
```
lowercase → clean punct → tokenize → POS tag →
filter stopwords → filter verbs? → lemmatize → validate → join
```
---

## Phrase Expansion

| Function | Parameters | Returns |
|---|---|---|
| `expand_phrases` | `phrases: List[str], filter_generic: bool = True, min_word_length: int = 3` | `List[str]` — sorted deduplicated sub-phrases |

**Expansion rules:**

| Input length | Generated sub-phrases |
|---|---|
| 2-word | unigrams |
| 3-word | bigrams + unigrams |
| 4+-word | trigrams + bigrams + unigrams |

---

## File I/O Utilities

| Function | Parameters | Returns |
|---|---|---|
| `load_phrases` | `phrases_path: Path, min_freq: int = 0` | `List[Tuple[str, int]]` — (phrase, freq) |
| `load_contexts` | `corpus_path: Path` | `List[Tuple[str, str]]` — (id, normalized_text) |
| `load_contexts_dict` | `corpus_path: Path` | `Dict[str, str]` — id → raw text |
| `find_phrase_occurrences` | `text: str, phrase: str, use_word_boundaries: bool = True` | `int` — count |

**File formats:**

# phrases file
machine learning:150
neural network:89

# corpus file (CSV)
ctx_0,Machine learning is a subset of AI
ctx_1,Neural networks are inspired by neurons

---

## Fingerprint Loading & Saving

| Function | Parameters | Returns |
|---|---|---|
| `load_phrase_fingerprints_sparse` | `fingerprints_path: Path, grid_size: int` | `Dict[str, Set[Tuple[int,int]]]` — phrase → {(x,y)} |
| `load_fingerprint_cache` | `cache_path: Path, grid_size: int` | `Dict[str, csr_matrix]` — doc_id → (1 × D) |
| `save_fingerprint_cache` | `fingerprints: Dict[str, csr_matrix], cache_path: Path, grid_size: int` | `None` |
| `load_document_fingerprints_sparse` | `doc_fps_path: Path` | `Dict[str, csr_matrix]` — doc_id → (1 × D) |

**Cache JSON format:**
```json
{
  "doc_id_1": {
    "coordinates": [[x1,y1], [x2,y2]],
    "values": [v1, v2]
  }
}
```
---

## Coordinate Utilities

| Function | Parameters | Returns |
|---|---|---|
| `load_context_coordinates` | `coords_path: Path` | `Dict[str, Tuple[int,int]]` — ctx_id → (x, y) |

**File format:**
```bash
context_id,x,y
ctx_0,45,67
```
---

## IDF Computation

| Function | Parameters | Returns |
|---|---|---|
| `compute_idf_weights` | `phrases: List[str], contexts: List[str]` | `Dict[str, float]` — phrase → IDF score |

$$IDF(p) = \log\left(\frac{N}{df(p)}\right)$$

Phrases absent from all contexts receive $\log(N)$ (maximum IDF).

---

## Similarity Computation

| Function | Parameters | Returns |
|---|---|---|
| `compute_cosine_similarity` | `vec1: np.ndarray, vec2: np.ndarray` | `float` ∈ $[-1, 1]$ |
| `compute_jaccard_similarity` | `set1: Set, set2: Set` | `float` ∈ $[0, 1]$ |
| `batch_compute_similarities` | `query_fp: csr_matrix, doc_fingerprints: Dict[str, csr_matrix], batch_size: int = 100` | `Dict[str, float]` — doc_id → score |
| `get_fingerprint_overlap` | `fp1: csr_matrix, fp2: csr_matrix` | `Tuple[int,int,int]` — (∩, fp1-only, fp2-only) |

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|} \qquad J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

Both handle zero vectors (returns `0.0`). `compute_cosine_similarity` accepts sparse matrices.

---

## Z-Order Curve Utilities

| Function | Parameters | Returns |
|---|---|---|
| `xy_to_morton` | `x: int, y: int` | `int` — Morton code |
| `morton_to_xy` | `morton: int` | `Tuple[int,int]` — (x, y) |
| `get_zorder_neighbors` | `x: int, y: int, grid_size: int, radius: int = 1` | `List[Tuple[int,int]]` — neighbors |

**Quick reference:**

| x | y | Morton |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 2 |
| 1 | 1 | 3 |

---

## Fingerprint Manipulation

| Function | Parameters | Returns |
|---|---|---|
| `normalize_fingerprint` | `fingerprint: csr_matrix, method: str = 'l2'` | `csr_matrix` |
| `merge_fingerprints` | `fingerprints: List[csr_matrix], weights: Optional[List[float]] = None` | `csr_matrix` |
| `sparsify_fingerprint` | `fingerprint: csr_matrix, top_k: int, use_zorder: bool = True, grid_size: Optional[int] = None` | `csr_matrix` |

**Normalization methods:**

| method | behavior |
|---|---|
| `'l2'` | unit vector (use with cosine similarity) |
| `'l1'` | sum-to-1 |
| `'binary'` | all non-zero → 1 |

---

## Validation & Stats

| Function | Parameters | Returns |
|---|---|---|
| `validate_fingerprint` | `fingerprint: csr_matrix, grid_size: int, min_active: int = 1, max_active: Optional[int] = None` | `bool` |
| `compute_fingerprint_stats` | `fingerprints: Dict[str, csr_matrix]` | `Dict[str, float]` |
| `compute_fingerprint_diversity` | `fingerprints: Dict[str, csr_matrix], sample_size: int = 100` | `Dict[str, float]` |

**Stats keys:** `n_fingerprints`, `total_dimensions`, `mean_active_bits`, `std_active_bits`, `min_active_bits`, `max_active_bits`, `mean_sparsity`

**Diversity keys:** `avg_similarity`, `diversity_score`, `num_samples`

---

## Extended Utilities

| Function | Parameters | Returns |
|---|---|---|
| `export_fingerprints_to_numpy` | `fingerprints: Dict[str, csr_matrix], output_path: Path, grid_size: int` | `None` — saves `.npz` |
| `visualize_fingerprint` | `fingerprint: csr_matrix, grid_size: int, title: str, output_path: Optional[Path] = None` | `None` — shows/saves heatmap |

---

## Quick Usage Guide

### Normalize & expand phrases
```python
from lib import normalize_phrase, expand_phrases

phrase = normalize_phrase("Machine Learning Algorithms")
# → 'machine learning algorithm'

sub_phrases = expand_phrases(["deep neural network"])
# → ['deep', 'neural', 'network', 'deep neural',
#    'neural network', 'deep neural network']
```

### Load corpus data
```python
from lib import load_phrases, load_contexts, load_contexts_dict

phrases  = load_phrases(Path("phrases.txt"), min_freq=5)
contexts = load_contexts(Path("corpus.csv"))          # normalized
ctx_raw  = load_contexts_dict(Path("corpus.csv"))     # raw text

### Load & use fingerprints
```python
from lib import (load_phrase_fingerprints_sparse,
                 load_fingerprint_cache,
                 compute_cosine_similarity,
                 normalize_fingerprint)

phrase_fps = load_phrase_fingerprints_sparse(Path("phrase_fps.csv"), grid_size=128)
doc_fps    = load_fingerprint_cache(Path("doc_fps.json"), grid_size=128)

q_fp = normalize_fingerprint(doc_fps["query_doc"], method="l2")
d_fp = normalize_fingerprint(doc_fps["target_doc"], method="l2")

score = compute_cosine_similarity(q_fp, d_fp)
```
### Build & sparsify a fingerprint
```python
from lib import merge_fingerprints, sparsify_fingerprint

merged = merge_fingerprints([fp1, fp2, fp3], weights=[0.5, 0.3, 0.2])
sparse = sparsify_fingerprint(merged, top_k=50, use_zorder=True, grid_size=128)
```
### Compute IDF weights
```python
from lib import compute_idf_weights

phrase_list = [p for p, _ in phrases]
ctx_texts   = [text for _, text in contexts]
idf         = compute_idf_weights(phrase_list, ctx_texts)
```
### Validate & inspect fingerprints
```python
from lib import validate_fingerprint, compute_fingerprint_stats

ok    = validate_fingerprint(fp, grid_size=128, min_active=10, max_active=200)
stats = compute_fingerprint_stats(doc_fps)
print(stats["mean_active_bits"], stats["mean_sparsity"])
```