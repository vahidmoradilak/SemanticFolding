# lib.py — Quick Reference

## Table of Contents
1. [Core NLP Utilities](#core-nlp-utilities)
2. [Text Normalization](#text-normalization)
3. [Phrase Expansion](#phrase-expansion)
4. [File I/O Utilities](#file-io-utilities)
5. [Coordinate Utilities](#coordinate-utilities)
6. [Similarity Computation](#similarity-computation)
7. [Z-Order Curve Utilities](#z-order-curve-utilities)
8. [Fingerprint Manipulation](#fingerprint-manipulation)
9. [Diversity Metrics](#diversity-metrics)

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
| `expand_phrases` | `phrases: List[str], remove_verbs: bool = True`, filter_generic: bool = True, min_word_length: int = 3` | `List[str]` — sorted deduplicated sub-phrases |

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

## Similarity Computation

| Function | Parameters | Returns |
|---|---|---|
| `compute_cosine_similarity` | `vec1: np.ndarray, vec2: np.ndarray` | `float` ∈ $[-1, 1]$ |

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

Handles zero vectors (returns `0.0`). Accepts sparse matrices.

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
| `sparsify_fingerprint` | `fingerprint: csr_matrix, top_k: int, use_zorder: bool = True, grid_size: Optional[int] = None` | `csr_matrix` |

**Normalization methods:**

| method | behavior |
|---|---|
| `'l2'` | unit vector (use with cosine similarity) |
| `'l1'` | sum-to-1 |
| `'binary'` | all non-zero → 1 |

---

## Diversity Metrics

| Function | Parameters | Returns |
|---|---|---|
| `compute_fingerprint_diversity` | `fingerprints: Dict[str, csr_matrix], sample_size: int = 100` | `Dict[str, float]` |

**Diversity keys:** `avg_similarity`, `diversity_score`, `num_samples`

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
from lib import load_phrases, load_contexts_dict

phrases  = load_phrases(Path("phrases.txt"), min_freq=5)
ctx_raw  = load_contexts_dict(Path("corpus.csv"))     # raw text

### Load & use fingerprints
```python
from lib import (load_phrase_fingerprints_sparse,
                 compute_cosine_similarity,
                 normalize_fingerprint)

phrase_fps = load_phrase_fingerprints_sparse(Path("phrase_fps.csv"), grid_size=128)

q_fp = normalize_fingerprint(phrase_fps["query_phrase"], method="l2")
d_fp = normalize_fingerprint(phrase_fps["target_phrase"], method="l2")

score = compute_cosine_similarity(q_fp, d_fp)
```
### Build & sparsify a fingerprint
```python
from lib import sparsify_fingerprint

sparse = sparsify_fingerprint(merged, top_k=50, use_zorder=True, grid_size=128)
```