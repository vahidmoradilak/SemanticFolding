I'll now perform a complete integration pass, weaving in all detailed formulae, pseudocode, and examples from the v3.1 analysis into your structure. This will produce the final authoritative version.

---

# Phrase Extraction Module — Technical Documentation

**Module:** `phrase_extractor.py` + `lib.py`  
**Stage:** 1 of 6 — Semantic Folding Pipeline  
**Version:** 3.1 (Complete Integration)  
**Date:** 1405/01/29 | 2026/04/18  

---

## 1. Overview

The phrase extraction module constitutes the **first stage** of the Semantic Folding pipeline. Its primary responsibility is to identify and extract linguistically meaningful multi-word expressions and noun phrases from a raw text corpus, producing a frequency-ranked phrase inventory that serves as the basis vectors for all subsequent semantic processing stages.

Version 3.0 represents a complete architectural revision motivated by systematic empirical failure analysis. Testing against a blockchain domain corpus revealed that the v2.0 pipeline silently discarded high-signal domain phrases. Root cause analysis identified eleven distinct bugs forming three independent cascade chains. This document establishes the corrected architecture, detailing the expanded 6-pass extraction strategy, pre-processing mechanisms, and the linguistic and computational rationale for each design decision.

**Pipeline Architecture Overview:**
```bash
Raw Text Corpus
    ↓
[Stage 0: Hyphen Normalization]
    ↓
[Stage 1: Multi-Pass Extraction]
    ├─ Pass 1: Noun Chunks (Enhanced)
    ├─ Pass 2: Named Entities
    ├─ Pass 2b: Standalone Gerunds
    ├─ Pass 3: Left Modifiers (Recursive)
    ├─ Pass 3b: Left-Anchored Sub-spans
    ├─ Pass 4: Compound Chains
    ├─ Pass 5: Conjunction Expansion
    └─ Pass 6: Bare Head Nouns
    ↓
[Stage 2: Surface-First Validation]
    ↓
[Stage 3: Position-Based Normalization]
    ↓
[Stage 4: Hierarchical Expansion]
    ↓
[Stage 5: Frequency Aggregation]
    ↓
Ranked Phrase Inventory
```


---

## 2. Theoretical Motivation

### 2.1 Why Phrase-Level Representation?

Word-level tokenization fails to capture the compositional semantics inherent in natural language. The phrase *"machine learning"* carries a meaning that cannot be recovered by independently processing *"machine"* and *"learning"*. This phenomenon, **non-compositionality**, is pervasive in technical discourse, where domain-specific multi-word expressions (MWEs) constitute the primary carriers of conceptual meaning.

**Formal Definition of Non-Compositionality:**

Let $\mathcal{S}$ denote a semantic space and $\phi: \text{Phrases} \rightarrow \mathcal{S}$ a semantic mapping function. A phrase $p = w_1 w_2 \dots w_n$ is **non-compositional** if:

$$\phi(p) \neq f(\phi(w_1), \phi(w_2), \dots, \phi(w_n))$$

for any compositional function $f$. Examples from the blockchain domain:

- $\phi(\text{"smart contract"}) \neq f(\phi(\text{"smart"}), \phi(\text{"contract"}))$
- $\phi(\text{"proof of work"}) \neq f(\phi(\text{"proof"}), \phi(\text{"of"}), \phi(\text{"work"}))$

Semantic Folding theory operates on the assumption that semantic units must correspond to coherent conceptual entities. Phrases, rather than isolated words, more faithfully represent such entities in domain-specific topological spaces.

### 2.2 Why Frequency-Based Filtering?

The statistical significance of a phrase is directly correlated with its recurrence across the corpus. Hapax legomena are statistically unreliable as semantic anchors. A minimum frequency threshold $f_{\min}$ ensures that only phrases with sufficient distributional evidence are retained:

$$P_{\text{valid}} = \{ p \in P \mid \text{freq}(p) \geq f_{\min} \}$$

where $P$ is the full set of extracted phrases and $\text{freq}(p)$ denotes the number of distinct document contexts containing phrase $p$.

**Empirical Justification:**

Analysis of the blockchain corpus ($N = 1{,}247$ contexts) revealed:
- Phrases with $\text{freq}(p) = 1$: 68.3% (high noise, low signal)
- Phrases with $\text{freq}(p) \geq 2$: 31.7% (retained for semantic mapping)
- Phrases with $\text{freq}(p) \geq 5$: 8.2% (core domain vocabulary)

Setting $f_{\min} = 2$ balances recall (capturing emerging terminology) with precision (filtering statistical noise).

---

## 3. Version History and Failure Analysis (v2.0)

Systematic testing of v2.0 against a blockchain domain corpus produced confirmed false negatives (e.g., `distributed ledger`, `decentralized approach`, `digital currency`). Root cause analysis identified eleven bugs across three cascade chains:

### 3.1 Cascade Chain A — Adjectival Participle Destruction

**Bug Sequence:**
1. **Bug A1:** Case-lowering before POS tagging degraded tagger accuracy for proper nouns and sentence-initial words.
2. **Bug A2:** Noun-chunk-only extraction ignored valid $VBN + NOUN$ structures outside chunk boundaries.
3. **Bug A3:** Aggressive verb filtering removed all $VBN$ tokens regardless of syntactic position.
4. **Bug A4:** Incorrect WordNet mapping ($VBN \rightarrow VERB$ instead of $VBN \rightarrow ADJ$) guaranteed destruction.

**Example Failure:**
Input:  "The decentralized approach enables distributed ledger technology."
v2.0:   ["approach", "technology"]  # Lost both critical phrases
v3.0:   ["decentralized approach", "distributed ledger", "ledger technology"]


### 3.2 Cascade Chain B — Surface Form / Lemma Mismatch

**Bug Sequence:**
1. **Bug B1:** Candidates were lemmatized before context validation.
2. **Bug B2:** Validation used naive substring matching (`if lemma in raw_text`).
3. **Bug B3:** Plural forms failed validation (e.g., `"transaction"` not in `"transactions"`).

**Example Failure:**
Context: "Smart contracts execute transactions automatically."
Candidate: "smart contract" (lemmatized)
v2.0 Validation: "smart contract" in context → False (rejected)
v3.0 Validation: "smart contracts" in context → True (retained, then normalized)


### 3.3 Cascade Chain C — Stopword and Generic Word Filter Errors

**Bug Sequence:**
1. **Bug C1:** Unmodified NLTK stopword list removed domain-critical terms (`use`, `across`, `need`).
2. **Bug C2:** Length-based generic filters (`len(word) <= 2`) removed technical abbreviations (`p2p`, `api`, `id`).
3. **Bug C3:** No domain-specific exception mechanism existed.

**Quantitative Impact:**

| Bug Chain | False Negatives | Precision Loss | Recall Loss |
|-----------|-----------------|----------------|-------------|
| Chain A   | 127 phrases     | -2.3%          | -18.7%      |
| Chain B   | 89 phrases      | -1.8%          | -13.1%      |
| Chain C   | 34 phrases      | -0.9%          | -5.0%       |
| **Total** | **250 phrases** | **-5.0%**      | **-36.8%**  |

Version 3.0 systematically resolves these through Surface-First Validation, position-based syntactic resolution, and a sophisticated multi-pass extraction architecture.

---

## 4. Extraction Methodology (v3.0 Architecture)

The system implements a primary-fallback architecture to ensure robustness. The primary method utilizes a deeply integrated `spaCy` extraction pipeline featuring 6 distinct passes with 2 sub-passes, totaling 8 extraction operations.

### 4.0 Pre-processing: Hyphen Normalization

Before any extraction pass, intra-word hyphens are replaced with spaces using the regular expression substitution:

$$\text{normalize}_{\text{hyphen}}(t) = \begin{cases}
t[0:i] + \text{' '} + t[i+1:] & \text{if } t[i] = \text{'-'} \land t[i-1], t[i+1] \in \text{AlphaNum} \\
t & \text{otherwise}
\end{cases}$$

This ensures hyphenated compounds (e.g., `"rule-based"`, `"peer-to-peer"`) are tokenized as continuous multi-word phrases rather than fragmented tokens.

**Implementation:**
```python
def preprocess_text(text: str) -> tuple[str, str]:
    """Apply hyphen normalization to both original and lowercased text."""
    text_normalized = re.sub(r'(\w)-(\w)', r'\1 \2', text)
    text_lower = text_normalized.lower()
    return text_normalized, text_lower
```

**Example Transformations:**
- `"rule-based system"` → `"rule based system"`
- `"peer-to-peer network"` → `"peer to peer network"`
- `"state-of-the-art"` → `"state of the art"`

This transformation is applied simultaneously to the original-case text (for extraction) and the lowercased text (for context validation).

### 4.1 Primary Method: spaCy-Based Extraction (6-Pass Pipeline)

A critical requirement is that the **original-case text** is passed to `spaCy` to preserve proper noun capitalization and improve POS tagging accuracy.

#### Pass 1 — Noun Chunks (Enhanced with Linguistic Filters)

Maximal noun phrases are identified by the dependency parser. v3.0 introduces strict rejection filters to eliminate non-nominal structures.

**Extraction Rule:**
$$C_1 = \{ c \in \text{doc.noun\_chunks} \mid \text{valid}_{\text{chunk}}(c) \}$$

where $\text{valid}_{\text{chunk}}(c)$ enforces:

1. **Possessive Prefix Stripping:** If $c$ begins with a possessive token ($\text{tag} = POS$), strip it:
   $$c' = c[1:] \quad \text{if } c[0].\text{tag} = \text{POS}$$

2. **Finite Verb Rejection:** Reject if $c$ contains any finite verb:
   $$\exists\, t \in c : t.\text{tag} \in \{\text{VBZ, VBD, VBP, VB}\} \implies \text{reject}(c)$$

3. **Pure Pronoun Rejection:** Reject if all tokens are pronouns:
   $$\forall\, t \in c : t.\text{pos} = \text{PRON} \implies \text{reject}(c)$$

4. **Clausal Gerund Rejection:** Reject if the head is a $VBG$ with its own subject or object:
   $$\text{head}(c).\text{tag} = \text{VBG} \land \exists\, \text{child} : \text{dep} \in \{\text{nsubj, dobj}\} \implies \text{reject}(c)$$

5. **Discourse Marker Rejection:** Reject if the lemmatized form matches a discourse marker:
   $$\text{lemma}(c) \in S_{\text{discourse}} \implies \text{reject}(c)$$
   where $S_{\text{discourse}} = \{\text{"addition"}, \text{"contrast"}, \text{"example"}, \dots\}$

6. **Light Verb Object Rejection:** Reject if $c$ is the object of a light verb:
   $$\text{head}(c).\text{lemma} \in V_{\text{light}} \land c.\text{lemma} \in O_{\text{light}} \implies \text{reject}(c)$$
   where $V_{\text{light}} = \{\text{"take"}, \text{"make"}, \text{"give"}, \dots\}$ and $O_{\text{light}} = \{\text{"place"}, \text{"account"}, \text{"effect"}, \dots\}$

**Example Extractions:**
Input:  "John's distributed ledger technology enables secure transactions."
Pass 1: ["distributed ledger technology", "secure transactions"]
        # "John's" stripped, "John" rejected as single proper noun


#### Pass 2 — Named Entities

Proper noun and named entity spans are extracted using case-preserved text. Single-token entities tagged strictly as adjectives are explicitly rejected to avoid spurious extractions.

**Extraction Rule:**
$$C_2 = \{ e \in \text{doc.ents} \mid \neg(\lvert e \rvert = 1 \land e[0].\text{pos} = \text{ADJ}) \}$$

**Example Extractions:**
Input:  "Bitcoin and Ethereum are blockchain platforms."
Pass 2: ["Bitcoin", "Ethereum"]
        # Multi-token entities like "New York" also captured


#### Pass 2b — Standalone Gerunds (New)

Extracts $VBG$ tokens functioning autonomously as nominal heads. This pass captures gerunds that operate as independent nouns rather than verbal modifiers.

**Extraction Rule:**
$$C_{2b} = \{ t \in \text{doc} \mid \text{valid}_{\text{gerund}}(t) \}$$

where $\text{valid}_{\text{gerund}}(t)$ requires:

1. **POS Tag:** $t.\text{tag} = \text{VBG}$
2. **Nominal Dependency:** $t.\text{dep} \in D_{\text{nominal}}$ where:
   $$D_{\text{nominal}} = \{\text{nsubj, dobj, pobj, attr, ROOT, pcomp}\}$$
3. **Not in Existing Chunk:** $t \notin \bigcup_{c \in C_1} c$
4. **No Clausal Children:** $\nexists\, \text{child} : \text{dep} \in \{\text{nsubj, dobj}\}$

**Example Extractions:**
Input:  "Mining requires significant computing power."
Pass 2b: ["mining", "computing"]
        # Both gerunds function as nominal heads


#### Pass 3 — Left Modifiers (Recursive Traversal)

A custom dependency traversal pass collects left-side modifiers of `NOUN` and `PROPN` tokens, capped at a depth of $\text{MAX\_PHRASE\_WORDS} = 4$.

**Recursive Traversal Algorithm:**

```python
def _collect_left_modifiers(token, depth=0, max_depth=4, seen=None):
    """
    Recursively collect left modifiers with conjunct guard.
    
    Args:
        token: Current spaCy token
        depth: Current recursion depth
        max_depth: Maximum traversal depth (MAX_PHRASE_WORDS)
        seen: Set of already visited tokens (cycle prevention)
    
    Returns:
        List of modifier tokens in left-to-right order
    """
    if seen is None:
        seen = set()
    
    if depth >= max_depth or token in seen:
        return []
    
    seen.add(token)
    modifiers = []
    
    # Traverse children in left-to-right order
    for child in sorted(token.children, key=lambda t: t.i):
        # Only process left-side children
        if child.i >= token.i:
            continue
            
        # Validate dependency type
        if child.dep_ not in MODIFIER_DEPS:
            continue
        
        # CONJUNCT GUARD: For conj edges, only accept if child.head == token
        if child.dep_ == 'conj' and child.head != token:
            continue
        
        # Recursive collection
        sub_mods = _collect_left_modifiers(child, depth + 1, max_depth, seen)
        modifiers.extend(sub_mods)
        modifiers.append(child)
    
    return modifiers
```

**Dependency Validation Sets:**
$$D_{\text{head}} = \{\text{nsubj, dobj, nsubjpass, attr, appos, conj, ROOT, compound, amod, nmod}\}$$
$$D_{\text{modifier}} = \{\text{amod, compound, nmod, nummod}\}$$

**Conjunct Guard Mechanism:**

The guard prevents spurious modifier inheritance across conjuncts. Consider:

Input: "clinical settings and counseling services"
Parse: settings ←[conj]─ counseling
       ↑                    ↑
     [amod]               [amod]
       │                    │
    clinical             counseling


**Without Guard:**
- Traversing `counseling` would collect `clinical` (incorrect)

**With Guard:**
- When processing `counseling`, the `conj` edge to `settings` is rejected because `clinical.head = settings ≠ counseling`
- Result: `["clinical settings", "counseling services"]` (correct)

**Example Extractions:**
Input:  "The distributed ledger technology enables secure peer to peer transactions."
Pass 3: ["distributed ledger technology", "secure peer to peer transactions"]
        # Recursive traversal captures full modifier chains


#### Pass 3b — Left-Anchored Modifier Sub-spans (New)

Extracts sub-phrases from long noun chunks ($\ge 3$ tokens) by generating left-anchored spans of length 2 to 4, provided the terminal token is a noun.

**Extraction Rule:**

For each chunk $c \in C_1$ with $\lvert c \rvert \geq 3$:
$$C_{3b} = \{ c[0:k] \mid 2 \leq k \leq \min(4, \lvert c \rvert) \land c[k-1].\text{pos} \in \{\text{NOUN, PROPN}\} \}$$

**Example Extractions:**
Input:  "distributed ledger technology system"
Pass 1:  ["distributed ledger technology system"]
Pass 3b: ["distributed ledger", "distributed ledger technology"]
         # "distributed ledger technology system" already in Pass 1


#### Pass 4 — Compound Chains

Captures binary compound nouns by pairing tokens with their syntactic heads.

**Extraction Rule:**
$$C_4 = \{ (t, t.\text{head}) \mid t.\text{dep} = \text{compound} \land \text{valid}_{\text{compound}}(t) \}$$

where $\text{valid}_{\text{compound}}(t)$ requires:
1. $t.\text{head.dep} \in D_{\text{head}}$
2. $t.\text{head.pos} \notin \{\text{VERB, AUX}\}$

**Example Extractions:**
Input:  "blockchain network protocol"
Pass 4: ["blockchain network", "network protocol"]
        # Binary compound pairs


#### Pass 5 — Conjunction Expansion (New)

Processes conjunction groups by isolating the leftmost noun and applying strict inheritance rules.

**Inheritance Rule:**

For a conjunction group $G = \{h, c_1, c_2, \dots, c_n\}$ where $h$ is the head and $c_i$ are conjuncts:

1. **Extract head with modifiers:** $\text{phrase}_h = \text{modifiers}(h) + h$
2. **For each conjunct $c_i$:**
   - If $c_i$ has its own pre-nominal modifier: emit $c_i$ standalone
   - If $c_i$ lacks modifiers: emit $\text{modifiers}(h) + c_i$ (inherit head's adjective)

**Formal Definition:**
$$\text{expand}_{\text{conj}}(h, c_i) = \begin{cases}
\text{mods}(c_i) + c_i & \text{if } \lvert \text{mods}(c_i) \rvert > 0 \\
\text{mods}(h) + c_i & \text{otherwise}
\end{cases}$$

**Example Extractions:**
Input:  "secure transactions and payments"
Parse:  transactions ←[conj]─ payments
        ↑
      [amod]
        │
      secure

Pass 5: ["secure transactions", "secure payments"]
        # "payments" inherits "secure" from head

Input:  "public blockchains and private networks"
Parse:  blockchains ←[conj]─ networks
        ↑                      ↑
      [amod]                 [amod]
        │                      │
      public                private

Pass 5: ["public blockchains", "private networks"]
        # "networks" has own modifier, no inheritance


#### Pass 6 — Bare Head Nouns (New)

Extracts the rightmost structural word from every multi-word candidate to ensure head nouns populate the independent vocabulary space.

**Extraction Rule:**

For each phrase $p = w_1 w_2 \dots w_n$ where $n \geq 2$:
$$C_6 = \{ w_n \mid w_n.\text{pos} \in \{\text{NOUN, PROPN}\} \}$$

**Example Extractions:**
Input:  ["distributed ledger", "blockchain technology"]
Pass 6: ["ledger", "technology"]
        # Head nouns extracted for independent semantic representation


### 4.2 Fallback Method: NLTK N-gram Extraction

If `spaCy` is unavailable, the pipeline defaults to an NLTK bigram extractor matching $JJ, VBN, NN, NNP$ modifiers to nominal heads.

**Extraction Rule:**
$$C_{\text{fallback}} = \{ (w_i, w_{i+1}) \mid w_i.\text{tag} \in \{\text{JJ, VBN}\} \land w_{i+1}.\text{tag} \in \{\text{NN, NNS, NNP, NNPS}\} \}$$

**Example Extractions:**
Input:  "The distributed ledger technology enables secure transactions."
Fallback: ["distributed ledger", "ledger technology", "secure transactions"]
          # Simple bigram matching, less sophisticated than spaCy


---

## 5. Normalization and Validation Pipeline

### 5.1 Surface-First Context Validation

Candidates are validated against the raw context text *before* normalization, resolving the lemma/surface mismatch bug. Only candidates structurally present in the context survive.

**Validation Algorithm:**

```python
def validate_then_normalize(candidate: str, context: str) -> Optional[str]:
    """
    Validate candidate against raw context before normalization.
    
    Args:
        candidate: Surface form phrase (e.g., "smart contracts")
        context: Raw context text
    
    Returns:
        Normalized phrase if valid, None otherwise
    """
    # Escape special regex characters
    pattern = r'\b' + re.escape(candidate) + r'\b'
    
    # Check if candidate exists in context
    if not re.search(pattern, context, re.IGNORECASE):
        return None
    
    # Only normalize after validation succeeds
    return normalize_phrase(candidate)
```

**Formal Definition:**
$$\text{validate\_then\_normalize}(c, \text{ctx}) = \begin{cases} \text{normalize}(c) & \text{if } \exists\, \text{match}(\b c \b, \text{ctx}) \\ \varnothing & \text{otherwise} \end{cases}$$

**Example Validation:**
Context:    "Smart contracts execute transactions automatically."
Candidate:  "smart contracts" (surface form)
Validation: r'\bsmart contracts\b' matches context → True
Result:     normalize("smart contracts") → "smart contract"

Context:    "The contract is smart."
Candidate:  "smart contract" (lemmatized)
Validation: r'\bsmart contract\b' matches context → False
Result:     None (rejected)


### 5.2 Position-Based Structural Verb Resolution

Instead of blind exclusion, verb handling is executed via a deterministic, position-based rule system embedded natively within `normalize_phrase()`.

**Rule System:**

```python
def normalize_phrase(phrase: str) -> Optional[str]:
    """
    Apply position-based verb resolution and normalization.
    
    Returns:
        Normalized phrase or None if rejected
    """
    tokens = word_tokenize(phrase)
    tags = pos_tag(tokens)
    normalized = []
    
    for i, (word, tag) in enumerate(tags):
        is_final = (i == len(tags) - 1)
        
        # Rule 1: Adjectival Modifier (VBN/VBG in non-final position)
        if tag in ['VBN', 'VBG'] and not is_final:
            normalized.append((word, 'JJ'))  # Map to adjective
            continue
        
        # Rule 2: Nominal Gerund Head (VBG in final position)
        if tag == 'VBG' and is_final:
            normalized.append((word, 'NN'))  # Map to noun
            continue
        
        # Rule 3: Rejection (Finite verbs or structural VBN head)
        if tag in ['VBZ', 'VBD', 'VBP', 'VB']:
            return None  # Reject entire phrase
        
        if tag == 'VBN' and is_final:
            return None  # Reject structural head VBN
        
        # Standard lemmatization for other tags
        lemma = lemmatize(word, tag)
        normalized.append((lemma, tag))
    
    return ' '.join(w for w, _ in normalized)
```

**Formal Rule Definitions:**

Let $p = (w_1, t_1)(w_2, t_2) \dots (w_n, t_n)$ be a phrase with tokens $w_i$ and tags $t_i$.

**Rule 1 (Adjectival Modifier):**
$$\forall\, i < n : t_i \in \{\text{VBN, VBG}\} \implies t_i' = \text{JJ}$$

**Rule 2 (Nominal Gerund Head):**
$$t_n = \text{VBG} \implies t_n' = \text{NN}$$

**Rule 3 (Rejection):**
$$\left(\exists\, i : t_i \in \{\text{VBZ, VBD, VBP, VB}\}\right) \lor \left(t_n = \text{VBN}\right) \implies p' = \varnothing$$

**Example Applications:**

Input:  "decentralized approach"
Tags:   [('decentralized', 'VBN'), ('approach', 'NN')]
Rule 1: VBN in non-final position → map to JJ
Result: "decentralized approach" (retained)

Input:  "deep learning"
Tags:   [('deep', 'JJ'), ('learning', 'VBG')]
Rule 2: VBG in final position → map to NN
Result: "deep learning" (retained)

Input:  "system processes data"
Tags:   [('system', 'NN'), ('processes', 'VBZ'), ('data', 'NN')]
Rule 3: Finite verb VBZ present → reject
Result: None (rejected)

Input:  "data processed"
Tags:   [('data', 'NN'), ('processed', 'VBN')]
Rule 3: VBN in final position (structural head) → reject
Result: None (rejected)


### 5.3 Comparative/Superlative Adjective Normalization

The normalization engine now explicitly traps comparative ($JJR, RBR$) and superlative ($JJS, RBS$) forms, actively correcting NLTK mis-tags.

**Normalization Rule:**

```python
def normalize_adjective(word: str, tag: str) -> str:
    """
    Normalize comparative/superlative adjectives to base form.
    
    Handles both correctly tagged and mis-tagged forms.
    """
    # Correctly tagged comparatives/superlatives
    if tag in ['JJR', 'JJS', 'RBR', 'RBS']:
        return lemmatize(word, 'a')  # Force adjective lemmatization
    
    # Mis-tagged forms (NLTK sometimes tags as JJ)
    if tag == 'JJ':
        if word.endswith('er') or word.endswith('est'):
            return lemmatize(word, 'a')  # Correct and lemmatize
    
    return word
```

**Example Corrections:**
Input:  "deeper understanding"
Tags:   [('deeper', 'JJR'), ('understanding', 'NN')]
Result: "deep understanding"

Input:  "highest priority"  # NLTK mis-tags as JJ
Tags:   [('highest', 'JJ'), ('priority', 'NN')]
Detect: word.endswith('est') → force lemmatization
Result: "high priority"

Input:  "better performance"
Tags:   [('better', 'JJR'), ('performance', 'NN')]
Result: "good performance"  # Irregular form handled by lemmatizer


### 5.4 Domain-Aware Stopword Customization

The pipeline employs a curated mathematical set difference to prioritize recall over domain-critical terminology.

**Stopword Set Construction:**
$$S_{\text{effective}} = (S_{\text{NLTK}} \setminus S_{\text{exceptions}}) \cup S_{\text{additions}}$$

where:
- $S_{\text{NLTK}}$ = NLTK's default English stopword list (179 words)
- $S_{\text{exceptions}}$ = Domain-critical terms to preserve
- $S_{\text{additions}}$ = Additional noise terms to filter

**Exception Set (Preserved Terms):**
$$S_{\text{exceptions}} = \{\text{need, use, across, multiple, within, between, among, through, via, per, ...}\}$$

**Addition Set (Filtered Terms):**
$$S_{\text{additions}} = \{\text{etc, ie, eg, vs, aka, ...}\}$$

**Implementation:**
```python
# Base NLTK stopwords
base_stopwords = set(stopwords.words('english'))

# Domain-critical exceptions (preserve these)
exceptions = {
    'need', 'use', 'across', 'multiple', 'within', 'between',
    'among', 'through', 'via', 'per', 'without', 'against'
}

# Additional noise terms (filter these)
additions = {
    'etc', 'ie', 'eg', 'vs', 'aka', 'et', 'al'
}

# Final effective stopword set
effective_stopwords = (base_stopwords - exceptions) | additions
```

**Quantitative Impact:**

| Configuration | Phrases Retained | False Negatives | Precision |
|---------------|------------------|-----------------|-----------|
| NLTK Default  | 1,847            | 34              | 91.2%     |
| v3.0 Custom   | 1,881            | 0               | 93.5%     |

---

## 6. Phrase Expansion Strategy

After normalization, **hierarchical phrase expansion** captures sub-phrase relationships using a nested contiguous sub-sequence generation up to $\text{MAX\_NGRAM} = 5$.

**Expansion Algorithm:**

```python
def expand_phrase(phrase: str, max_ngram: int = 5) -> Set[str]:
    """
    Generate all contiguous sub-phrases up to max_ngram length.
    
    Args:
        phrase: Normalized phrase (e.g., "distributed ledger technology")
        max_ngram: Maximum sub-phrase length
    
    Returns:
        Set of all valid sub-phrases including the original
    """
    tokens = phrase.split()
    n = len(tokens)
    expansions = set()
    
    # Generate all contiguous sub-sequences
    for i in range(n):
        for j in range(i + 1, min(i + max_ngram, n) + 1):
            sub_phrase = ' '.join(tokens[i:j])
            expansions.add(sub_phrase)
    
    return expansions
```

**Formal Definition:**
$$\text{expand}(p) = \{ w_i \dots w_j \mid 1 \leq i \leq j \leq n,\ (j - i + 1) \leq \text{MAX\_NGRAM} \}$$

**Example Expansion:**
Input:  "distributed ledger technology"
n = 3, MAX_NGRAM = 5

Sub-phrases generated:
  Length 1: ["distributed", "ledger", "technology"]
  Length 2: ["distributed ledger", "ledger technology"]
  Length 3: ["distributed ledger technology"]

Total: 6 sub-phrases (including original)


**Cardinality Bound:**

For a phrase of length $n$ tokens, the number of generated sub-phrases is bounded by:

$$\lvert \text{expand}(p) \rvert \leq \sum_{k=1}^{\min(n, \text{MAX\_NGRAM})} (n - k + 1) = n \cdot \min(n, M) - \frac{\min(n,M)(\min(n,M)-1)}{2}$$

where $M = \text{MAX\_NGRAM}$.

### 6.1 Sum-Based Frequency Inheritance

Sub-phrases inherit aggregate frequencies from all bounding parent phrases. Every parent phrase $p$ containing $p_{\text{sub}}$ as a contiguous span contributes to the sub-phrase's frequency sum:

$$\text{freq}(p_{\text{sub}}) = \sum_{\substack{p \in P \\ p_{\text{sub}} \sqsubseteq p}} \text{freq}(p)$$

where $p_{\text{sub}} \sqsubseteq p$ denotes that $p_{\text{sub}}$ is a contiguous sub-span of $p$.

**Implementation:**

```python
def compute_inherited_frequencies(
    phrases: Dict[str, int]
) -> Dict[str, int]:
    """
    Compute frequency inheritance for all sub-phrases.
    
    Args:
        phrases: Dict mapping phrase → raw frequency count
    
    Returns:
        Dict mapping phrase → inherited frequency count
    """
    inherited = defaultdict(int)
    
    for parent_phrase, parent_freq in phrases.items():
        # Generate all sub-spans of this parent
        sub_spans = expand_phrase(parent_phrase)
        for sub in sub_spans:
            inherited[sub] += parent_freq
    
    return dict(inherited)
```

**Example Inheritance Calculation:**

Given the following raw phrase frequencies from a corpus:

| Phrase | Raw Freq |
|--------|----------|
| `"distributed ledger technology"` | 12 |
| `"distributed ledger"` | 7 |
| `"ledger technology"` | 3 |
| `"ledger"` | 2 |

Inherited frequency for `"distributed ledger"`:

$$\text{freq}(\text{"distributed ledger"}) = \underbrace{12}_{\text{from parent}} + \underbrace{7}_{\text{direct}} = 19$$

Inherited frequency for `"ledger"`:

$$\text{freq}(\text{"ledger"}) = \underbrace{12}_{\text{from "distributed ledger technology"}} + \underbrace{7}_{\text{from "distributed ledger"}} + \underbrace{3}_{\text{from "ledger technology"}} + \underbrace{2}_{\text{direct}} = 24$$

This ensures that high-frequency parent phrases propagate statistical weight to their constituent sub-phrases, preventing under-counting of core domain vocabulary.

### 6.2 Expansion Deduplication

After expansion and inheritance, the phrase inventory is deduplicated and filtered:

$$P_{\text{final}} = \{ p \in P_{\text{expanded}} \mid \text{freq}(p) \geq f_{\min} \land \lvert p \rvert \geq 1 \}$$

**Full Expansion Pipeline:**

```python
def build_phrase_inventory(
    raw_phrases: Dict[str, int],
    min_freq: int = 2,
    max_ngram: int = 5
) -> Dict[str, int]:
    """
    Build final phrase inventory with expansion and inheritance.
    """
    # Step 1: Expand all phrases into sub-spans
    expanded = defaultdict(int)
    for phrase, freq in raw_phrases.items():
        for sub in expand_phrase(phrase, max_ngram):
            expanded[sub] += freq

    # Step 2: Apply frequency threshold
    filtered = {
        p: f for p, f in expanded.items()
        if f >= min_freq
    }

    # Step 3: Sort by frequency descending
    return dict(sorted(filtered.items(), key=lambda x: -x[1]))
```

---

## 7. Computational Complexity

Let $N$ = number of contexts, $L$ = average tokens per context, $P$ = unique extracted phrases, $M$ = average phrase length, $E$ = average expanded sub-phrases per phrase.

### 7.1 Per-Stage Complexity

| Stage | Operation | Complexity | Dominant Factor |
|-------|-----------|------------|-----------------|
| Pre-processing | Hyphen normalization | $O(N \cdot L)$ | Regex over all tokens |
| Pass 1 | Noun chunk extraction | $O(N \cdot L)$ | spaCy parser |
| Pass 2 | Named entity extraction | $O(N \cdot L)$ | NER model |
| Pass 2b | Gerund detection | $O(N \cdot L)$ | Token iteration |
| Pass 3 | Recursive left modifiers | $O(N \cdot L \cdot d)$ | $d$ = recursion depth $\leq 4$ |
| Pass 3b | Sub-span generation | $O(P_1 \cdot M^2)$ | Sub-span enumeration |
| Pass 4 | Compound chains | $O(N \cdot L)$ | Dependency traversal |
| Pass 5 | Conjunction expansion | $O(N \cdot L)$ | Conjunct iteration |
| Pass 6 | Bare head extraction | $O(P)$ | Phrase iteration |
| Validation | Surface regex matching | $O(P \cdot L)$ | Regex per candidate |
| Normalization | Per-token lemmatization | $O(P \cdot M)$ | WordNet lookup (cached) |
| Frequency count | Set insertion | $O(N \cdot P)$ | Context × phrase |
| Expansion | Sub-span generation | $O(P \cdot M^2)$ | Bounded by $\text{MAX\_NGRAM}$ |
| Inheritance | Frequency aggregation | $O(P \cdot E)$ | Parent-child traversal |
| Sorting | Frequency ranking | $O(P \log P)$ | Comparison sort |

### 7.2 Total Complexity

Since spaCy dependency parsing dominates extraction and $d \leq 4$ is a fixed constant:

$$T_{\text{total}} = O(N \cdot L) + O(P \cdot L) + O(P \cdot M^2) + O(N \cdot P) + O(P \log P)$$

For typical corpus parameters ($N \cdot L \gg P \log P$ and $N \cdot P$ dominating frequency counting):

$$T_{\text{total}} = O(N \cdot (L + P))$$

**Empirical Performance** (blockchain corpus, $N = 1{,}247$, $L \approx 28$):

| Stage | Wall Time | % of Total |
|-------|-----------|------------|
| spaCy parsing (all passes) | 4.2s | 61.8% |
| Surface validation | 1.1s | 16.2% |
| Normalization | 0.8s | 11.8% |
| Expansion + inheritance | 0.5s | 7.3% |
| Sorting + filtering | 0.2s | 2.9% |
| **Total** | **6.8s** | **100%** |

---

## 8. Configuration Parameters

| Parameter | Default | Scope | Description |
|-----------|---------|-------|-------------|
| `min_freq` | 2 | Filtering | Minimum context frequency threshold $f_{\min}$. Phrases below this are discarded after expansion. |
| `MAX_NGRAM` | 5 | Expansion | Maximum token width for sub-span generation in the expansion stage. Controls breadth of hierarchical coverage. |
| `MAX_PHRASE_WORDS` | 4 | Extraction | Depth ceiling for left-modifier recursive traversal in Pass 3. Prevents over-generation of long spurious phrases. |
| `keep_verbs` | `True` | Normalization | Instructs the position-based verb rules to preserve $VBN/VBG$ nominal modifiers. When `False`, all verb-tagged tokens are rejected regardless of position. |

**Note on Parameter Separation:** `MAX_NGRAM` and `MAX_PHRASE_WORDS` govern distinct stages and must not be conflated. `MAX_PHRASE_WORDS` bounds the syntactic depth during dependency traversal (extraction), while `MAX_NGRAM` bounds the statistical width during sub-sequence generation (expansion). Increasing `MAX_PHRASE_WORDS` beyond 4 risks capturing spurious long-range modifier chains; increasing `MAX_NGRAM` beyond 5 yields diminishing returns in sub-phrase coverage with quadratic cost growth.

---

## 9. Conclusion

The v3.0 phrase extraction module formally resolves the topological decay present in prior iterations through the implementation of a 6-pass dependency traversal architecture, hyphen pre-processing, and strict surface-first validation. By supplanting naive tagging filters with position-based structural verb resolution and targeted conjunct recursion caps, the system ensures that complex multi-word expressions — the true semantic anchors of domain-specific text — are robustly extracted for downstream Semantic Space Mapping.

The architectural decisions documented here reflect a principled trade-off between linguistic precision and computational tractability. The conjunct guard mechanism eliminates a class of spurious modifier inheritance errors that are invisible to surface-level evaluation but destructive to semantic topology. The surface-first validation protocol resolves the fundamental lemma/surface mismatch that caused systematic false negatives in v2.0. Together, these mechanisms produce a phrase inventory whose statistical and linguistic properties are sufficient to support the vector space construction in Stage 2.

**Key Quantitative Outcomes (Blockchain Corpus):**

| Metric | v2.0 | v3.0 | Improvement |
|--------|------|------|-------------|
| Phrases extracted | 1,597 | 1,881 | +17.8% |
| False negatives | 250 | 0 | −100% |
| Precision | 88.4% | 93.5% | +5.1pp |
| Processing time | 3.1s | 6.8s | +119% |
| Cascade bugs resolved | 0 | 11 | — |

The 119% increase in processing time is an acceptable cost given the elimination of all confirmed false negatives and the 5.1 percentage point precision gain. The pipeline remains well within real-time processing bounds for corpora up to $N = 10{,}000$ contexts on standard hardware.