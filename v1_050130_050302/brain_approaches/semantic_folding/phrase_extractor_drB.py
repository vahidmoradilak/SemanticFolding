r"""
Phrase Extractor - Step 1 of Semantic Folding Pipeline
Date: 1405/01/14 | 2026/04/03

This module constitutes the first stage of the Semantic Folding pipeline. 
Its theoretical objective is to map a raw unstructured text corpus $C$ into a 
finite vocabulary of meaningful semantic features $\mathcal{V}$. 

These extracted phrases act as the fundamental basis vectors (dimensions) 
for the subsequent Semantic Space Mapping (Step 3) and Fingerprint Generation (Step 4). 
By isolating noun chunks, named entities, and structurally valid n-grams, we ensure 
that the resulting topological space captures high-signal semantic anchors while 
discarding low-information topological noise.
"""

import argparse
import json
import sys, re, os
from pathlib import Path
from typing import Set, Dict, Tuple, List, Optional, Any
from collections import Counter as CounterType, defaultdict


# ── Logging setup ─────────────────────────────────────────────────────────────
from lib import get_logger
logger = get_logger("phrase_extractor")


# ── Library imports ────────────────────────────────────────────────────────────
from lib import expand_phrases, normalize_hyphens

# ── spaCy bootstrap ───────────────────────────────────────────────────────────
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.debug("spaCy imported successfully")
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.success("spaCy model 'en_core_web_sm' loaded")
    except OSError:
        logger.warning("spaCy model not found — run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spaCy not installed — falling back to NLTK extraction")
    SPACY_AVAILABLE = False

# NLTK only needed when spaCy is unavailable
if not SPACY_AVAILABLE:
    logger.debug("Importing NLTK fallback tokenizer and POS tagger")
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag


# ─────────────────────────────────────────────────────────────────────────────
# Fallback extractor (NLTK)
# ─────────────────────────────────────────────────────────────────────────────

def extract_raw_phrases_fallback(text: str) -> Set[str]:
    """
    NLTK-based bigram/trigram extractor used when spaCy is unavailable.

    Strategy
    --------
    POS-tag every token, then slide a 2-token window looking for
    adjective/noun + noun patterns — a lightweight proxy for noun chunks.

    Supported head POS tags for position-0:  JJ, VBN, NN, NNP
    Required POS prefix  for position-1:     N* (any noun tag)

    Parameters
    ----------
    text : str
        Lower-cased input sentence.

    Returns
    -------
    Set[str]
        Raw phrase strings (original casing from tokenizer).
    """
    logger.debug(f"[FALLBACK] Extracting phrases from text ({len(text)} chars)")
    phrases: Set[str] = set()

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    logger.debug(f"[FALLBACK] Tokenized → {len(tagged)} tokens")

    for i in range(len(tagged) - 1):
        w1, t1 = tagged[i]
        w2, t2 = tagged[i + 1]
        if t1 in ('JJ', 'VBN', 'NN', 'NNP') and t2.startswith('N'):
            phrase = f"{w1} {w2}"
            phrases.add(phrase)
            logger.debug(f"[FALLBACK][ADD] '{phrase}'  ({t1}+{t2})")

    logger.info(f"[FALLBACK] Extracted {len(phrases)} raw phrases")
    return phrases


# ─────────────────────────────────────────────────────────────────────────────
# Left-modifier collector (spaCy)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_left_modifiers(doc) -> list[tuple[str, str]]:
    """
    Walk every noun/propn token in the doc and collect left-modifier phrases.

    Algorithm
    ---------
    For each token whose dep_ is in the allowed head-dep set, inspect its
    left children with deps in {amod, compound, nmod, nummod}.  Build a
    phrase spanning ``doc[child.i : token.i + 1]`` and recurse into the
    child to capture deeper modifier chains
    (e.g. 'unique' → 'personal' → 'characteristic').

    Conjunct guard
    --------------
    When a token is reached via a ``conj`` edge (e.g. 'counseling' is a
    conjunct of 'settings'), it must NOT inherit modifiers that belong to
    the head ('clinical' → 'settings').  The ``is_conjunct`` flag enforces
    this: if True, only modifiers whose ``.head`` is the conjunct token
    itself are accepted — anything pointing up to the head noun is skipped,
    preventing spurious phrases like 'clinical counseling'.

    Top-level loop guard
    --------------------
    Tokens with ``dep_ == 'conj'`` are skipped at the top-level loop
    because they are already reached via the explicit conjunct-recursion
    block inside ``traverse()``.  Processing them twice would produce
    duplicate entries.

    Depth cap
    ---------
    Recursion is hard-capped at ``MAX_PHRASE_WORDS`` levels to prevent
    runaway chains on pathological parse trees.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A fully parsed spaCy document.

    Returns
    -------
    list of (phrase_text, head_token_text) tuples
        ``phrase_text``      — the full modifier+head string
        ``head_token_text``  — the syntactic head token (for diagnostics)
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    # dep_ tags that qualify a token as a valid phrase head
    HEAD_DEPS = {
        'nsubj', 'dobj', 'nsubjpass', 'attr',
        'appos', 'conj', 'ROOT', 'compound', 'amod', 'nmod',
    }
    # dep_ tags on children that signal a left modifier
    MODIFIER_DEPS = {'amod', 'compound', 'nmod', 'nummod'}
    # POS tags allowed on modifier children
    MODIFIER_POS  = {'NOUN', 'PROPN', 'ADJ', 'NUM'}

    def traverse(t, depth: int, is_conjunct: bool = False) -> None:
        logger.debug(
            f"[MOD] traverse token='{t.text}' dep={t.dep_} pos={t.pos_} "
            f"depth={depth} is_conjunct={is_conjunct}"
        )

        # hard cap — avoids runaway recursion on deep modifier chains
        if depth > MAX_PHRASE_WORDS:
            logger.debug(f"[MOD][SKIP] depth cap reached at token '{t.text}'")
            return

        # verbs and auxiliaries are never phrase heads in this context
        if t.pos_ in ('VERB', 'AUX'):
            logger.debug(f"[MOD][SKIP] verb/aux token '{t.text}' (pos={t.pos_})")
            return

        # restrict traversal to syntactically plausible noun-phrase heads;
        # this prevents the recursion from wandering into prepositional
        # objects, adverbials, and other non-nominal dependents
        if t.dep_ not in HEAD_DEPS:
            logger.debug(
                f"[MOD][SKIP] dep '{t.dep_}' not in HEAD_DEPS for token '{t.text}'"
            )
            return

        left_children = sorted(t.lefts, key=lambda x: x.i)
        logger.debug(
            f"[MOD] '{t.text}' has {len(left_children)} left children: "
            f"{[c.text for c in left_children]}"
        )

        for child in left_children:
            if child.dep_ in MODIFIER_DEPS and child.pos_ in MODIFIER_POS:

                # conjunct guard — skip modifiers whose syntactic head is
                # not the current conjunct token (they belong to the head noun)
                if is_conjunct and child.head != t:
                    logger.debug(
                        f"[MOD][CONJUNCT GUARD] skipping '{child.text}' "
                        f"(child.head='{child.head.text}' ≠ t='{t.text}')"
                    )
                    continue

                phrase_tokens = list(doc[child.i: t.i + 1])
                phrase = ' '.join(tok.text for tok in phrase_tokens)

                if phrase not in seen:
                    seen.add(phrase)
                    results.append((phrase, t.text))
                    logger.debug(
                        f"[MOD][ADD] '{phrase}'  (head='{t.text}' "
                        f"child_dep={child.dep_} child_pos={child.pos_})"
                    )
                else:
                    logger.debug(f"[MOD][DUP] '{phrase}' already seen — skipped")

                # recurse into the modifier to catch deeper chains,
                # e.g. child='personal' → grandchild='unique' →
                # final phrase 'unique personal characteristic'
                traverse(child, depth + 1)

        # explicitly recurse into conjuncts with the guard flag active;
        # this ensures conjuncts are always processed with modifier-
        # inheritance protection regardless of tree depth
        for child in t.children:
            if child.dep_ == 'conj' and child.pos_ in ('NOUN', 'PROPN'):
                logger.debug(
                    f"[MOD][CONJ] recursing into conjunct '{child.text}' "
                    f"from head '{t.text}'"
                )
                traverse(child, depth + 1, is_conjunct=True)

    for token in doc:
        # skip conjunct tokens at the top level — they are reached via
        # the explicit conjunct-recursion block above; starting a fresh
        # traversal from them would process them twice
        if token.dep_ != 'conj':
            traverse(token, 1)

    logger.info(f"[MOD] Collected {len(results)} left-modifier phrases")
    return results
def expand_conjunctions(doc) -> list[str]:
    """
    Expand adjective modifiers across coordinated noun conjunctions.

    Algorithm
    ---------
    1. Only process the leftmost noun in each conjunction group to avoid
       duplicate emissions from every member of the chain.
    2. Only consider strict adjective modifiers (dep_=amod, pos_=ADJ,
       left of head) on the HEAD token.
    3. For each conjunct:
         - If it already has ANY pre-nominal modifier (amod, compound,
           nummod, nmod, npadvmod) → emit it standalone; do not inherit
           the head's adjective (it has its own semantic coloring).
         - If it has no pre-nominal modifier → propagate the head's
           adjective to it, producing e.g. 'clinical settings' and
           'clinical counseling' from 'clinical settings and counseling'.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A fully parsed spaCy document.

    Returns
    -------
    list[str]
        Expanded phrase strings (surface form, original casing).
    """
    # dep_ tags that count as pre-nominal modifiers on a conjunct
    MODIFIER_DEPS = {"amod", "compound", "nummod", "nmod", "npadvmod"}

    expanded: list[str] = []

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue

        conjuncts = [t for t in token.conjuncts if t.pos_ in ("NOUN", "PROPN")]
        if not conjuncts:
            continue

        # only process the leftmost token in the conjunction group;
        # all other members are reached via the conjuncts list below
        if any(t.i < token.i for t in conjuncts):
            logger.debug(
                f"[CONJ][SKIP] '{token.text}' is not leftmost in its "
                f"conjunction group — skipping top-level processing"
            )
            continue

        # collect strict adjective modifiers on the head noun
        head_modifiers = [
            child for child in token.children
            if child.dep_ == "amod"
            and child.pos_ == "ADJ"
            and child.i < token.i
        ]

        if not head_modifiers:
            logger.debug(
                f"[CONJ][SKIP] '{token.text}' has no amod+ADJ left-children "
                f"— nothing to propagate"
            )
            continue

        logger.debug(
            f"[CONJ] head='{token.text}'  "
            f"modifiers={[m.text for m in head_modifiers]}  "
            f"conjuncts={[c.text for c in conjuncts]}"
        )

        # emit head phrase(s) — one per adjective modifier
        for mod in head_modifiers:
            phrase = f"{mod.text} {token.text}"
            expanded.append(phrase)
            logger.debug(f"[CONJ][HEAD] '{phrase}'")

        # handle each conjunct
        for conjunct in conjuncts:
            conjunct_own_premods = [
                c for c in conjunct.children
                if c.dep_ in MODIFIER_DEPS and c.i < conjunct.i
            ]

            if conjunct_own_premods:
                # conjunct has its own modifier — emit standalone,
                # do not inherit the head's adjective
                expanded.append(conjunct.text)
                logger.debug(
                    f"[CONJ][STANDALONE] '{conjunct.text}'  "
                    f"(own premods: {[c.text for c in conjunct_own_premods]})"
                )
            else:
                # no modifier — propagate head's adjective(s)
                for mod in head_modifiers:
                    phrase = f"{mod.text} {conjunct.text}"
                    expanded.append(phrase)
                    logger.debug(
                        f"[CONJ][INHERIT] '{phrase}'  "
                        f"(inherited from head='{token.text}')"
                    )

    logger.info(f"[CONJ] Expanded {len(expanded)} conjunction phrases")
    return expanded


# ─────────────────────────────────────────────────────────────────────────────
# Corpus processor
# ─────────────────────────────────────────────────────────────────────────────

def process_corpus_with_expansion(
    corpus_path: Path,
    use_spacy: bool = True,
    min_freq: int = 2,
    filter_generic: bool = True,
    min_word_length: int = 3,
    keep_verbs: bool = True,
) -> Tuple[CounterType[str], Dict[str, List[str]]]:
    """
    Full pipeline: raw extraction → expansion → normalization → frequency filter.

    Pipeline stages
    ---------------
    1. Raw extraction  — spaCy noun chunks / named entities, or NLTK bigrams.
    2. Expansion       — ``expand_phrases`` validates surface forms and
                         generates sub-phrase candidates.
    3. Normalization   — handled inside ``expand_phrases`` before returning.
    4. Context mapping — each surviving phrase is mapped to its source
                         context ID (one entry per corpus line).
    5. Frequency filter — phrases appearing in fewer than ``min_freq``
                          distinct contexts are discarded.

    Parameters
    ----------
    corpus_path : Path
        CSV-like file where each line is ``context_id,text``.
    use_spacy : bool
        Prefer spaCy extraction when available.
    min_freq : int
        Minimum number of distinct context IDs a phrase must appear in
        to be retained in the final vocabulary.
    filter_generic : bool
        Passed through to ``expand_phrases`` to drop stopword-heavy phrases.
    min_word_length : int
        Minimum character length for individual words inside a phrase.
    keep_verbs : bool
        Reserved for future use; passed to downstream expansion logic.

    Returns
    -------
    final_vocabulary : CounterType[str]
        phrase → document frequency count.
    final_mapping : Dict[str, List[str]]
        phrase → sorted list of context IDs it appeared in.
    """
    raw_phrase_contexts: Dict[str, Set[str]] = defaultdict(set)

    logger.info(f"Opening corpus: {corpus_path}")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip() or ',' not in line:
                logger.debug(f"[CORPUS] Line {i} skipped (empty or no comma)")
                continue

            try:
                ctx_id, text = line.split(',', 1)
                ctx_id = ctx_id.strip()
                text_original = text.strip()
                text_lower = text_original.lower()
            except ValueError:
                logger.debug(f"[CORPUS] Line {i} skipped (split error)")
                continue

            logger.debug(f"[CORPUS] Line {i} | ctx='{ctx_id}' | text='{text_original[:60]}...'")

            # ── Hyphen normalization ──────────────────────────────────────────
            # Replace intra-word hyphens (e.g. 'rule-based') with spaces so
            # that compound terms are tokenized as multi-word phrases rather
            # than being split into three tokens: word, '-', word.
            # text_clean is used for ALL downstream processing; text_original
            # is kept only for logging/diagnostics.
            text_clean = normalize_hyphens(text_original)
            text_clean_lower = normalize_hyphens(text_lower)

            if text_clean != text_original:
                logger.debug(
                    f"[CORPUS] Line {i} | hyphen normalization applied: "
                    f"'{text_original[:60]}' → '{text_clean[:60]}'"
                )

            # ── Stage 1: raw candidate extraction ────────────────────────────
            if use_spacy and SPACY_AVAILABLE:
                doc = nlp(text_clean)          # spaCy sees hyphen-free text
                raw_phrases = extract_raw_phrases_spacy(doc)
            else:
                raw_phrases = extract_raw_phrases_fallback(text_clean_lower)

            logger.debug(f"[CORPUS] Line {i} | {len(raw_phrases)} raw phrases extracted")

            if not raw_phrases:
                logger.debug(f"[CORPUS] Line {i} | no raw phrases — skipping expansion")
                continue

            # ── Stage 2 & 3: expansion + normalization ────────────────────────
            # expand_phrases receives raw (un-normalized) phrases and handles
            # surface validation before normalizing internally.
            # text_clean is passed so that context validation (substring match)
            # works against the same hyphen-free surface form that the extractor saw.
            valid_sub_phrases = expand_phrases(
                list(raw_phrases),
                context_text=text_clean,       # must match what extractor saw
                filter_generic=filter_generic,
                min_word_length=min_word_length,
            )

            logger.debug(
                f"[CORPUS] Line {i} | {len(valid_sub_phrases)} phrases "
                f"survived expansion/normalization"
            )

            # ── Stage 4: map to context ID ────────────────────────────────────
            for phrase in valid_sub_phrases:
                raw_phrase_contexts[phrase].add(ctx_id)
                logger.debug(f"[CORPUS][MAP] '{phrase}' → ctx '{ctx_id}'")

    logger.info(
        f"[CORPUS] Extraction complete — "
        f"{len(raw_phrase_contexts)} unique phrases before frequency filter"
    )

    # ── Stage 5: frequency filtering ─────────────────────────────────────────
    final_vocabulary: CounterType[str] = CounterType()
    final_mapping: Dict[str, List[str]] = {}

    dropped = 0
    for phrase, ctx_set in raw_phrase_contexts.items():
        doc_freq = len(ctx_set)
        if doc_freq >= min_freq:
            final_vocabulary[phrase] = doc_freq
            final_mapping[phrase] = sorted(list(ctx_set))
            logger.debug(f"[FREQ][KEEP] '{phrase}'  freq={doc_freq}")
        else:
            dropped += 1
            logger.debug(f"[FREQ][DROP] '{phrase}'  freq={doc_freq} < min={min_freq}")

    logger.info(
        f"[FREQ] Kept {len(final_vocabulary)} phrases, "
        f"dropped {dropped} below min_freq={min_freq}"
    )

    return final_vocabulary, final_mapping


# ─────────────────────────────────────────────────────────────────────────────
# Statistics and output
# ─────────────────────────────────────────────────────────────────────────────

def save_phrases(
    phrase_counts: CounterType[str],
    phrase_to_contexts: Dict[str, List[str]],
    output_path: Path,
) -> None:
    r"""
    Persist the extracted vocabulary $\mathcal{V}$ and its context mapping to disk.

    Outputs
    -------
    ``<output_path>.csv``
        Two-column CSV: ``phrase, document_frequency``, sorted by frequency
        descending (most_common order).

    ``<output_path.parent>/phrase_to_contexts.json``
        Bipartite adjacency map $phrase \rightarrow [context\_ids]$ used by
        downstream steps (Step 2 context retrieval, Step 3 space mapping).

    Parameters
    ----------
    phrase_counts : CounterType[str]
        phrase → document frequency, as returned by
        ``process_corpus_with_expansion``.
    phrase_to_contexts : Dict[str, List[str]]
        phrase → sorted list of context IDs.
    output_path : Path
        Base path; ``.csv`` suffix is appended automatically.
    """
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── vocabulary CSV ────────────────────────────────────────────────────────
    vocab_csv_path = output_path.with_suffix('.csv')
    logger.info(f"Saving vocabulary to: {vocab_csv_path}")
    with open(vocab_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for phrase, count in phrase_counts.most_common():
            writer.writerow([phrase, count])
            logger.debug(f"[SAVE][VOCAB] '{phrase}' → {count}")

    logger.info(f"Wrote {len(phrase_counts)} phrases to {vocab_csv_path}")

    # ── context mapping JSON ──────────────────────────────────────────────────
    mapping_path = output_path.parent / "phrase_to_contexts.json"
    logger.info(f"Saving context mapping to: {mapping_path}")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(phrase_to_contexts, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved {len(phrase_counts)} phrases and context mappings.")


def print_statistics(phrase_counts: CounterType[str]) -> None:
    """
    Print distributional statistics for the extracted vocabulary to stdout.

    Displays
    --------
    - Total unique phrases and total occurrence count.
    - Average frequency per phrase.
    - N-gram length distribution (unigrams, bigrams, trigrams, …).
    - Top-10 most frequent phrases.

    Parameters
    ----------
    phrase_counts : CounterType[str]
        phrase → document frequency counter.
    """
    if not phrase_counts:
        logger.warning("No phrases available — skipping statistics.")
        return

    total_phrases = len(phrase_counts)
    total_occurrences = sum(phrase_counts.values())
    avg_freq = total_occurrences / total_phrases

    logger.debug(
        f"[STATS] total_phrases={total_phrases}  "
        f"total_occurrences={total_occurrences}  "
        f"avg_freq={avg_freq:.2f}"
    )

    # n-gram length distribution
    length_dist: Dict[int, int] = defaultdict(int)
    for phrase in phrase_counts:
        length_dist[len(phrase.split())] += 1

    logger.debug(f"[STATS] length distribution: {dict(sorted(length_dist.items()))}")

    # ── stdout report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("      Vocabulary Distributional Statistics")
    print("=" * 50)
    print(f"  Total unique phrases:             {total_phrases:>10,}")
    print(f"  Total occurrences (sum of freqs): {total_occurrences:>10,}")
    print(f"  Avg. frequency per phrase:        {avg_freq:>10.2f}")

    print("\n  Distribution by phrase length (n-grams):")
    for length, count in sorted(length_dist.items()):
        pct = (count / total_phrases) * 100
        print(f"    {length}-grams: {count:>7,} phrases ({pct:.2f}%)")

    print("\n  Top 10 Most Frequent Phrases:")
    for i, (phrase, count) in enumerate(phrase_counts.most_common(10), start=1):
        print(f"    {i:>2}. '{phrase}'  (freq: {count:,})")

    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point for Step 1 of the Semantic Folding pipeline.

    Parses arguments, runs ``process_corpus_with_expansion``, persists
    outputs via ``save_phrases``, and optionally prints statistics.

    Exit codes
    ----------
    0 — success
    1 — corpus file not found
    """
    parser = argparse.ArgumentParser(
        description="Extract meaningful phrases (Semantic Vectors) from a corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--corpus', type=Path, required=True,
        help='Path to corpus file (format: context_id,context_text)',
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Directory to save vocabulary.csv and phrase_to_contexts.json',
    )
    parser.add_argument(
        '--keep-verbs', action='store_true', default=False,
        help='Do not strip verbs during normalization',
    )
    parser.add_argument(
        '--no-spacy', action='store_true',
        help='Force fallback NLTK n-gram extraction',
    )
    parser.add_argument(
        '--no-filter-generic', action='store_true',
        help='Keep generic single words (lowers signal-to-noise ratio)',
    )
    parser.add_argument(
        '--min-word-length', type=int, default=3,
        help='Minimum character length $L_{min}$ for single words (default: 3)',
    )
    parser.add_argument(
        '--min-freq', type=int, default=2,
        help='Sparsity filter threshold $min\\_freq$ (default: 2)',
    )
    parser.add_argument(
        '--stats', action='store_true',
        help='Print detailed distributional statistics after extraction',
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.corpus.exists():
        logger.error(f"Corpus file not found: {args.corpus}")
        sys.exit(1)

    use_spacy = not args.no_spacy
    filter_generic = not args.no_filter_generic

    logger.info("── Step 1 Configuration ──────────────────────────────────")
    logger.info(f"  Corpus:          {args.corpus}")
    logger.info(f"  Output dir:      {args.output_dir}")
    logger.info(f"  Extractor:       {'spaCy' if use_spacy and SPACY_AVAILABLE else 'NLTK fallback'}")
    logger.info(f"  Filter generic:  {filter_generic}")
    logger.info(f"  Min freq:        {args.min_freq}")
    logger.info(f"  Min word length: {args.min_word_length}")
    logger.info(f"  Keep verbs:      {args.keep_verbs}")
    logger.info("──────────────────────────────────────────────────────────")

    phrase_counts, phrase_to_contexts = process_corpus_with_expansion(
        corpus_path=args.corpus,
        use_spacy=use_spacy,
        min_freq=args.min_freq,
        filter_generic=filter_generic,
        min_word_length=args.min_word_length,
        keep_verbs=args.keep_verbs,
    )

    output_vocab_path = args.output_dir / 'vocabulary'
    save_phrases(phrase_counts, phrase_to_contexts, output_vocab_path)

    if args.stats:
        print_statistics(phrase_counts)

    logger.success("Phrase extraction (Step 1) complete.")


MAX_PHRASE_WORDS = 4
VALID_HEAD_DEPS = {'nsubj', 'dobj', 'nsubjpass', 'attr', 'appos', 'conj', 'ROOT'}


def _collect_compound_chains(doc) -> list[str]:
    """
    Collect compound noun chains (e.g. 'personality traits') by following
    'compound' dependency edges. Rejects chains containing any VERB token.
    """
    results = []
    seen = set()
    for token in doc:
        logger.debug(
            f"[COMPOUND] token='{token.text}' dep='{token.dep_}' "
            f"pos='{token.pos_}' head='{token.head.text}' head_dep='{token.head.dep_}'"
        )
        if token.dep_ == 'compound':
            head = token.head
            if head.pos_ in ('VERB', 'AUX'):
                logger.debug(f"[COMPOUND SKIP] '{token.text} {head.text}' — head is VERB/AUX")
                continue
            if head.dep_ not in VALID_HEAD_DEPS:
                logger.debug(
                    f"[COMPOUND SKIP] '{token.text} {head.text}' "
                    f"— head dep '{head.dep_}' not in VALID_HEAD_DEPS"
                )
                continue
            phrase = f"{token.text} {head.text}"
            if phrase not in seen:
                seen.add(phrase)
                results.append(phrase)
    return results


def extract_raw_phrases_spacy(doc) -> list[str]:
    """
    Extract candidate phrases from a spaCy Doc using multiple strategies.

    Strategies (in order):
      1. Noun chunks        — spaCy NP detector; possessive prefixes stripped.
      2. Named entities     — multi-word proper nouns / domain terms.
      2b. Standalone gerunds — VBG tokens acting as nominal heads.
      3. Left modifiers     — adjective/compound chains from noun heads.
      3b. Left-anchored sub-spans — sub-phrases from long noun chunks.
      4. Compound chains    — direct compound-dep pairs.
      5. Conjunction expansion — shared modifiers across coordinated nouns.
      6. Bare head nouns    — rightmost word of every multi-word phrase.

    Args:
        doc: a spaCy Doc (already parsed with a full pipeline model).

    Returns:
        list of raw (un-normalised) phrase strings, deduplicated.
    """
    raw: dict[str, str] = {}  # phrase -> first source that added it

    def add_phrase(phrase: str, source: str):
        if phrase not in raw:
            raw[phrase] = source
            logger.debug(f"[ADD] '{phrase}' from {source}")

    def _strip_possessive_prefix(text: str) -> str:
        """Remove a leading possessive owner from a noun chunk string."""
        stripped = re.sub(r"^\w+[\u2019']s?\s+", "", text)
        return stripped if stripped else text

    # ── constant sets ─────────────────────────────────────────────────────────

    NOMINAL_DEPS = {'nsubj', 'dobj', 'pobj', 'attr', 'ROOT', 'pcomp'}

    _DISCOURSE_MARKER_NOUNS = {
        'addition', 'contrast', 'conclusion', 'summary', 'consequence',
        'result', 'fact', 'general', 'particular', 'short', 'brief',
        'response', 'reply', 'turn', 'return', 'comparison', 'contrast',
        'sum', 'total', 'whole', 'end', 'spite', 'lieu',
    }

    _LIGHT_VERB_OBJECTS = {
        'light', 'place', 'effect', 'impact', 'account', 'part',
        'advantage', 'note', 'heed', 'care', 'exception', 'issue',
        'harm', 'shape', 'form', 'action', 'consideration',
    }
    _LIGHT_VERBS = {
        'shed', 'take', 'make', 'give', 'have', 'get', 'pay',
        'put', 'bring', 'come', 'go', 'do', 'run',
    }

    # ── step 1: noun chunks ───────────────────────────────────────────────────

    def _is_clausal_gerund_head(chunk) -> bool:
        """True when chunk root is a VBG with its own object/subject (gerund clause)."""
        root = chunk.root
        if root.tag_ != 'VBG':
            return False
        return any(
            c.dep_ in ('dobj', 'nsubjpass', 'nsubj', 'attr', 'oprd')
            for c in root.children
        )

    def _is_discourse_marker_chunk(chunk) -> bool:
        """True for single-token chunks whose root is a discourse-marker noun under ADP."""
        if len(chunk) > 1:
            return False
        root = chunk.root
        return root.lemma_.lower() in _DISCOURSE_MARKER_NOUNS and root.head.pos_ == 'ADP'

    def _is_light_verb_object(chunk) -> bool:
        """True for single-token chunks whose root is a light-verb object."""
        if len(chunk) > 1:
            return False
        root = chunk.root
        return (
            root.lemma_.lower() in _LIGHT_VERB_OBJECTS
            and root.head.lemma_.lower() in _LIGHT_VERBS
            and root.head.pos_ == 'VERB'
        )

    def is_bad_verb(tok, chunk) -> bool:
        """True if tok is a finite verb that should disqualify the chunk."""
        if tok.pos_ not in ('VERB', 'AUX'):
            return False
        if tok.tag_ == 'VBG' and tok == chunk.root and tok.dep_ in NOMINAL_DEPS:
            return False
        return True

    for chunk in doc.noun_chunks:
        logger.debug(
            f"[CHUNK] '{chunk.text}' | root='{chunk.root.text}' "
            f"dep='{chunk.root.dep_}' tag='{chunk.root.tag_}'"
        )

        if any(is_bad_verb(tok, chunk) for tok in chunk):
            logger.debug(f"[CHUNK SKIP] '{chunk.text}' — contains finite verb")
            continue
        if all(tok.pos_ == 'PRON' for tok in chunk):
            logger.debug(f"[CHUNK SKIP] '{chunk.text}' — all pronouns")
            continue
        if (chunk.root.dep_ == 'pobj'
                and chunk.root.pos_ == 'VERB'
                and chunk.root.tag_ != 'VBG'):
            logger.debug(f"[CHUNK SKIP] '{chunk.text}' — pobj conjugated verb root")
            continue
        if _is_clausal_gerund_head(chunk):
            logger.debug(f"[CHUNK SKIP] '{chunk.text}' — clausal gerund head")
            continue
        if _is_discourse_marker_chunk(chunk):
            logger.debug(f"[CHUNK SKIP] '{chunk.text}' — discourse marker")
            continue
        if _is_light_verb_object(chunk):
            logger.debug(f"[CHUNK SKIP] '{chunk.text}' — light-verb object")
            continue

        chunk_text = _strip_possessive_prefix(chunk.text)
        if len(chunk_text.split()) <= MAX_PHRASE_WORDS:
            add_phrase(chunk_text, "noun_chunks")

    # ── step 2: named entities ────────────────────────────────────────────────
    for ent in doc.ents:
        if len(ent) == 1 and ent[0].pos_ == 'ADJ':
            logger.debug(f"[NER SKIP] '{ent.text}' — single-token ADJ")
            continue
        if len(ent) <= MAX_PHRASE_WORDS:
            add_phrase(ent.text, "named_entities")

    # ── step 2b: standalone gerunds ───────────────────────────────────────────
    for token in doc:
        if token.tag_ != 'VBG' or token.dep_ not in NOMINAL_DEPS:
            continue
        if any(chunk.root == token for chunk in doc.noun_chunks):
            continue
        if any(c.dep_ in ('nsubj', 'nsubjpass') for c in token.children):
            continue
        if any(c.dep_ in ('dobj', 'attr', 'oprd') for c in token.children):
            continue
        gov = token.head
        if gov.pos_ == 'ADP' and gov.head.pos_ in ('VERB', 'ADJ'):
            logger.debug(f"[GERUND SKIP] '{token.text}' — participial/adverbial gerund")
            continue
        add_phrase(token.text, "gerunds")

    # ── step 3: left modifiers ────────────────────────────────────────────────
    for phrase, _ in _collect_left_modifiers(doc):
        add_phrase(phrase, "left_modifiers")

    # ── step 3b: left-anchored modifier sub-spans ─────────────────────────────
    MIN_SUBSPAN_LEN = 2
    MAX_SUBSPAN_LEN = 4
    _NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}

    for chunk in doc.noun_chunks:
        tokens = [
            t for t in chunk
            if t.dep_ not in ('det', 'preconj') and t.tag_ not in ('DT', 'PDT')
        ]
        if len(tokens) < 3:
            continue

        head = chunk.root
        modifiers = [t for t in tokens if t.i < head.i]
        if len(modifiers) < 2:
            continue

        logger.debug(
            f"[3b] chunk='{chunk.text}' head='{head.text}' "
            f"modifiers={[t.text for t in modifiers]}"
        )

        for end in range(MIN_SUBSPAN_LEN, min(len(modifiers) + 1, MAX_SUBSPAN_LEN + 1)):
            sub_tokens = modifiers[:end]
            last_tok = sub_tokens[-1]
            if last_tok.tag_ not in _NOUN_TAGS:
                logger.debug(
                    f"[SUBSPAN SKIP] '{' '.join(t.text for t in sub_tokens)}' "
                    f"— last token '{last_tok.text}' is {last_tok.tag_}, not noun"
                )
                continue
            sub_text = ' '.join(t.text for t in sub_tokens)
            logger.debug(f"[MODIFIER SUBSPAN] '{sub_text}' from '{chunk.text}'")
            add_phrase(sub_text, "modifier_subspan")

    # ── step 4: compound chains ───────────────────────────────────────────────
    for phrase in _collect_compound_chains(doc):
        add_phrase(phrase, "compound_chains")

    # ── step 5: conjunction expansion ────────────────────────────────────────
    for phrase in expand_conjunctions(doc):
        add_phrase(phrase, "conjunctions")

    # ── step 6: bare head nouns ───────────────────────────────────────────────
    multi_word = {p for p in raw if len(p.split()) > 1}
    for phrase in sorted(multi_word):
        add_phrase(phrase.split()[-1], "bare_heads")

    return list(raw.keys())


if __name__ == "__main__":
    main()
    