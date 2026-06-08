from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import HippoRagConfig


@dataclass
class CorpusEntry:
    """Typed view over one entry in the *_corpus.json files."""

    idx: int
    title: str
    text: str


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_corpus(config: HippoRagConfig) -> List[CorpusEntry]:
    """
    Load the retrieval corpus for a given dataset.

    Each item is expected to have at least 'idx', 'title' and 'text' fields,
    matching the HippoRAG2 dataset format described in data/HippoRAG2/ReadMe.md.
    """

    raw = _read_json(config.corpus_path)
    if not isinstance(raw, Sequence):
        raise ValueError(f"Corpus JSON must be a list, got {type(raw)} from {config.corpus_path}")

    entries: List[CorpusEntry] = []
    for i, item in enumerate(raw):
        if not isinstance(item, Dict):
            raise ValueError(f"Corpus item must be an object, got {type(item)}: {item!r}")
        try:
            # Some HippoRAG2 corpora include an explicit 'idx', others do not.
            # Fall back to the list index when 'idx' is missing.
            raw_idx = item.get("idx", i)
            idx = int(raw_idx)
            title = str(item.get("title", ""))
            text = str(item["text"])
        except KeyError as e:
            raise KeyError(f"Missing expected key {e} in corpus item: {item!r}") from e

        entries.append(CorpusEntry(idx=idx, title=title, text=text))

    return entries


def load_queries(config: HippoRagConfig) -> Optional[List[Dict[str, Any]]]:
    """
    Load the optional query file for a dataset, if present.

    The JSON format follows the structure documented in the HippoRAG2 README.
    This function returns the raw dictionaries for maximum flexibility.
    """

    path = config.queries_path
    if not path.exists():
        return None

    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Query JSON must be a list, got {type(data)} from {path}")
    return data  # type: ignore[return-value]


def summarize_dataset(config: HippoRagConfig) -> str:
    """
    Return a short human-readable summary of the selected dataset.
    """

    corpus = load_corpus(config)
    queries = load_queries(config)
    num_docs = len(corpus)
    num_queries = len(queries) if queries is not None else 0
    return (
        f"Dataset '{config.dataset}': "
        f"{num_docs} corpus passages, "
        f"{num_queries} queries (if non-zero)."
    )


if __name__ == "__main__":
    # Lightweight manual test / demo for the first task:
    # load a dataset and print a basic summary so you can run:
    #   python -m src.hipporag.load_dataset
    # from the project root to verify everything is wired correctly.
    default_cfg = HippoRagConfig()
    print(summarize_dataset(default_cfg))

