from __future__ import annotations

from typing import List
import argparse

from loguru import logger

from .config import HippoRagConfig, OUTPUT_DIR, list_available_datasets
from .extract_triples import extract_triples_for_corpus
from .load_dataset import load_corpus, summarize_dataset
from .load_to_memgraph import load_triples_to_memgraph


def _prompt_dataset_selection() -> List[str]:
    """
    Ask the user on the command line which dataset(s) to process.

    Returns a list of dataset names (matching keys from DATASET_FILES).
    """

    available = list(list_available_datasets().keys())
    print("Available datasets:")
    for i, name in enumerate(available, start=1):
        print(f"  [{i}] {name}")
    print("Enter dataset numbers separated by comma (e.g. 1,3) or 'all':")

    choice = input("> ").strip().lower()
    if choice in {"all", "*"}:
        return available

    selected: List[str] = []
    for part in choice.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            print(f"Ignoring invalid selection: {part!r}")
            continue
        if 1 <= idx <= len(available):
            selected.append(available[idx - 1])
        else:
            print(f"Ignoring out-of-range selection: {part!r}")

    # Fallback: if nothing valid was selected, process nothing explicitly.
    return selected


def main() -> None:
    """
    Entry point for the HippoRAG2 → triples → Memgraph pipeline.

    For now, this performs the first step:
    - Ask which datasets to process.
    - For each selected dataset, print a short summary of the corpus/queries
      using the loader functions.

    Later, this will be extended to:
    - Extract triples with OpenRouter.
    - Normalize & export triples.
    - Load triples into Memgraph.
    """

    parser = argparse.ArgumentParser(
        description="HippoRAG2 → triples → Memgraph pipeline (OpenRouter + Memgraph)."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Comma-separated list of dataset names to process "
            "(e.g. 'sample,musique') or 'all'. "
            "If omitted, you will be prompted interactively."
        ),
    )
    args = parser.parse_args()

    if args.datasets:
        available = list(list_available_datasets().keys())
        if args.datasets.lower() in {"all", "*"}:
            selected = available
        else:
            requested = [x.strip() for x in args.datasets.split(",") if x.strip()]
            invalid = [x for x in requested if x not in available]
            if invalid:
                raise SystemExit(
                    f"Unknown dataset(s): {', '.join(invalid)}. "
                    f"Available: {', '.join(available)}"
                )
            selected = requested
    else:
        selected = _prompt_dataset_selection()
    if not selected:
        print("No valid datasets selected. Exiting.")
        return

    print()
    for name in selected:
        cfg = HippoRagConfig(dataset=name)

        logger.info("=== Processing dataset '{}' ===", name)
        print(summarize_dataset(cfg))

        # 1) Load corpus
        corpus = load_corpus(cfg)
        logger.info("Loaded {} corpus passages for dataset='{}'", len(corpus), name)

        # 2) Extract triples with OpenRouter (includes chunking, progress tracking, and incremental CSV export)
        triples = extract_triples_for_corpus(corpus, dataset_name=name)
        if not triples:
            logger.warning("No triples extracted for dataset='{}'; skipping Memgraph load.", name)
            continue

        # 3) Triples are already exported incrementally during extraction
        # The final CSV is at data/HippoRAG2/output/<dataset>_triples.csv
        csv_path = OUTPUT_DIR / f"{name}_triples.csv"
        progress_file = OUTPUT_DIR / f"{name}_progress.wal"

        # 4) Load triples into Memgraph
        load_triples_to_memgraph(triples)

        print(
            f"Dataset '{name}': extracted {len(triples)} triples, "
            f"written to '{csv_path}', progress tracked in '{progress_file}', "
            f"and loaded into Memgraph."
        )


if __name__ == "__main__":
    main()

