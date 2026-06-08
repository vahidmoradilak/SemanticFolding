from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from loguru import logger

from src.models.data_models import Triple

from .config import OUTPUT_DIR, HippoRagConfig


def export_triples_to_csv(
    triples: List[Triple],
    config: HippoRagConfig,
) -> Path:
    """
    Export triples to a CSV file under data/HippoRAG2/output.

    Columns:
      - subject
      - predicate
      - object
      - source_chunk_id
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{config.dataset}_triples.csv"

    logger.info("Writing {} triples to CSV at {}", len(triples), out_path)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object", "source_chunk_id"])
        for t in triples:
            writer.writerow([t.subject, t.predicate, t.object, t.source_chunk_id])

    logger.info("Finished writing triples CSV for dataset='{}'", config.dataset)
    return out_path


__all__ = ["export_triples_to_csv"]

