from __future__ import annotations

from typing import List

from loguru import logger

from src.models.data_models import Triple
from src.storage.memgraph_client import MemgraphClient


def load_triples_to_memgraph(triples: List[Triple]) -> None:
    """
    Load triples directly into Memgraph using the shared MemgraphClient.

    Assumes Memgraph is reachable at the default URI configured in MemgraphClient.
    """

    if not triples:
        logger.warning("No triples provided for Memgraph load; skipping.")
        return

    logger.info("Connecting to Memgraph to insert {} triples", len(triples))
    with MemgraphClient() as client:
        client.bulk_insert_triples(triples)
        stats = client.get_stats()

    logger.info(
        "Memgraph load complete. Graph now has {} nodes and {} edges.",
        stats.get("node_count"),
        stats.get("edge_count"),
    )


__all__ = ["load_triples_to_memgraph"]

