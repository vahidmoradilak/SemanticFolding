#!/usr/bin/env python3
"""
LanceDB Integration for Semantic Folding Pipeline

Provides fast vector similarity search for semantic fingerprints using LanceDB.
Stores and retrieves both phrase and document fingerprints with metadata.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

import loguru
from loguru import logger

# Try to import required dependencies
try:
    import lancedb
    import numpy as np
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LanceDB dependencies not available: {e}")
    logger.warning("Install with: uv add lancedb pyarrow")
    LANCEDB_AVAILABLE = False


class LanceStorage:
    """
    LanceDB storage for semantic fingerprints with vector similarity search.

    Supports both phrase fingerprints and document fingerprints with metadata.
    """

    def __init__(self, db_path: Union[str, Path], connection_uri: Optional[str] = None):
        """
        Initialize LanceDB storage.

        Args:
            db_path: Path to LanceDB database directory
            connection_uri: Optional connection URI (for remote LanceDB)
        """
        if not LANCEDB_AVAILABLE:
            raise RuntimeError("LanceDB dependencies not available. Install with: uv add lancedb pyarrow")

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        try:
            if connection_uri:
                self.db = lancedb.connect(connection_uri)
                logger.info(f"Connected to remote LanceDB: {connection_uri}")
            else:
                self.db = lancedb.connect(str(self.db_path))
                logger.success(f"Connected to local LanceDB: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            raise

        # Initialize tables
        self._init_tables()

    def _init_tables(self):
        """Initialize LanceDB tables for fingerprints and metadata."""
        # Table for phrase fingerprints
        self.phrase_table_name = "phrase_fingerprints"
        self._create_phrase_table()

        # Table for document fingerprints
        self.doc_table_name = "document_fingerprints"
        self._create_document_table()

        logger.success("LanceDB tables initialized")

    def _create_phrase_table(self):
        """Create table for phrase fingerprints."""
        schema = pa.schema([
            ('phrase', pa.string()),
            ('fingerprint_vector', pa.list_(pa.float32())),  # Flattened grid_size*grid_size dims
            ('grid_size', pa.int32()),
            ('frequency', pa.int32()),
            ('context_count', pa.int32()),  # How many contexts this phrase appears in
            ('metadata', pa.string()),  # JSON string with additional metadata
        ])

        try:
            self.db.create_table(self.phrase_table_name, schema=schema, exist_ok=True)
            logger.debug(f"Created/reused phrase fingerprints table")
        except Exception as e:
            logger.warning(f"Could not create phrase table: {e}")

    def _create_document_table(self):
        """Create table for document fingerprints."""
        schema = pa.schema([
            ('context_id', pa.string()),
            ('title', pa.string()),
            ('text', pa.string()),
            ('fingerprint_vector', pa.list_(pa.float32())),  # Flattened grid_size*grid_size dims
            ('grid_size', pa.int32()),
            ('matched_phrases', pa.int32()),
            ('total_phrases', pa.int32()),
            ('coverage', pa.float32()),
            ('metadata', pa.string()),  # JSON string with additional metadata
        ])

        try:
            self.db.create_table(self.doc_table_name, schema=schema, exist_ok=True)
            logger.debug(f"Created/reused document fingerprints table")
        except Exception as e:
            logger.warning(f"Could not create document table: {e}")

    def store_phrase_fingerprints(self, fingerprints: Dict[str, np.ndarray],
                                metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store phrase fingerprints in LanceDB.

        Args:
            fingerprints: Dict mapping phrase -> fingerprint matrix
            metadata: Optional metadata for each phrase

        Returns:
            Number of fingerprints stored
        """
        if not fingerprints:
            logger.warning("No phrase fingerprints to store")
            return 0

        table = self.db.open_table(self.phrase_table_name)
        grid_size = int(np.sqrt(fingerprints[next(iter(fingerprints.keys()))].size))

        data = []
        for phrase, fingerprint in fingerprints.items():
            # Flatten the 2D fingerprint to 1D vector
            fingerprint_flat = fingerprint.flatten().astype(np.float32)

            # Prepare metadata
            phrase_meta = metadata.get(phrase, {}) if metadata else {}
            phrase_meta.update({
                'phrase': phrase,
                'shape': fingerprint.shape,
                'grid_size': grid_size
            })

            data.append({
                'phrase': phrase,
                'fingerprint_vector': fingerprint_flat.tolist(),
                'grid_size': grid_size,
                'frequency': phrase_meta.get('frequency', 0),
                'context_count': phrase_meta.get('context_count', 0),
                'metadata': json.dumps(phrase_meta, ensure_ascii=False)
            })

        try:
            table.add(data)
            logger.success(f"Stored {len(data)} phrase fingerprints in LanceDB")
            return len(data)
        except Exception as e:
            logger.error(f"Failed to store phrase fingerprints: {e}")
            return 0

    def store_document_fingerprints(self, fingerprints: Dict[str, np.ndarray],
                                  contexts: Dict[str, str],
                                  metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
        """
        Store document fingerprints in LanceDB.

        Args:
            fingerprints: Dict mapping context_id -> fingerprint matrix
            contexts: Dict mapping context_id -> full text
            metadata: Optional metadata for each document

        Returns:
            Number of fingerprints stored
        """
        if not fingerprints:
            logger.warning("No document fingerprints to store")
            return 0

        table = self.db.open_table(self.doc_table_name)
        grid_size = int(np.sqrt(fingerprints[next(iter(fingerprints.keys()))].size))

        data = []
        for context_id, fingerprint in fingerprints.items():
            # Flatten the 2D fingerprint to 1D vector
            fingerprint_flat = fingerprint.flatten().astype(np.float32)

            # Get context text
            context_text = contexts.get(context_id, "")

            # Extract title from text (assuming format "title: content")
            title = context_id
            text = context_text
            if ": " in context_text:
                title_part, text_part = context_text.split(": ", 1)
                title = title_part.strip()
                text = text_part.strip()

            # Prepare metadata
            doc_meta = metadata.get(context_id, {}) if metadata else {}
            doc_meta.update({
                'context_id': context_id,
                'title': title,
                'grid_size': grid_size
            })

            data.append({
                'context_id': context_id,
                'title': title,
                'text': text,
                'fingerprint_vector': fingerprint_flat.tolist(),
                'grid_size': grid_size,
                'matched_phrases': doc_meta.get('matched_phrases', 0),
                'total_phrases': doc_meta.get('total_doc_phrases', 0),
                'coverage': doc_meta.get('coverage', 0.0),
                'metadata': json.dumps(doc_meta, ensure_ascii=False)
            })

        try:
            table.add(data)
            logger.success(f"Stored {len(data)} document fingerprints in LanceDB")
            return len(data)
        except Exception as e:
            logger.error(f"Failed to store document fingerprints: {e}")
            return 0

    def retrieve_similar_phrases(self, query_fingerprint: np.ndarray,
                               top_k: int = 10,
                               distance_metric: str = "cosine") -> List[Dict[str, Any]]:
        """
        Find phrases with similar fingerprints to the query.

        Args:
            query_fingerprint: Query fingerprint matrix (2D)
            top_k: Number of similar phrases to return
            distance_metric: Distance metric ("cosine", "l2", "ip")

        Returns:
            List of similar phrases with scores and metadata
        """
        try:
            table = self.db.open_table(self.phrase_table_name)

            # Flatten query fingerprint
            query_vector = query_fingerprint.flatten().astype(np.float32)

            # Perform vector search
            results = table.search(query_vector, vector_column_name="fingerprint_vector") \
                          .metric(distance_metric) \
                          .limit(top_k) \
                          .to_pandas()

            # Format results
            similar_phrases = []
            for _, row in results.iterrows():
                metadata = json.loads(row['metadata']) if row['metadata'] else {}

                similar_phrases.append({
                    'phrase': row['phrase'],
                    'score': float(row['_distance']),  # Lower is more similar for cosine
                    'frequency': int(row['frequency']),
                    'context_count': int(row['context_count']),
                    'metadata': metadata
                })

            logger.info(f"Found {len(similar_phrases)} similar phrases")
            return similar_phrases

        except Exception as e:
            logger.error(f"Failed to retrieve similar phrases: {e}")
            return []

    def retrieve_similar_documents(self, query_fingerprint: np.ndarray,
                                 top_k: int = 10,
                                 distance_metric: str = "cosine") -> List[Dict[str, Any]]:
        """
        Find documents with similar fingerprints to the query.

        Args:
            query_fingerprint: Query fingerprint matrix (2D)
            top_k: Number of similar documents to return
            distance_metric: Distance metric ("cosine", "l2", "ip")

        Returns:
            List of similar documents with scores and metadata
        """
        try:
            table = self.db.open_table(self.doc_table_name)

            # Flatten query fingerprint
            query_vector = query_fingerprint.flatten().astype(np.float32)

            # Perform vector search
            results = table.search(query_vector, vector_column_name="fingerprint_vector") \
                          .metric(distance_metric) \
                          .limit(top_k) \
                          .to_pandas()

            # Format results
            similar_docs = []
            for _, row in results.iterrows():
                metadata = json.loads(row['metadata']) if row['metadata'] else {}

                similar_docs.append({
                    'context_id': row['context_id'],
                    'title': row['title'],
                    'text': row['text'][:200] + "..." if len(row['text']) > 200 else row['text'],
                    'score': float(row['_distance']),  # Lower is more similar for cosine
                    'matched_phrases': int(row['matched_phrases']),
                    'coverage': float(row['coverage']),
                    'metadata': metadata
                })

            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs

        except Exception as e:
            logger.error(f"Failed to retrieve similar documents: {e}")
            return []

    def get_phrase_fingerprint(self, phrase: str) -> Optional[np.ndarray]:
        """Retrieve fingerprint for a specific phrase."""
        try:
            table = self.db.open_table(self.phrase_table_name)
            result = table.search().where(f"phrase = '{phrase}'").limit(1).to_pandas()

            if not result.empty:
                fingerprint_vector = result.iloc[0]['fingerprint_vector']
                grid_size = int(result.iloc[0]['grid_size'])
                return np.array(fingerprint_vector).reshape((grid_size, grid_size))

        except Exception as e:
            logger.error(f"Failed to retrieve phrase fingerprint for '{phrase}': {e}")

        return None

    def get_document_fingerprint(self, context_id: str) -> Optional[np.ndarray]:
        """Retrieve fingerprint for a specific document."""
        try:
            table = self.db.open_table(self.doc_table_name)
            result = table.search().where(f"context_id = '{context_id}'").limit(1).to_pandas()

            if not result.empty:
                fingerprint_vector = result.iloc[0]['fingerprint_vector']
                grid_size = int(result.iloc[0]['grid_size'])
                return np.array(fingerprint_vector).reshape((grid_size, grid_size))

        except Exception as e:
            logger.error(f"Failed to retrieve document fingerprint for '{context_id}': {e}")

        return None

    def get_passage_by_id(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full passage information by context ID."""
        try:
            table = self.db.open_table(self.doc_table_name)
            result = table.search().where(f"context_id = '{context_id}'").limit(1).to_pandas()

            if not result.empty:
                row = result.iloc[0]
                metadata = json.loads(row['metadata']) if row['metadata'] else {}

                return {
                    'context_id': row['context_id'],
                    'title': row['title'],
                    'text': row['text'],
                    'matched_phrases': int(row['matched_phrases']),
                    'total_phrases': int(row['total_phrases']),
                    'coverage': float(row['coverage']),
                    'metadata': metadata
                }

        except Exception as e:
            logger.error(f"Failed to retrieve passage '{context_id}': {e}")

        return None

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}

        try:
            # Phrase table stats
            phrase_table = self.db.open_table(self.phrase_table_name)
            phrase_count = phrase_table.count_rows()
            stats['phrase_fingerprints'] = phrase_count

            # Document table stats
            doc_table = self.db.open_table(self.doc_table_name)
            doc_count = doc_table.count_rows()
            stats['document_fingerprints'] = doc_count

            stats['total_fingerprints'] = phrase_count + doc_count

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats['error'] = str(e)

        return stats

    # ------------------------------------------------------------------
    # ANN document index (WS-A: LanceDB-backed retrieval)
    # ------------------------------------------------------------------
    #
    # The base `document_fingerprints` table stores vectors as variable-length
    # `list<float32>`, which LanceDB cannot use for vector search.  ANN search
    # requires a fixed-length vector column, so the document index is kept in a
    # dedicated table (`doc_fingerprints_ann`) whose `fingerprint_vector` is a
    # `FixedSizeList<float32, grid_size**2>`.  Building the table happens once at
    # index time (Phase 1); queries only ever read from it.
    # ------------------------------------------------------------------

    DOC_ANN_TABLE = "doc_fingerprints_ann"

    def build_document_index(
        self,
        doc_fp_dir: Union[str, Path],
        table_name: str = None,
    ) -> Dict[str, Any]:
        """
        Build (or rebuild) the ANN document index from Step-5 outputs.

        Reads ``doc_fingerprints.npz``, ``doc_fingerprints_meta.json`` and
        ``doc_fingerprints_stats.json`` and stores every document fingerprint
        (flattened float32, Morton order as stored) in a dedicated ANN table
        with a fixed-length vector column.  An HNSW (IVF_HNSW_FLAT) index with
        cosine metric is created on top so queries can use approximate nearest
        neighbour search.

        Parameters
        ----------
        doc_fp_dir : Path
            Step-5 output directory containing the doc fingerprint files.
        table_name : str, optional
            Table name override (defaults to ``doc_fingerprints_ann``).

        Returns
        -------
        Dict[str, Any]
            Summary with ``num_docs``, ``dim``, ``grid_size``,
            ``build_seconds``, ``table_name``, ``use_morton``.
        """
        import time

        doc_fp_dir = Path(doc_fp_dir)
        npz_path = doc_fp_dir / "doc_fingerprints.npz"
        meta_path = doc_fp_dir / "doc_fingerprints_meta.json"

        if not npz_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Document fingerprint files missing in {doc_fp_dir} "
                f"(need doc_fingerprints.npz + doc_fingerprints_meta.json)"
            )

        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        archive = np.load(str(npz_path))
        matrix = archive["fingerprints"].astype(np.float32)  # (N, grid_size²)
        n_docs, dim = matrix.shape
        grid_size = int(meta.get("grid_size", int(np.sqrt(dim))))
        use_morton = bool(meta.get("use_morton", True))

        doc_to_row: Dict[str, int] = meta.get("doc_to_row", {})
        doc_ids = [None] * n_docs
        for doc_id, row_idx in doc_to_row.items():
            if 0 <= row_idx < n_docs:
                doc_ids[row_idx] = doc_id
        # Fall back to positional ids if meta mapping is missing rows
        for i in range(n_docs):
            if doc_ids[i] is None:
                doc_ids[i] = f"doc_{i:06d}"

        table_name = table_name or self.DOC_ANN_TABLE
        try:
            self.db.drop_table(table_name)
        except Exception:
            pass

        schema = pa.schema([
            ("context_id", pa.string()),
            ("text", pa.string()),
            ("grid_size", pa.int32()),
            ("fingerprint_vector", pa.list_(pa.float32(), dim)),  # FixedSizeList
        ])

        data = [
            {
                "context_id": doc_ids[i],
                "text": "",
                "grid_size": grid_size,
                "fingerprint_vector": matrix[i].tolist(),
            }
            for i in range(n_docs)
        ]

        t0 = time.perf_counter()
        table = self.db.create_table(table_name, data=data, schema=schema)
        table.create_index(
            metric="cosine",
            vector_column_name="fingerprint_vector",
            index_type="IVF_HNSW_FLAT",
        )
        build_seconds = time.perf_counter() - t0

        logger.success(
            f"ANN document index built: {n_docs} docs, dim={dim}, "
            f"grid_size={grid_size} ({build_seconds:.2f}s)"
        )

        return {
            "num_docs": n_docs,
            "dim": dim,
            "grid_size": grid_size,
            "use_morton": use_morton,
            "build_seconds": round(build_seconds, 4),
            "table_name": table_name,
        }

    def search_documents(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        metric: str = "cosine",
        table_name: str = None,
        exact: bool = False,
        nprobes: int = 64,
    ) -> List[Tuple[str, float]]:
        """
        Approximate (or exact) nearest-neighbour search over the ANN doc index.

        Parameters
        ----------
        query_vector : np.ndarray
            Flattened float32 query fingerprint ``(grid_size²,)``.
        top_k : int
            Maximum number of neighbours to return.
        metric : str
            Distance metric used at query time (default ``"cosine"``).
        table_name : str, optional
            ANN table name (defaults to ``doc_fingerprints_ann``).
        exact : bool
            When True, run a full scan instead of using the ANN index.
        nprobes : int
            Number of IVF partitions to probe during ANN search (default 64).

        Returns
        -------
        List[Tuple[str, float]]
            ``[(context_id, similarity)]`` sorted descending by similarity.
            Distance is converted to similarity (``1 - distance`` for cosine) at
            this boundary so downstream ranking code stays distance-agnostic.
        """
        table = self.db.open_table(table_name or self.DOC_ANN_TABLE)

        qvec = np.asarray(query_vector, dtype=np.float32).ravel().tolist()

        builder = table.search(qvec, vector_column_name="fingerprint_vector")
        builder = builder.select(["context_id"])  # avoid transferring the full vector column
        builder = builder.nprobes(0 if exact else nprobes)
        results = builder.metric(metric).limit(top_k).to_list()

        out: List[Tuple[str, float]] = []
        for row in results:
            context_id = row.get("context_id", "")
            distance = float(row.get("_distance", 0.0))
            similarity = 1.0 - distance  # cosine distance → similarity
            out.append((context_id, similarity))

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def get_ann_index_stats(self, table_name: str = None) -> Dict[str, Any]:
        """Return row count and rough on-disk size for the ANN doc table."""
        stats: Dict[str, Any] = {}
        try:
            table = self.db.open_table(table_name or self.DOC_ANN_TABLE)
            stats["num_docs"] = table.count_rows()
        except Exception as e:
            stats["error"] = str(e)
        return stats

    def close(self):
        """Close the database connection."""
        try:
            # LanceDB connections are typically auto-managed
            logger.info("LanceDB connection closed")
        except Exception as e:
            logger.warning(f"Error closing LanceDB connection: {e}")


def create_storage(db_path: Union[str, Path], connection_uri: Optional[str] = None) -> LanceStorage:
    """
    Factory function to create LanceStorage instance.

    Args:
        db_path: Path to database directory
        connection_uri: Optional remote connection URI

    Returns:
        LanceStorage instance
    """
    return LanceStorage(db_path, connection_uri)


# CLI interface for testing
def main():
    """Command-line interface for testing LanceDB storage."""
    import argparse

    parser = argparse.ArgumentParser(description="LanceDB storage for semantic fingerprints")
    parser.add_argument("--db_path", required=True, help="Database directory path")
    parser.add_argument("--action", choices=['stats', 'test'], default='stats', help="Action to perform")

    args = parser.parse_args()

    logger.info("LanceDB Storage CLI")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Action: {args.action}")

    try:
        storage = create_storage(args.db_path)

        if args.action == 'stats':
            stats = storage.get_database_stats()
            logger.info("Database Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

        elif args.action == 'test':
            # Basic connectivity test
            logger.success("LanceDB connection test successful")

        storage.close()

    except Exception as e:
        logger.error(f"LanceDB operation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()