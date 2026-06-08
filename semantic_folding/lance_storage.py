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
            ('fingerprint_vector', pa.list_(pa.float32())),  # Flattened 16x16 = 256 dims
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
            ('fingerprint_vector', pa.list_(pa.float32())),  # Flattened 16x16 = 256 dims
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