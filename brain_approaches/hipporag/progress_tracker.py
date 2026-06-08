"""
WAL-style progress tracker for triple extraction.

Tracks progress of triple extraction per dataset with write-ahead logging:
- Before sending API request: log chunk_id, timestamp, status="pending"
- After receiving result: log chunk_id, timestamp, status="completed", triple_count
- Appends to final output files incrementally
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger

from src.models.data_models import Triple

from .config import OUTPUT_DIR


class ProgressTracker:
    """
    WAL-style progress tracker for triple extraction.

    Maintains a progress log file per dataset that tracks:
    - Before API call: chunk_id, timestamp, status="pending"
    - After API call: chunk_id, timestamp, status="completed|failed", triple_count, error (if any)

    Saves API call results in batches of 100 per file: batch_0000.json, batch_0001.json, ...
    (No per-chunk files; legacy doc_*_chunk_*.json are removed on init.)
    """

    def __init__(self, dataset_name: str):
        """
        Initialize progress tracker for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., "musique")
        """
        self.dataset_name = dataset_name
        self.progress_file = OUTPUT_DIR / f"{dataset_name}_progress.wal"
        self.results_dir = OUTPUT_DIR / f"{dataset_name}_api_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Remove legacy per-chunk files so only batch_*.json (100 records each) remain
        self._remove_legacy_per_chunk_files()

        # Batch tracking: accumulate 100 API results per file
        self.batch_size = 100
        self.current_batch: List[Dict] = []
        self.batch_index = self._next_batch_index()

        # Ensure progress file exists
        if not self.progress_file.exists():
            self.progress_file.write_text("", encoding="utf-8")
            logger.info("Created progress file: {}", self.progress_file)

    def _remove_legacy_per_chunk_files(self) -> None:
        """Remove legacy doc_*_chunk_*.json files; we only use batch_XXXX.json (100 records each)."""
        removed = 0
        for p in self.results_dir.iterdir():
            if p.is_file() and p.suffix == ".json" and p.name.startswith("doc_") and "chunk_" in p.name:
                p.unlink()
                removed += 1
        if removed:
            logger.info("Removed {} legacy per-chunk API result files from {}", removed, self.results_dir)

    def _next_batch_index(self) -> int:
        """Return next batch index from existing batch_XXXX.json files (for resume)."""
        max_idx = -1
        for p in self.results_dir.iterdir():
            if p.is_file() and p.name.startswith("batch_") and p.suffix == ".json":
                try:
                    idx = int(p.stem.split("_")[1])
                    max_idx = max(max_idx, idx)
                except (ValueError, IndexError):
                    pass
        return max_idx + 1

    def log_before_request(self, chunk_id: str, passage_preview: str = "") -> None:
        """
        Log before sending API request (WAL entry).

        Args:
            chunk_id: Unique identifier for the chunk
            passage_preview: Optional preview of passage content (first 100 chars)
        """
        entry = {
            "chunk_id": chunk_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
            "passage_preview": passage_preview[:100] if passage_preview else "",
        }

        with self.progress_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.debug("Progress: {} -> pending", chunk_id)

    def log_after_response(
        self,
        chunk_id: str,
        triples: List[Triple],
        raw_response: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log after receiving API response (WAL entry).

        Args:
            chunk_id: Unique identifier for the chunk
            triples: List of extracted triples (empty if failed)
            raw_response: Optional raw API response text
            error: Optional error message if request failed
        """
        status = "failed" if error else "completed"
        entry = {
            "chunk_id": chunk_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "triple_count": len(triples),
            "error": error,
        }

        with self.progress_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        # Batch API call results (100 per file)
        result_data = {
            "chunk_id": chunk_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "triples": [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "source_chunk_id": t.source_chunk_id,
                }
                for t in triples
            ],
            "raw_response": raw_response[:1000] if raw_response else None,  # Truncate for storage
            "error": error,
        }

        self.current_batch.append(result_data)

        # Write batch when it reaches batch_size
        if len(self.current_batch) >= self.batch_size:
            self._flush_batch()

        logger.info(
            "Progress: {} -> {} ({} triples)",
            chunk_id,
            status,
            len(triples),
        )

    def _flush_batch(self) -> None:
        """Write current batch to file and reset."""
        if not self.current_batch:
            return

        batch_file = self.results_dir / f"batch_{self.batch_index:04d}.json"
        with batch_file.open("w", encoding="utf-8") as f:
            json.dump(self.current_batch, f, indent=2, ensure_ascii=False)

        logger.debug("Wrote batch {} with {} API results to {}", self.batch_index, len(self.current_batch), batch_file)
        self.current_batch = []
        self.batch_index += 1

    def flush(self) -> None:
        """Flush any remaining batched results to file."""
        if self.current_batch:
            self._flush_batch()

    def append_to_final_output(self, triples: List[Triple], output_file: Path) -> None:
        """
        Append triples to the final output CSV file.

        Note: The output file should already have a header written by the caller.
        This function only appends data rows.

        Args:
            triples: List of triples to append
            output_file: Path to final CSV output file (must exist with header)
        """
        import csv

        if not triples:
            return

        with output_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for t in triples:
                writer.writerow([t.subject, t.predicate, t.object, t.source_chunk_id])

        logger.debug("Appended {} triples to {}", len(triples), output_file)

    def get_progress_summary(self) -> Dict[str, int]:
        """
        Read progress file and return summary statistics.

        Returns:
            Dictionary with counts: pending, completed, failed, total_triples
        """
        if not self.progress_file.exists():
            return {"pending": 0, "completed": 0, "failed": 0, "total_triples": 0}

        pending = 0
        completed = 0
        failed = 0
        total_triples = 0

        with self.progress_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    status = entry.get("status", "")
                    if status == "pending":
                        pending += 1
                    elif status == "completed":
                        completed += 1
                        total_triples += entry.get("triple_count", 0)
                    elif status == "failed":
                        failed += 1
                except json.JSONDecodeError:
                    logger.warning("Failed to parse progress entry: {}", line[:100])

        return {
            "pending": pending,
            "completed": completed,
            "failed": failed,
            "total_triples": total_triples,
        }

    def get_processed_chunk_ids(self) -> Set[str]:
        """
        Return chunk_ids that have already been processed (completed or failed).

        Used for resume: skip these chunks and only run extraction for the rest.
        """
        if not self.progress_file.exists():
            return set()

        processed: Set[str] = set()
        with self.progress_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    status = entry.get("status", "")
                    if status in ("completed", "failed"):
                        cid = entry.get("chunk_id")
                        if cid:
                            processed.add(cid)
                except json.JSONDecodeError:
                    continue
        return processed


__all__ = ["ProgressTracker"]
