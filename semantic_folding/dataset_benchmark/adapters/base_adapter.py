"""
Base Dataset Adapter — Abstract interface for all dataset converters.

Each adapter is responsible for:
  1. Downloading the raw dataset from its source (HuggingFace, GitHub, etc.)
  2. Converting it to MuSiQue-like JSONL format with this structure:
     {
       "id": "<query_id>",
       "question": "<query text>",
       "answer": "<ground truth answer>",
       "paragraphs": [
         {
           "idx": 0,
           "title": "<passage title>",
           "paragraph_text": "<passage text>",
           "is_supporting": true|false
         },
         ...
       ]
     }
  3. Reporting dataset-specific stats and metadata.

The generic benchmark runner (generic_benchmark.py) consumes this format
identically to the original MuSiQue benchmark.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class BaseDatasetAdapter(ABC):
    """Abstract base for all dataset adapters."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Lowercase short name used in CLI / output paths."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for reports."""
        ...

    @property
    @abstractmethod
    def default_subset(self) -> str:
        """Which subset to use by default (e.g., 'dev', 'train', 'test')."""
        ...

    @abstractmethod
    def download(self, output_dir: Path) -> Path:
        """
        Download the raw dataset to output_dir.
        Returns the path to the raw data directory.
        """
        ...

    @abstractmethod
    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        """
        Convert raw dataset to MuSiQue-like JSONL.
        Returns the path to the converted JSONL file.
        """
        ...

    def get_recommended_params(self) -> Dict[str, Any]:
        """Override per-dataset if non-default parameters are required."""
        return {
            "grid_size": 64,
            "spreading_steps": 1,
            "top_percent": 0.10,
            "weighting": "idf",
            "smoothing_sigma": 1.5,
            "morton": True,
            "min_word_length": 3,
            "min_freq": 1,
            "keep_verbs": True,
            "top_k": 5,
            "tsne_perplexity": 50,
            "tsne_iter": 1000,
        }

    def validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Sanity check: entry has required fields with valid values."""
        required = ["id", "question", "paragraphs"]
        for k in required:
            if k not in entry:
                return False
        if not isinstance(entry["paragraphs"], list) or len(entry["paragraphs"]) == 0:
            return False
        for p in entry["paragraphs"]:
            if "paragraph_text" not in p or "is_supporting" not in p:
                return False
        return True

    def count_stats(self, jsonl_path: Path) -> Dict[str, int]:
        """Compute basic stats for a converted JSONL file."""
        n_entries = 0
        n_paragraphs = 0
        n_gold = 0
        unique_paragraphs = set()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                n_entries += 1
                for p in e.get("paragraphs", []):
                    key = (p.get("title", ""), p.get("paragraph_text", ""))
                    if key in unique_paragraphs:
                        continue
                    unique_paragraphs.add(key)
                    n_paragraphs += 1
                    if p.get("is_supporting", False):
                        n_gold += 1
        return {
            "num_queries": n_entries,
            "num_unique_paragraphs": n_paragraphs,
            "num_gold_passages": n_gold,
        }
