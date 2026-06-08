from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional
import yaml

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "HippoRAG2" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "HippoRAG2" / "output"
FREE_MODELS_YAML = PROJECT_ROOT / "data" / "HippoRAG2" / "free-models.yml"

# Default OpenRouter model for HippoRAG-style extraction.
# Using free tier model to avoid credit limits.
# Alternatives: "z-ai/glm-4.5-air:free", "alibaba/tongyi-deepresearch-30b-a3b:free"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v3.1:free"


# Map logical dataset names to their corpus and (optional) query files.
DATASET_FILES: Mapping[str, Dict[str, str]] = {
    "sample": {
        "corpus": "sample_corpus.json",
        "queries": "sample.json",
    },
    "musique": {
        "corpus": "musique_corpus.json",
        "queries": "musique.json",
    },
    "hotpotqa": {
        "corpus": "hotpotqa_corpus.json",
        "queries": "hotpotqa.json",
    },
    "2wikimultihopqa": {
        "corpus": "2wikimultihopqa_corpus.json",
        "queries": "2wikimultihopqa.json",
    },
}


@dataclass
class HippoRagConfig:
    """
    Central configuration for the HippoRAG2 → triples → Memgraph pipeline.

    This first task focuses on dataset paths. Later steps will extend this
    with OpenRouter and Memgraph connection details.
    """

    dataset: str = "sample"

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_FILES:
            available = ", ".join(sorted(DATASET_FILES))
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. "
                f"Available datasets: {available}"
            )

        # Ensure the output directory exists early.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def corpus_path(self) -> Path:
        return DATASET_DIR / DATASET_FILES[self.dataset]["corpus"]

    @property
    def queries_path(self) -> Path:
        return DATASET_DIR / DATASET_FILES[self.dataset]["queries"]


def get_default_openrouter_model() -> str:
    """
    Return the OpenRouter model to use for HippoRAG triple extraction.

    Resolution order:
    1. Environment variable OPENROUTER_MODEL, if set.
    2. Fallback to DEFAULT_OPENROUTER_MODEL defined in this module.
    """

    return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)


def list_available_datasets() -> Dict[str, Dict[str, str]]:
    """
    Return the known datasets and their underlying file names.

    This is useful both for CLIs (to show user choices) and for diagnostics.
    """

    return dict(DATASET_FILES)


def load_free_models(active_only: bool = True) -> List[str]:
    """
    Load free model IDs from free-models.yml.

    Args:
        active_only: If True, only return models with status="active". If False, return all models.

    Returns:
        List of model IDs (e.g., ["tngtech/deepseek-r1t2-chimera:free", ...])
    """
    if not FREE_MODELS_YAML.exists():
        logger.warning("free-models.yml not found at {}, using default model only", FREE_MODELS_YAML)
        return [DEFAULT_OPENROUTER_MODEL]

    try:
        with FREE_MODELS_YAML.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            models = data.get("models", [])
            
            if active_only:
                # Only return models with status="active" (or no status field, for backward compatibility)
                model_ids = [
                    m.get("id")
                    for m in models
                    if m.get("id") and (m.get("status") == "active" or "status" not in m)
                ]
            else:
                # Return all models
                model_ids = [m.get("id") for m in models if m.get("id")]
            
            if not model_ids:
                logger.warning(
                    "No {} model IDs found in free-models.yml, using default",
                    "active" if active_only else "",
                )
                return [DEFAULT_OPENROUTER_MODEL]
            
            if active_only:
                logger.info("Loaded {} active models from free-models.yml", len(model_ids))
            else:
                logger.info("Loaded {} models from free-models.yml", len(model_ids))
            
            return model_ids
    except Exception as e:
        logger.warning("Failed to load free-models.yml: {}, using default model", e)
        return [DEFAULT_OPENROUTER_MODEL]


__all__ = [
    "HippoRagConfig",
    "DATASET_DIR",
    "OUTPUT_DIR",
    "DATASET_FILES",
    "DEFAULT_OPENROUTER_MODEL",
    "get_default_openrouter_model",
    "list_available_datasets",
    "load_free_models",
    "FREE_MODELS_YAML",
]

