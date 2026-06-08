from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.config import get_config
from src.models.data_models import Triple
from src.utils.openrouter_client import OpenRouterClient

from .chunker import chunk_corpus, ChunkedPassage
from .config import OUTPUT_DIR, get_default_openrouter_model, load_free_models
from .load_dataset import CorpusEntry
from .progress_tracker import ProgressTracker


def _load_triples_from_csv(path: Path) -> List[Triple]:
    """Load triples from a CSV file (subject, predicate, object, source_chunk_id)."""
    if not path.exists():
        return []
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub, pred, obj = row.get("subject", ""), row.get("predicate", ""), row.get("object", "")
            sid = row.get("source_chunk_id", "")
            if sub and pred and obj:
                triples.append(
                    Triple(
                        subject=sub,
                        predicate=pred,
                        object=obj,
                        confidence=1.0,
                        source_chunk_id=sid,
                        metadata={},
                    )
                )
    return triples


DEFAULT_OPENROUTER_MODEL = get_default_openrouter_model()


@dataclass
class TripleExtractionConfig:
    """Configuration for OpenRouter-based triple extraction."""

    api_key: str
    model: str = DEFAULT_OPENROUTER_MODEL


def _load_extraction_config() -> TripleExtractionConfig:
    """
    Load OpenRouter configuration using the project's Pydantic settings.

    This automatically loads values from the root `.env` via `get_config()`,
    so you don't need to export environment variables manually.
    """

    cfg = get_config()
    api_key = cfg.openrouter_api_key
    model = get_default_openrouter_model()
    return TripleExtractionConfig(api_key=api_key, model=model)


def _build_prompt_for_passage(passage: str) -> str:
    """
    Build a simplified HippoRAG-style prompt for a single passage.

    We reuse the original HippoRAG instruction (see scratchpad) but
    inline it as a single user prompt here for minimal integration.
    """

    return (
        "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\n"
        "Paragraph:\n"
        "```\n"
        f"{passage}\n"
        "```\n\n"
        # We do not run a separate NER step here; instead we let the model
        # infer entities directly.
        '{ "named_entity_list": "INFER FROM PARAGRAPH" }\n\n'
        'Respond ONLY with JSON of the form: {"triples": [[subject, relation, object], ...]}'
    )


def _extract_json_block(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from a model response.

    We search for the first '{' and last '}' and attempt to parse.
    """

    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from model response snippet: {}", candidate[:200])
        return None


def _is_credit_error(error: str) -> bool:
    """
    Check if error message indicates a credit/affordability issue.

    Args:
        error: Error message string

    Returns:
        True if error is credit-related
    """
    if not error:
        return False
    error_lower = error.lower()
    credit_keywords = ["credit", "afford", "insufficient", "requires more credits", "can only afford"]
    return any(keyword in error_lower for keyword in credit_keywords)


async def _extract_triples_for_chunk_async(
    client: OpenRouterClient,
    chunk: ChunkedPassage,
    tracker: ProgressTracker,
    output_file: Path,
    model_id: str,
) -> tuple[List[Triple], Optional[str]]:
    """
    Extract triples for a single chunk with progress tracking.

    Args:
        client: OpenRouter client
        chunk: ChunkedPassage to process
        tracker: ProgressTracker for WAL logging
        output_file: Final output CSV file to append to
        model_id: Current model ID being used

    Returns:
        Tuple of (triples, error). error is None on success, or "CREDIT_ERROR" if credit issue detected.
    """
    prompt = _build_prompt_for_passage(chunk.content)

    # Log before request (WAL)
    tracker.log_before_request(chunk.chunk_id, chunk.content[:100])

    logger.debug("Requesting triples for chunk_id='{}' ({} tokens)", chunk.chunk_id, chunk.token_count)

    # We use the system prompt to mirror the original HippoRAG instruction.
    system_prompt = (
        "Your task is to construct an RDF (Resource Description Framework) graph from the "
        "given passages and (possibly implicit) named entities. "
        "Respond with a JSON list of triples, with each triple representing a relationship "
        "in the RDF graph. Each triple should contain at least one named entity. "
        "Resolve pronouns to their specific names when possible."
    )

    raw_response: Optional[str] = None
    error: Optional[str] = None
    result: List[Triple] = []

    try:
        raw_response = await client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=4096,  # Limit to avoid credit issues with free tier
            response_format={"type": "json_object"},
        )

        data = _extract_json_block(raw_response)
        if not data:
            error = "No JSON triples returned"
            logger.warning("No JSON triples returned for chunk_id='{}'", chunk.chunk_id)
        else:
            triples_raw = data.get("triples") or []
            for item in triples_raw:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                subject, predicate, obj = (str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip())
                if not subject or not predicate or not obj:
                    continue
                result.append(
                    Triple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=1.0,
                        source_chunk_id=chunk.chunk_id,
                        metadata={},
                    )
                )

    except Exception as e:
        error = str(e)
        logger.exception("Error extracting triples for chunk_id='{}' with model '{}': {}", chunk.chunk_id, model_id, e)

    # Check for credit error
    credit_error = _is_credit_error(error) if error else False
    if credit_error:
        logger.warning("Credit error detected for chunk_id='{}' with model '{}': {}", chunk.chunk_id, model_id, error)

    # Log after response (WAL) and save individual result (batched)
    tracker.log_after_response(chunk.chunk_id, result, raw_response, error)

    # Append to final output file incrementally
    if result:
        tracker.append_to_final_output(result, output_file)

    logger.info("Extracted {} triples for chunk_id='{}' with model '{}'", len(result), chunk.chunk_id, model_id)
    
    # Return error indicator for credit errors
    return_error = "CREDIT_ERROR" if credit_error else None
    return result, return_error


async def _extract_triples_for_corpus_async(
    corpus: List[CorpusEntry],
    cfg: TripleExtractionConfig,
    dataset_name: str,
) -> List[Triple]:
    """
    Extract triples for corpus with chunking and progress tracking.

    Args:
        corpus: List of CorpusEntry objects
        cfg: TripleExtractionConfig
        dataset_name: Dataset name for progress tracking

    Returns:
        List of all extracted triples
    """
    # Step 1: Chunk passages using HippoRAG chunker
    logger.info("Chunking {} passages before triple extraction", len(corpus))
    chunks = chunk_corpus(corpus)
    logger.info("Created {} chunks from {} passages", len(chunks), len(corpus))

    # Step 2: Initialize progress tracker and output file; apply resume
    tracker = ProgressTracker(dataset_name)
    output_file = OUTPUT_DIR / f"{dataset_name}_triples.csv"

    processed_ids = tracker.get_processed_chunk_ids()
    chunks_to_run = [c for c in chunks if c.chunk_id not in processed_ids]
    if processed_ids:
        logger.info(
            "Resume: skipping {} already-processed chunks, {} chunks left to run",
            len(processed_ids),
            len(chunks_to_run),
        )

    # Ensure output file exists with header (will be appended to during extraction)
    if not output_file.exists():
        with output_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "predicate", "object", "source_chunk_id"])
        logger.info("Created new output file with header: {}", output_file)
    else:
        logger.info("Output file exists, will append to: {}", output_file)

    # Step 3: Load only active free models (status=active in free-models.yml) and extract with rotation
    free_models = load_free_models(active_only=True)
    logger.info("Loaded {} active free models from free-models.yml", len(free_models))
    
    # Validate that we have at least one model to use
    if not free_models:
        error_msg = (
            "No active models found in free-models.yml. "
            "All models are marked as 'inactive' (likely due to rate limits or errors). "
            "Please run 'uv run python -m src.hipporag.test_models' to refresh model status, "
            "or wait for rate limits to reset, or add credits to your OpenRouter account."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    triples_from_run: List[Triple] = []
    if chunks_to_run:
        model_index = 0
        current_model = free_models[model_index]
        logger.info("Starting extraction with model: {}", current_model)
        
        client = None
        for chunk_idx, chunk in enumerate(chunks_to_run):
            # Create or recreate client if needed
            if client is None:
                client = OpenRouterClient(api_key=cfg.api_key, model=current_model)
                await client.__aenter__()
            
            triples, error_type = await _extract_triples_for_chunk_async(
                client, chunk, tracker, output_file, current_model
            )
            triples_from_run.extend(triples)
            
            # Rotate model on credit error
            if error_type == "CREDIT_ERROR":
                model_index += 1
                if model_index >= len(free_models):
                    # All models exhausted - prompt user
                    remaining = len(chunks_to_run) - chunk_idx - 1
                    logger.error(
                        "All {} free models exhausted due to credit errors. "
                        "Chunk '{}' failed with credit error.",
                        len(free_models),
                        chunk.chunk_id,
                    )
                    print("\n" + "=" * 70)
                    print("⚠️  ALL FREE MODELS EXHAUSTED")
                    print("=" * 70)
                    print(f"All {len(free_models)} free models have been tried and exhausted due to credit errors.")
                    print(f"Last failed chunk: {chunk.chunk_id}")
                    print(f"Remaining chunks: {remaining}")
                    print("\nOptions:")
                    print("1. Wait and retry later (credits may reset)")
                    print("2. Upgrade to a paid OpenRouter account")
                    print("3. Rotate and retry from first model (may work if credits reset)")
                    print()
                    response = input("Rotate and retry from first model? (y/n): ").strip().lower()
                    if response == "y":
                        model_index = 0
                        current_model = free_models[model_index]
                        logger.info("Rotating to first model: {}", current_model)
                        # Recreate client with new model
                        await client.__aexit__(None, None, None)
                        client = OpenRouterClient(api_key=cfg.api_key, model=current_model)
                        await client.__aenter__()
                        # Retry current chunk
                        triples, error_type = await _extract_triples_for_chunk_async(
                            client, chunk, tracker, output_file, current_model
                        )
                        triples_from_run.extend(triples)
                    else:
                        logger.warning("User chose not to rotate. Stopping extraction.")
                        break
                else:
                    # Switch to next model
                    current_model = free_models[model_index]
                    logger.warning(
                        "Credit error with model '{}', rotating to model {}: '{}'",
                        free_models[model_index - 1],
                        model_index + 1,
                        current_model,
                    )
                    # Recreate client with new model
                    await client.__aexit__(None, None, None)
                    client = OpenRouterClient(api_key=cfg.api_key, model=current_model)
                    await client.__aenter__()
                    # Retry current chunk with new model
                    triples, error_type = await _extract_triples_for_chunk_async(
                        client, chunk, tracker, output_file, current_model
                    )
                    triples_from_run.extend(triples)
        
        # Clean up client
        if client:
            await client.__aexit__(None, None, None)
        
        # Flush any remaining batched results
        tracker.flush()

    # Step 4: Return full triple set (from CSV when resuming, else from this run)
    if processed_ids and output_file.exists():
        triples = _load_triples_from_csv(output_file)
        logger.info(
            "Triple extraction complete for dataset '{}': {} total triples in CSV "
            "(this run: {}, resumed from {} processed chunks)",
            dataset_name,
            len(triples),
            len(triples_from_run),
            len(processed_ids),
        )
        return triples

    summary = tracker.get_progress_summary()
    logger.info(
        "Triple extraction complete for dataset '{}': {} triples extracted "
        "(completed: {}, failed: {}, pending: {})",
        dataset_name,
        len(triples_from_run),
        summary["completed"],
        summary["failed"],
        summary["pending"],
    )
    return triples_from_run


def extract_triples_for_corpus(corpus: List[CorpusEntry], dataset_name: str) -> List[Triple]:
    """
    Public synchronous API to extract triples for an entire corpus.

    This wraps the async implementation and returns a flat list of Triple objects.
    Uses HippoRAG chunking and WAL-style progress tracking.

    Args:
        corpus: List of CorpusEntry objects
        dataset_name: Dataset name for progress tracking and output files

    Returns:
        List of all extracted triples
    """
    cfg = _load_extraction_config()
    logger.info(
        "Starting triple extraction for dataset '{}': {} passages using OpenRouter model '{}'",
        dataset_name,
        len(corpus),
        cfg.model,
    )
    return asyncio.run(_extract_triples_for_corpus_async(corpus, cfg, dataset_name))


__all__ = ["extract_triples_for_corpus", "TripleExtractionConfig"]

