"""
HippoRAG chunker - adapts the original chunker logic for HippoRAG2 passages.

Chunks passages into optimal sizes for triple extraction, using the same
logic as src/agents/chunker_node.py but adapted for CorpusEntry objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import tiktoken
from loguru import logger

from src.config import get_config

from .load_dataset import CorpusEntry

# Try to get encoding, fallback to cl100k_base (GPT-4 tokenizer)
try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = None
    logger.warning("tiktoken encoding not available, using character-based estimation")


@dataclass
class ChunkedPassage:
    """A chunked portion of a passage."""

    chunk_id: str  # e.g., "doc_0_chunk_0"
    passage_idx: int  # Original passage index
    chunk_index: int  # Index within the passage
    content: str
    token_count: int
    overlap_with_next: int = 0  # Characters overlapping with next chunk


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    if encoding:
        return len(encoding.encode(text))
    # Fallback: estimate 1 token ≈ 4 characters
    return len(text) // 4


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Pattern to match sentence endings
    # Matches: . ! ? followed by space or end of string
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\n\n)"
    sentences = re.split(sentence_pattern, text)
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


def chunk_passage(
    passage: CorpusEntry,
    target_size: int = 1000,
    overlap_ratio: float = 0.15,
) -> List[ChunkedPassage]:
    """
    Chunk a passage into smaller pieces with overlap.

    Uses the same logic as the original HippoRAG chunker:
    - Splits at sentence boundaries
    - Target size in tokens (default 1000)
    - Applies overlap ratio (default 15%)
    - Validates chunk sizes (min 200, max 2000 tokens)

    Args:
        passage: CorpusEntry to chunk
        target_size: Target chunk size in tokens
        overlap_ratio: Overlap ratio (0.0-0.5)

    Returns:
        List of ChunkedPassage objects
    """
    content = passage.text
    passage_id = f"doc_{passage.idx}"

    if not content:
        return []

    # Split into sentences
    sentences = _split_sentences(content)
    if not sentences:
        # Fallback: split by paragraphs
        paragraphs = content.split("\n\n")
        sentences = [p.strip() for p in paragraphs if p.strip()]

    if not sentences:
        # Last resort: single chunk
        token_count = _estimate_tokens(content)
        chunk = ChunkedPassage(
            chunk_id=f"{passage_id}_chunk_0",
            passage_idx=passage.idx,
            chunk_index=0,
            content=content,
            token_count=token_count,
            overlap_with_next=0,
        )
        return [chunk]

    chunks: List[ChunkedPassage] = []
    current_chunk_sentences: List[str] = []
    current_tokens = 0
    chunk_index = 0

    # Calculate overlap size
    overlap_tokens = int(target_size * overlap_ratio)

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If adding this sentence would exceed target, finalize current chunk
        if current_tokens + sentence_tokens > target_size and current_chunk_sentences:
            # Create chunk
            chunk_content = " ".join(current_chunk_sentences)
            chunk = ChunkedPassage(
                chunk_id=f"{passage_id}_chunk_{chunk_index}",
                passage_idx=passage.idx,
                chunk_index=chunk_index,
                content=chunk_content,
                token_count=_estimate_tokens(chunk_content),
                overlap_with_next=0,  # Will be set after next chunk is created
            )
            chunks.append(chunk)
            chunk_index += 1

            # Start new chunk with overlap
            # Take last sentences that fit in overlap
            overlap_sentences: List[str] = []
            overlap_token_count = 0
            for s in reversed(current_chunk_sentences):
                s_tokens = _estimate_tokens(s)
                if overlap_token_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_token_count += s_tokens
                else:
                    break

            current_chunk_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_token_count + sentence_tokens

            # Update previous chunk's overlap
            if chunks:
                chunks[-1].overlap_with_next = len(" ".join(overlap_sentences))
        else:
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Add final chunk
    if current_chunk_sentences:
        chunk_content = " ".join(current_chunk_sentences)
        chunk = ChunkedPassage(
            chunk_id=f"{passage_id}_chunk_{chunk_index}",
            passage_idx=passage.idx,
            chunk_index=chunk_index,
            content=chunk_content,
            token_count=_estimate_tokens(chunk_content),
            overlap_with_next=0,
        )
        chunks.append(chunk)

    # Validate chunks
    validated_chunks: List[ChunkedPassage] = []
    for chunk in chunks:
        token_count = chunk.token_count
        # Min 200 tokens, max 2000 tokens
        if token_count < 200:
            logger.warning(
                "Chunk {} is too small ({} tokens), considering merging with next chunk",
                chunk.chunk_id,
                token_count,
            )
            # Try to merge with next chunk if available
            if validated_chunks:
                last_chunk = validated_chunks[-1]
                merged_content = last_chunk.content + " " + chunk.content
                merged_tokens = _estimate_tokens(merged_content)
                if merged_tokens <= 2000:
                    last_chunk.content = merged_content
                    last_chunk.token_count = merged_tokens
                    last_chunk.overlap_with_next = chunk.overlap_with_next
                    continue
        elif token_count > 2000:
            logger.warning(
                "Chunk {} is too large ({} tokens), may cause issues with LLM",
                chunk.chunk_id,
                token_count,
            )

        validated_chunks.append(chunk)

    logger.debug(
        "Passage {}: Created {} chunks (avg {} tokens)",
        passage_id,
        len(validated_chunks),
        sum(c.token_count for c in validated_chunks) / len(validated_chunks) if validated_chunks else 0,
    )

    return validated_chunks


def chunk_corpus(
    corpus: List[CorpusEntry],
    target_size: Optional[int] = None,
    overlap_ratio: Optional[float] = None,
) -> List[ChunkedPassage]:
    """
    Chunk an entire corpus into passages.

    Args:
        corpus: List of CorpusEntry objects
        target_size: Target chunk size in tokens (defaults to config.chunk_size)
        overlap_ratio: Overlap ratio (defaults to config.chunk_overlap)

    Returns:
        List of ChunkedPassage objects
    """
    config = get_config()
    target_size = target_size or config.chunk_size
    overlap_ratio = overlap_ratio or config.chunk_overlap

    logger.info("Chunking {} passages with target_size={}, overlap_ratio={}", len(corpus), target_size, overlap_ratio)

    all_chunks: List[ChunkedPassage] = []
    for passage in corpus:
        chunks = chunk_passage(passage, target_size=target_size, overlap_ratio=overlap_ratio)
        all_chunks.extend(chunks)

    logger.info(
        "Created {} chunks from {} passages (avg {} tokens per chunk)",
        len(all_chunks),
        len(corpus),
        sum(c.token_count for c in all_chunks) / len(all_chunks) if all_chunks else 0,
    )

    return all_chunks


__all__ = ["chunk_passage", "chunk_corpus", "ChunkedPassage"]
