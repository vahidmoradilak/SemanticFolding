#!/usr/bin/env python3
"""
Semantic Folding Evaluation Pipeline - Main Orchestration Script

This script implements a complete semantic folding evaluation pipeline for the MuSiQue dataset,
comparing semantic folding against multiple baseline retrieval methods.

Each phase now delegates execution to semantic_folder.py for consistent error handling,
logging, and resume functionality.

Usage:
    python scratchpad.py --corpus_path ../../data/HippoRAG2/dataset/musique_corpus.json

For interactive management with resume capabilities:
    python semantic_folder.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any

import loguru
from loguru import logger

# Import TUI for configuration and error checking
try:
    from brain_approaches.semantic_folding.semantic_folder import SemanticFoldingTUI
    TUI_AVAILABLE = True
except ImportError:
    logger.warning("TUI not available, running in basic mode")
    TUI_AVAILABLE = False


def setup_logging(output_dir: Path, timestamp: str, log_level: str = "INFO", debug_mode: bool = False) -> Path:
    """
    Setup comprehensive logging with loguru for both console and file output.

    Args:
        output_dir: Output directory for log files
        timestamp: Timestamp string for log filename
        log_level: Console log level (default: INFO)
        debug_mode: Enable debug logging to console (default: False)

    Returns:
        Path to the log file
    """
    # Create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Determine console log level
    console_level = "DEBUG" if debug_mode else log_level

    # Add console handler with colors and clean format
    logger.add(
        lambda msg: print(msg, end="", flush=True),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | <level>{message}</level>",
        level=console_level,
        colorize=True,
        backtrace=True if debug_mode else False,
        diagnose=True if debug_mode else False
    )

    # Add file handler with detailed format and DEBUG level
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
        encoding="utf-8",
        backtrace=True,
        diagnose=True
    )

    # Add separate error log file for warnings and above
    error_log_file = logs_dir / f"errors_{timestamp}.log"
    logger.add(
        error_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="WARNING",
        rotation="50 MB",
        retention="30 days",
        encoding="utf-8"
    )

    # Test logging
    logger.info("Semantic Folding Pipeline logging initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Error log: {error_log_file}")
    logger.info(f"Console level: {console_level}, File level: DEBUG")

    if debug_mode:
        logger.debug("Debug mode enabled - showing detailed tracebacks")

    return log_file


def create_output_structure(base_dir: Path, timestamp: str) -> Path:
    """Create the complete output directory structure"""
    output_dir = base_dir / f"musique_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create all required subdirectories
    subdirs = [
        "fingerprints",
        "doc_fingerprints",
        "lance_db",
        "visualizations",
        "logs",
        "temp"
    ]

    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.success(f"Created output directory structure: {output_dir}")
    return output_dir


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """Load and validate the MuSiQue corpus"""
    logger.info(f"Loading corpus from: {corpus_path}")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    if not isinstance(corpus, list):
        raise ValueError("Corpus must be a list of passage objects")

    # Validate structure
    for i, item in enumerate(corpus):
        if not isinstance(item, dict) or 'title' not in item or 'text' not in item:
            raise ValueError(f"Invalid corpus item at index {i}: missing 'title' or 'text' field")

    logger.success(f"Loaded {len(corpus)} passages from corpus")
    logger.info(f"Sample passage: {corpus[0]['title'][:50]}...")

    return corpus


def convert_to_corpus_txt(corpus: List[Dict[str, Any]], output_path: Path) -> None:
    """Convert corpus to corpus.txt format: idx,title: text"""
    logger.info(f"Converting corpus to txt format: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(corpus):
            title = item['title'].replace('\n', ' ').replace('\r', ' ')
            text = item['text'].replace('\n', ' ').replace('\r', ' ')
            line = f"{idx},{title}: {text}\n"
            f.write(line)

    logger.success(f"Converted {len(corpus)} passages to corpus.txt format")


def log_corpus_statistics(corpus: List[Dict[str, Any]]) -> None:
    """Log comprehensive corpus statistics"""
    total_passages = len(corpus)
    total_tokens = sum(len(item['text'].split()) for item in corpus)
    avg_length = total_tokens / total_passages if total_passages > 0 else 0
    total_chars = sum(len(item['text']) for item in corpus)
    avg_chars = total_chars / total_passages if total_passages > 0 else 0

    logger.info("Corpus Statistics:")
    logger.info(f"  Total passages: {total_passages:,}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Average tokens per passage: {avg_length:.1f}")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Average characters per passage: {avg_chars:.1f}")


def load_corpus_contexts(corpus_path: Path) -> Dict[str, str]:
    """Load context texts from corpus file for LanceDB storage"""
    logger.info(f"Loading corpus contexts from: {corpus_path}")

    contexts = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue

            context_id, context_text = line.split(',', 1)
            context_id = context_id.strip()
            contexts[context_id] = context_text.strip()

    logger.success(f"Loaded {len(contexts)} context texts")
    return contexts


def main():
    """Main orchestration function"""
    parser = argparse.ArgumentParser(description="Semantic Folding Evaluation Pipeline")
    parser.add_argument(
        "--corpus_path",
        required=True,
        help="Path to musique_corpus.json file"
    )
    parser.add_argument(
        "--queries_path",
        default="../../data/HippoRAG2/dataset/musique.json",
        help="Path to musique.json queries file"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=16,
        help="Semantic space grid size (default: 16)"
    )
    parser.add_argument(
        "--top_k",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="Top-K values for evaluation (default: [1, 5, 10, 20])"
    )
    parser.add_argument(
        "--output_base",
        default="outputs",
        help="Base output directory name"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level (default: INFO)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed tracebacks"
    )

    args = parser.parse_args()

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting Semantic Folding Pipeline - {timestamp}")

    # Create output directory structure
    base_output_dir = Path(args.output_base)
    output_dir = create_output_structure(base_output_dir, timestamp)

    # Setup logging
    log_file = setup_logging(output_dir, timestamp, args.log_level, args.debug)

    # Phase 1: Load and preprocess corpus
    logger.info("=" * 60)
    logger.info("PHASE 1: Corpus Loading & Preprocessing")
    logger.info("=" * 60)

    try:
        # Load corpus
        corpus = load_corpus(args.corpus_path)

        # Log statistics
        log_corpus_statistics(corpus)

        # Convert to corpus.txt format
        corpus_txt_path = output_dir / "corpus.txt"
        convert_to_corpus_txt(corpus, corpus_txt_path)

        logger.success("Phase 1 completed successfully")

        # Save resume state
        try:
            from brain_approaches.semantic_folding.semantic_folder import SemanticFoldingTUI
            tui = SemanticFoldingTUI()
            tui.save_resume_state(str(output_dir), 1)
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")

    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        raise

    # Phase 2: Phrase Extraction
    logger.info("=" * 60)
    logger.info("PHASE 2: Phrase Extraction")
    logger.info("=" * 60)

    try:
        logger.info("Starting phrase extraction...")
        # Use semantic_folder.py for phase execution
        import subprocess

        cmd = [
            "uv", "run", "python", "semantic_folder.py",
            "--config", "../config/semantic_folding.yml",
            "--run-phase", "2",
            "--output-dir", str(output_dir)
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Phase 2 failed with exit code {result.returncode}")
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Phase 2 execution failed")

        logger.success("Phase 2 completed successfully")

        # Save resume state
        try:
            from brain_approaches.semantic_folding.semantic_folder import SemanticFoldingTUI
            tui = SemanticFoldingTUI()
            tui.save_resume_state(str(output_dir), 2)
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")

    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        raise

    # Phase 3: Term-Context Matrix Construction
    logger.info("=" * 60)
    logger.info("PHASE 3: Term-Context Matrix Construction")
    logger.info("=" * 60)

    try:
        logger.info("Starting term-context matrix construction...")
        # Use semantic_folder.py for phase execution
        import subprocess

        cmd = [
            "uv", "run", "python", "semantic_folder.py",
            "--config", "../config/semantic_folding.yml",
            "--run-phase", "3",
            "--output-dir", str(output_dir)
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Phase 3 failed with exit code {result.returncode}")
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Phase 3 execution failed")

        logger.success("Phase 3 completed successfully")

        # Save resume state
        try:
            from brain_approaches.semantic_folding.semantic_folder import SemanticFoldingTUI
            tui = SemanticFoldingTUI()
            tui.save_resume_state(str(output_dir), 3)
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")

    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        raise

    # Phase 4: Semantic Space Construction
    logger.info("=" * 60)
    logger.info("PHASE 4: Semantic Space Construction")
    logger.info("=" * 60)

    try:
        logger.info("Starting semantic space construction...")
        # Use semantic_folder.py for phase execution
        import subprocess

        cmd = [
            "uv", "run", "python", "semantic_folder.py",
            "--config", "../config/semantic_folding.yml",
            "--run-phase", "4",
            "--output-dir", str(output_dir)
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Phase 4 failed with exit code {result.returncode}")
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Phase 4 execution failed")

        logger.success("Phase 4 completed successfully")

        # Save resume state
        try:
            from brain_approaches.semantic_folding.semantic_folder import SemanticFoldingTUI
            tui = SemanticFoldingTUI()
            tui.save_resume_state(str(output_dir), 4)
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")

    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        raise

    # Phase 5: Fingerprint Generation
    logger.info("=" * 60)
    logger.info("PHASE 5: Fingerprint Generation")
    logger.info("=" * 60)

    try:
        logger.info("Starting fingerprint generation...")
        # Use semantic_folder.py for phase execution
        import subprocess

        cmd = [
            "uv", "run", "python", "semantic_folder.py",
            "--config", "../config/semantic_folding.yml",
            "--run-phase", "5",
            "--output-dir", str(output_dir)
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Phase 5 failed with exit code {result.returncode}")
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Phase 5 execution failed")

        logger.success("Phase 5 (fingerprint generation) completed successfully")

        # Save resume state
        try:
            from brain_approaches.semantic_folding.semantic_folder import SemanticFoldingTUI
            tui = SemanticFoldingTUI()
            tui.save_resume_state(str(output_dir), 5)
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")

    except Exception as e:
        logger.error(f"Phase 5 failed: {e}")
        raise

    # Phase 6: LanceDB Integration
    logger.info("=" * 60)
    logger.info("PHASE 6: LanceDB Integration")
    logger.info("=" * 60)

    try:
        logger.info("Starting LanceDB integration...")
        # Use semantic_folder.py for phase execution
        import subprocess

        cmd = [
            "uv", "run", "python", "semantic_folder.py",
            "--config", "../config/semantic_folding.yml",
            "--run-phase", "6",
            "--output-dir", str(output_dir)
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Phase 6 failed with exit code {result.returncode}")
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Phase 6 execution failed")

        logger.success("Phase 6 (LanceDB integration) completed successfully")

        # Clear resume state on successful completion
        try:
            from brain_approaches.semantic_folding.semantic_folder import SemanticFoldingTUI
            tui = SemanticFoldingTUI()
            tui.clear_resume_state()
        except Exception as e:
            logger.warning(f"Failed to clear resume state: {e}")

    except Exception as e:
        logger.error(f"Phase 6 failed: {e}")
        raise

    logger.success("Pipeline phases 1-6 complete. Ready for Phase 7 implementation.")


if __name__ == "__main__":
    main()
