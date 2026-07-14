"""
Generic Benchmark Runner for Semantic Folding.

Generalizes the MuSiQue three-phase design so it works for ANY dataset that
has been converted to MuSiQue-like JSONL format.

Three phases:
  Phase 1 (index)    — Build combined corpus from unique paragraphs, run Steps 1-5
  Phase 2 (benchmark)— Run Step 6 per query against pre-built fingerprints
  Phase 3 (report)   — Generate markdown report + deep analysis

Usage:
    from semantic_folding.dataset_benchmark.adapters import get_adapter
    from semantic_folding.dataset_benchmark.generic_benchmark import GenericBenchmarkRunner

    adapter = get_adapter("pubmedqa")
    runner = GenericBenchmarkRunner(adapter, params={...})

    run_dir = runner.phase1_index(max_queries=500)
    bench_dir = runner.phase2_benchmark(run_dir, query_start=0, query_end=500)
    runner.phase3_report(bench_dir)
    # Optional: deep-dive analysis
    runner.analyze(bench_dir)
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import get_logger

from .adapters import BaseDatasetAdapter

logger = get_logger("generic_bench")

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # dataset_benchmark -> semantic_folding
SEMANTIC_FOLDING = PROJECT_ROOT / "semantic_folding"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

STEP_SCRIPTS = {
    1: SEMANTIC_FOLDING / "phrase_extractor.py",
    2: SEMANTIC_FOLDING / "term_context.py",
    3: SEMANTIC_FOLDING / "semantic_space.py",
    4: SEMANTIC_FOLDING / "phrase_fingerprints.py",
    5: SEMANTIC_FOLDING / "doc_fingerprints.py",
    6: SEMANTIC_FOLDING / "query_processor.py",
}

PIPELINE_DEFAULTS = {
    "grid_size": 64,
    "spreading_steps": 1,
    "top_k": 5,
    "weighting": "idf",
    "top_percent": 0.10,
    "smoothing_sigma": 1.5,
    "keep_verbs": True,
    "min_word_length": 3,
    "min_freq": 1,
    "max_doc_freq": 20,
    "morton": True,
    "method": "umap",
    "tsne_perplexity": 50,
    "tsne_iter": 1000,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.0,
    "umap_metric": "cosine",
    # Dynamic spreading: if True, picks spread=2 for short queries (≤ short_query_max_words)
    # and spread=1 for longer ones. Otherwise uses spreading_steps for all.
    "dynamic_spreading": False,
    "short_query_max_words": 10,
    "spreading_steps_long": 1,
    "doc_norm": "l2",
    # New feature defaults
    "spreading_decay": 0.5,
    "normalize_after_spreading": False,
    "normalization": "l2",
    "block_size": 8,
    "attention_temperature": 1.0,
    "snippet_window": 3,
    "snippet_stride": 2,
    "snippet_ranking": False,
    "cross_attention": False,
    "oov_expansion": False,
    "synonym_weight": 0.5,
}

# ============================================================================
# Dataset Registry
# ============================================================================
def load_dataset_registry(registry_path: Optional[Path] = None, dataset: str = None) -> Dict[str, Any]:
    """
    Load per-dataset parameter overrides from a YAML registry.
    
    Parameters
    ----------
    registry_path : Path, optional
        Path to dataset_registry.yml. If None, uses default location.
    dataset : str, optional
        Dataset name to load overrides for.
    
    Returns
    -------
    Dict[str, Any]
        Merged parameters: defaults + dataset-specific overrides.
    """
    if registry_path is None:
        registry_path = Path(__file__).resolve().parents[2] / "config" / "dataset_registry.yml"
    
    if not registry_path.exists():
        logger.warning(f"  [REGISTRY] Not found: {registry_path}")
        return {}
    
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"  [REGISTRY] Failed to load: {e}")
        return {}
    
    # Start with global defaults
    params = dict(registry.get("defaults", {}))
    
    # Apply dataset-specific overrides
    if dataset and dataset in registry:
        dataset_params = registry[dataset]
        # Skip non-parameter keys
        for key in ["description", "max_queries"]:
            dataset_params.pop(key, None)
        params.update(dataset_params)
        logger.info(f"  [REGISTRY] Loaded params for '{dataset}': {params}")
    elif dataset:
        logger.warning(f"  [REGISTRY] Dataset '{dataset}' not found in registry")
    
    return params


# ============================================================================
# Terminal Colors
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# ============================================================================
# Run registry
# ============================================================================
REGISTRY_PATH = SCRIPT_DIR / "runs_registry.yml"


def load_registry() -> dict:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        empty = {"runs": {}}
        with open(REGISTRY_PATH, "w") as f:
            yaml.dump(empty, f, default_flow_style=False)
        return empty
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"runs": {}}


def save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False)


def register_run(run_dir: Path, dataset: str, run_type: str, params: dict, status: str = "created"):
    registry = load_registry()
    run_id = f"{dataset}__{run_dir.name}"
    registry["runs"][run_id] = {
        "dataset": dataset,
        "type": run_type,
        "path": str(run_dir.resolve()),
        "created_at": datetime.now().isoformat(),
        "status": status,
        "params": {k: str(v) for k, v in params.items()},
    }
    save_registry(registry)


def update_run_status(run_dir: Path, dataset: str, status: str):
    registry = load_registry()
    run_id = f"{dataset}__{run_dir.name}"
    if run_id in registry["runs"]:
        registry["runs"][run_id]["status"] = status
        save_registry(registry)


# ============================================================================
# Data loading
# ============================================================================
def load_entries(jsonl_path: Path) -> List[dict]:
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ============================================================================
# Run step helper
# ============================================================================
def run_step(script: Path, args: List[str], workdir: Path, step_name: str,
             timeout: int = 1800) -> bool:
    args = [a for a in args if a]
    cmd = [sys.executable, str(script)] + args
    logger.info(f"  [{step_name}] starting...")
    try:
        result = subprocess.run(
            cmd, cwd=str(workdir), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            stderr_tail = result.stderr[-800:].replace("\n", " | ")
            logger.error(f"  [{step_name}] FAILED (rc={result.returncode}): {stderr_tail}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"  [{step_name}] TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"  [{step_name}] ERROR: {e}")
        return False


# ============================================================================
# Phase 1 — Index
# ============================================================================
def build_combined_corpus(entries: List[dict]):
    seen = {}
    corpus_lines = []
    query_doc_map = {}
    query_gold = {}

    next_id = 0
    for q_idx, entry in enumerate(entries):
        doc_ids = []
        gold_ids = []
        for p in entry["paragraphs"]:
            key = (p.get("title", ""), p.get("paragraph_text", ""))
            if key not in seen:
                gid = f"doc_{next_id:06d}"
                seen[key] = gid
                title = p.get("title", "")
                text = p.get("paragraph_text", "")
                corpus_lines.append(f"{gid}, {title} {text}")
                next_id += 1
            else:
                gid = seen[key]
            doc_ids.append(gid)
            if p.get("is_supporting", False):
                gold_ids.append(gid)
        query_doc_map[str(q_idx)] = doc_ids
        if gold_ids:
            query_gold[str(q_idx)] = gold_ids

    logger.info(f"Combined corpus: {len(corpus_lines)} unique paragraphs across {len(entries)} queries")
    return corpus_lines, query_doc_map, query_gold


# ============================================================================
# Phase 2 — Benchmark helpers (same as musique)
# ============================================================================
def filter_results_to_candidates(full_results: List[list], candidate_ids: List[str]) -> List[Tuple[str, float]]:
    cand_set = set(candidate_ids)
    return [(doc_id, score) for doc_id, score in full_results if doc_id in cand_set]


def compute_metrics(retrieved: List[Tuple[str, float]], relevant: List[str],
                    top_k_list: List[int] = None) -> dict:
    if top_k_list is None:
        top_k_list = [1, 2, 3, 5]
    retrieved_ids = [doc_id for doc_id, _ in retrieved]
    rel_set = set(relevant)

    found_at = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            found_at = rank
            break
    mrr = 1.0 / found_at if found_at > 0 else 0.0

    ap = 0.0
    hits = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in rel_set:
            hits += 1
            ap += hits / rank
    ap /= len(relevant) if relevant else 1

    metrics = {"mrr": mrr, "ap": ap, "found_at": found_at}
    for k in top_k_list:
        retrieved_k = retrieved_ids[:k]
        rel_k = sum(1 for d in retrieved_k if d in rel_set)
        metrics[f"p@{k}"] = rel_k / k
        metrics[f"r@{k}"] = rel_k / len(relevant) if relevant else 0.0

    for k in top_k_list:
        dcg_k = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in rel_set:
                dcg_k += 1.0 / (rank + 1).bit_length()
        num_rel = min(len(relevant), k)
        idcg_k = sum(1.0 / (i + 1).bit_length() for i in range(num_rel))
        metrics[f"ndcg@{k}"] = dcg_k / idcg_k if idcg_k > 0 else 0.0
    return metrics


def load_query_results(result_path: Path) -> List[dict]:
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Generic Runner
# ============================================================================
class GenericBenchmarkRunner:
    def __init__(self, adapter: BaseDatasetAdapter, params: dict = None):
        self.adapter = adapter
        self.params = dict(PIPELINE_DEFAULTS)
        if params:
            self.params.update(params)
        # Allow adapter to override
        adapter_params = self.adapter.get_recommended_params()
        for k, v in adapter_params.items():
            self.params.setdefault(k, v)

        self.bench_base = OUTPUTS_DIR / f"{adapter.dataset_name}_benchmark"
        self.runs_dir = self.bench_base / "runs"
        self.benchmarks_dir = self.bench_base / "benchmarks"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    def phase1_index(self, jsonl_path: Path, max_queries: int = None) -> Optional[Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        entries = load_entries(jsonl_path)
        if max_queries is not None:
            entries = entries[:max_queries]
        logger.info(f"Index run: {run_dir.name} ({len(entries)} queries, "
                    f"params: {self.params})")

        # Build combined corpus
        corpus_lines, query_doc_map, query_gold = build_combined_corpus(entries)

        # P1.3: Corpus-level glossary expansion — append synonyms to corpus
        # lines so phrase extraction picks them up and creates fingerprints
        glossary_path = self.params.get("glossary_path")
        if self.params.get("glossary_corpus_expansion", False) and glossary_path:
            from ontology_expander import expand_corpus_with_glossary
            corpus_lines = expand_corpus_with_glossary(corpus_lines, glossary_path)

        corpus_path = run_dir / "corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for line in corpus_lines:
                f.write(line + "\n")

        # Store corpus path in params for Step 6 hybrid/SPLADE scoring
        self.params["corpus_path"] = str(corpus_path)

        with open(run_dir / "query_doc_map.json", "w") as f:
            json.dump(query_doc_map, f, indent=2)
        with open(run_dir / "query_gold.json", "w") as f:
            json.dump(query_gold, f, indent=2)
        with open(run_dir / "metadata.json", "w") as f:
            json.dump({
                "dataset": self.adapter.dataset_name,
                "display_name": self.adapter.display_name,
                "num_queries": len(entries),
                "num_docs": len(corpus_lines),
                "source_jsonl": str(jsonl_path),
                "created_at": ts,
            }, f, indent=2)

        run_config = {
            "phase1": {
                "mode": "index",
                "dataset": self.adapter.dataset_name,
                "timestamp": ts,
                "num_queries": len(entries),
                "num_docs": len(corpus_lines),
            },
            "pipeline": {k: v for k, v in self.params.items()},
        }
        with open(run_dir / "config.yml", "w") as f:
            yaml.dump(run_config, f, default_flow_style=False)

        register_run(run_dir, self.adapter.dataset_name, "index", self.params, "indexing")

        # Run steps 1-5
        # Step 1 (skip if prebuilt vocab provided)
        prebuilt = self.params.get("prebuilt_vocab")
        if prebuilt:
            prebuilt_path = Path(prebuilt)
            target = run_dir / "extracted_phrases"
            target.mkdir(parents=True, exist_ok=True)
            import shutil
            for fname in ("vocabulary.csv", "phrase_to_contexts.json"):
                src = prebuilt_path / fname
                if src.exists():
                    shutil.copy2(str(src), str(target / fname))
                    logger.info(f"  [PREBUILT] copied {fname} from {prebuilt_path}")
            logger.info("  [PREBUILT] skipping Step 1 + 0.5 (using pre-built vocabulary)")
        else:
            out = run_dir / "extracted_phrases"
            ok = run_step(STEP_SCRIPTS[1], [
                "--corpus", str(corpus_path), "--output", str(out),
                "--keep-verbs", "--min-word-length", str(self.params["min_word_length"]),
                "--min-freq", str(self.params["min_freq"]),
                "--max-doc-freq", str(self.params.get("max_doc_freq", 0)),
            ], PROJECT_ROOT, "Step 1 phrase_extractor")
            if not ok:
                update_run_status(run_dir, self.adapter.dataset_name, "failed_step1")
                return None

            # ── Step 0.5: LLM phrase enrichment (optional) ──────────────────────
            if self.params.get("llm_phrases", False):
                logger.info("  [LLM] Step 0.5: enriching phrase vocabulary via LLM extraction")
                from llm_phrase_extractor import (
                    extract_phrases_from_corpus,
                    merge_with_spacy,
                )
                llm_cache = run_dir / "llm_phrases"
                llm_counts, llm_mapping = extract_phrases_from_corpus(
                    corpus_lines,
                    min_freq=self.params.get("min_freq", 1),
                    max_doc_freq=self.params.get("max_doc_freq", 0),
                    batch_size=self.params.get("llm_batch_size", 8),
                    cache_dir=llm_cache,
                )
                # Merge into Step 1 output directory (overwrites)
                merge_with_spacy(
                    llm_counts, llm_mapping,
                    out / "vocabulary.csv",
                    out / "phrase_to_contexts.json",
                    run_dir,
                )

        # Step 2
        out = run_dir / "term_context_matrix"
        ok = run_step(STEP_SCRIPTS[2], [
            "--vocab", str(run_dir / "extracted_phrases" / "vocabulary.csv"),
            "--mapping", str(run_dir / "extracted_phrases" / "phrase_to_contexts.json"),
            "--corpus", str(corpus_path), "--output", str(out),
        ], PROJECT_ROOT, "Step 2 term_context")
        if not ok:
            update_run_status(run_dir, self.adapter.dataset_name, "failed_step2")
            return None

        # Step 3
        out = run_dir / "semantic_space"
        step3_args = [
            "--matrix", str(run_dir / "term_context_matrix" / "term_context_matrix.npz"),
            "--metadata", str(run_dir / "term_context_matrix" / "term_context_matrix.json"),
            "--output", str(out),
            "--grid-size", str(self.params["grid_size"]),
            "--method", self.params["method"],
        ]
        if self.params["method"] == "tsne":
            step3_args.extend([
                "--perplexity", str(self.params["tsne_perplexity"]),
                "--tsne-iter", str(self.params["tsne_iter"]),
            ])
        elif self.params["method"] == "umap":
            step3_args.extend([
                "--n-neighbors", str(self.params["umap_n_neighbors"]),
                "--min-dist", str(self.params["umap_min_dist"]),
                "--metric", self.params["umap_metric"],
            ])
        elif self.params["method"] == "learned":
            step3_args.extend([
                "--corpus", str(corpus_path),
            ])
        ok = run_step(STEP_SCRIPTS[3], step3_args, PROJECT_ROOT, "Step 3 semantic_space", timeout=900)
        if not ok:
            update_run_status(run_dir, self.adapter.dataset_name, "failed_step3")
            return None

        # Step 4
        out = run_dir / "phrase_fingerprints"
        morton_flag = "--morton" if self.params["morton"] else "--no-morton"
        ok = run_step(STEP_SCRIPTS[4], [
            "--coordinates", str(run_dir / "semantic_space" / "context_coordinates.json"),
            "--metadata", str(run_dir / "term_context_matrix" / "term_context_matrix.json"),
            "--output", str(out),
            "--grid-size", str(self.params["grid_size"]),
            "--smoothing-sigma", str(self.params["smoothing_sigma"]),
            morton_flag,
        ], PROJECT_ROOT, "Step 4 phrase_fingerprints")
        if not ok:
            update_run_status(run_dir, self.adapter.dataset_name, "failed_step4")
            return None

        # Step 5
        out = run_dir / "doc_fingerprints"
        ok = run_step(STEP_SCRIPTS[5], [
            "--corpus", str(corpus_path),
            "--fingerprints", str(run_dir / "phrase_fingerprints"),
            "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
            "--output", str(out),
            "--grid-size", str(self.params["grid_size"]),
            "--top-percent", str(self.params["top_percent"]),
            "--normalize-method", "l2",
            "--min-word-length", str(self.params["min_word_length"]),
            "--smoothing-sigma", str(self.params["smoothing_sigma"]),
            "--min-peak-distance", "2",
            morton_flag,
        ], PROJECT_ROOT, "Step 5 doc_fingerprints")
        if not ok:
            update_run_status(run_dir, self.adapter.dataset_name, "failed_step5")
            return None

        update_run_status(run_dir, self.adapter.dataset_name, "completed")
        logger.success(f"Index phase complete -> {run_dir}")
        return run_dir

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------
    def phase2_benchmark(self, run_dir: Path, jsonl_path: Path,
                         query_start: int = 0, query_end: int = None) -> Optional[Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_dir = self.benchmarks_dir / f"benchmark_{ts}"
        bench_dir.mkdir(parents=True, exist_ok=True)
        per_query_dir = bench_dir / "per_query"
        per_query_dir.mkdir(exist_ok=True)

        with open(run_dir / "query_doc_map.json", encoding="utf-8") as f:
            query_doc_map = json.load(f)
        with open(run_dir / "query_gold.json", encoding="utf-8") as f:
            query_gold = json.load(f)

        bench_config = {
            "phase2": {
                "mode": "benchmark",
                "dataset": self.adapter.dataset_name,
                "timestamp": ts,
                "run_dir": str(run_dir),
                "query_start": query_start,
                "query_end": query_end,
            },
            "pipeline": {k: v for k, v in self.params.items()},
        }
        with open(bench_dir / "config.yml", "w") as f:
            yaml.dump(bench_config, f, default_flow_style=False)

        register_run(bench_dir, self.adapter.dataset_name, "benchmark", self.params, "running")
        logger.info(f"Benchmark: {bench_dir.name} - queries {query_start}-{query_end}")

        entries = load_entries(jsonl_path)
        if query_end is None:
            query_end = len(entries)
        else:
            query_end = min(query_end, len(entries))

        all_metrics = []
        results_log = bench_dir / "results_log.csv"
        failed = 0
        total = query_end - query_start

        # ── Pass 1: collect gold-bearing queries for batch processing ─────
        batch_entries = []
        for q_idx in range(query_start, query_end):
            q_idx_str = str(q_idx)
            entry = entries[q_idx]
            query_text = entry["question"]
            candidate_ids = query_doc_map.get(q_idx_str, [])
            gold_ids = query_gold.get(q_idx_str, [])

            if not gold_ids:
                logger.debug(f"  [{q_idx}] no gold passages, skipping")
                continue

            n_words = len(query_text.split())
            if self.params.get("dynamic_spreading", False):
                if n_words <= self.params.get("short_query_max_words", 10):
                    spread = self.params.get("spreading_steps_long", 2)
                    spread_reason = f"short ({n_words} words)"
                else:
                    spread = self.params.get("spreading_steps", 1)
                    spread_reason = f"long ({n_words} words)"
            else:
                spread = self.params["spreading_steps"]
                spread_reason = "uniform"

            query_out_dir = per_query_dir / f"{q_idx:04d}"
            query_out_dir.mkdir(exist_ok=True)

            with open(query_out_dir / "candidate_docs.json", "w") as f:
                json.dump({"candidate_ids": candidate_ids, "gold_ids": gold_ids}, f, indent=2)

            batch_entries.append({
                "q_idx": q_idx,
                "entry": entry,
                "query_text": query_text,
                "n_words": n_words,
                "spread": spread,
                "spread_reason": spread_reason,
                "candidate_ids": candidate_ids,
                "gold_ids": gold_ids,
                "query_out_dir": query_out_dir,
            })

        if not batch_entries:
            logger.error("No queries with gold passages to benchmark!")
            update_run_status(bench_dir, self.adapter.dataset_name, "failed_no_gold")
            return None

        # ── Write query file for batch processing ─────────────────────────
        query_file = bench_dir / "queries.txt"
        with open(query_file, "w", encoding="utf-8") as f:
            for be in batch_entries:
                f.write(be["query_text"] + "\n")
        logger.info(f"Batch: {len(batch_entries)} queries written to {query_file}")

        # ── Pre-extract LLM query phrases if requested ────────────────────
        llm_query_phrases_path = None
        if self.params.get("llm_query_phrases"):
            try:
                from llm_phrase_extractor import extract_query_phrases_batch
                query_texts = [be["query_text"] for be in batch_entries]
                logger.info(f"  [LLM] extracting query phrases for {len(query_texts)} queries...")
                all_phrases = extract_query_phrases_batch(query_texts)
                llm_query_phrases_path = bench_dir / "llm_query_phrases.json"
                with open(llm_query_phrases_path, "w", encoding="utf-8") as f:
                    json.dump(all_phrases, f)
                n_nonempty = sum(1 for p in all_phrases if p)
                logger.info(
                    f"  [LLM] query phrases: {n_nonempty}/{len(all_phrases)} "
                    f"queries got extra phrases"
                )
            except ImportError:
                logger.warning("  [LLM] llm_phrase_extractor not available — skipping query extraction")
            except Exception as exc:
                logger.warning(f"  [LLM] query phrase extraction failed: {exc} — proceeding without")

        # ── Build step6 args (shared across all queries) ──────────────────
        batch_output = bench_dir / "all_results.json"
        step6_args = [
            "--query-file", str(query_file),
            "--fingerprints", str(run_dir / "phrase_fingerprints"),
            "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
            "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
            "--grid-size", str(self.params["grid_size"]),
            "--top-k", str(self.params["top_k"]),
            "--weighting", self.params["weighting"],
            "--spreading-steps", str(self.params["spreading_steps"]),
            "--output", str(batch_output),
            "--keep-verbs", "--min-word-length", str(self.params["min_word_length"]),
        ]
        if self.params.get("geometric", False):
            step6_args.append("--geometric")
        if self.params.get("hybrid", False):
            step6_args.extend(["--hybrid", "--hybrid-alpha", str(self.params.get("hybrid_alpha", 0.5))])
            if self.params.get("corpus_path"):
                step6_args.extend(["--corpus", self.params["corpus_path"]])
        if self.params.get("splade", False):
            step6_args.extend(["--splade", "--splade-model", self.params.get("splade_model", "naver/splade-cocondenser-ensembledistil")])
            if self.params.get("hybrid_alpha") is not None:
                step6_args.extend(["--hybrid-alpha", str(self.params["hybrid_alpha"])])
            if self.params.get("corpus_path"):
                step6_args.extend(["--corpus", self.params["corpus_path"]])
            fusion_method = self.params.get("fusion_method", "linear")
            if fusion_method == "rrf":
                step6_args.extend(["--fusion-method", "rrf"])
                step6_args.extend(["--rrf-k", str(self.params.get("rrf_k", 60))])
        if self.params.get("doc_norm", "sqrt_nnz") != "sqrt_nnz":
            step6_args.extend(["--doc-norm", self.params["doc_norm"]])
        if self.params.get("sim_metric", "cosine") != "cosine":
            step6_args.extend(["--sim-metric", self.params["sim_metric"]])
        if self.params.get("asymmetric", False):
            step6_args.append("--asymmetric")
            step6_args.extend(["--asym-alpha", str(self.params.get("asym_alpha", 0.7))])
        if self.params.get("score_norm", "none") != "none":
            step6_args.extend(["--score-norm", self.params["score_norm"]])
        if self.params.get("rerank", False):
            step6_args.append("--rerank")
            if self.params.get("rerank_model"):
                step6_args.extend(["--rerank-model", str(self.params["rerank_model"])])
            step6_args.extend(["--rerank-top-k", str(self.params.get("rerank_top_k", 100))])
        if self.params.get("negation_aware", False):
            step6_args.append("--negation-aware")
            step6_args.extend(["--negation-penalty", str(self.params.get("negation_penalty", 0.5))])
            step6_args.extend(["--negation-boost", str(self.params.get("negation_boost", 0.3))])
        if self.params.get("expand_synonyms", False):
            step6_args.append("--expand-synonyms")
            if self.params.get("glossary_path"):
                step6_args.extend(["--glossary", self.params["glossary_path"]])
        if self.params.get("tfidf_rerank", False):
            step6_args.extend(["--tfidf-rerank", "--tfidf-alpha", str(self.params.get("tfidf_alpha", 0.3))])
            if self.params.get("corpus_path"):
                step6_args.extend(["--corpus", self.params["corpus_path"]])
        if self.params.get("decompose", False):
            step6_args.append("--decompose")
        if self.params.get("multi_resolution", False):
            step6_args.append("--multi-resolution")
        if self.params.get("storage", "file") == "faiss":
            step6_args.extend(["--storage", "faiss"])
            if self.params.get("faiss_path"):
                step6_args.extend(["--faiss-path", str(self.params["faiss_path"])])

        # ── Cross-attention scoring (P2.2) ────────────────────────────────────
        if self.params.get("cross_attention", False):
            step6_args.append("--cross-attention")
            step6_args.extend(["--block-size", str(self.params.get("block_size", 8))])
            step6_args.extend(["--attention-temperature", str(self.params.get("attention_temperature", 1.0))])

        # ── Snippet ranking (P2.4) ────────────────────────────────────────────
        if self.params.get("snippet_ranking", False):
            step6_args.append("--snippet-ranking")
            step6_args.extend(["--snippet-window", str(self.params.get("snippet_window", 3))])
            step6_args.extend(["--snippet-stride", str(self.params.get("snippet_stride", 2))])
            if self.params.get("snippet_dir"):
                step6_args.extend(["--snippet-dir", self.params["snippet_dir"]])

        # ── Adaptive spreading ─────────────────────────────────────────────────
        if self.params.get("adaptive_spreading", False):
            step6_args.append("--adaptive-spreading")

        # ── Normalize after spreading ──────────────────────────────────────────
        if self.params.get("normalize_after_spreading", False):
            step6_args.append("--normalize-after-spreading")

        # ── Spreading decay ────────────────────────────────────────────────────
        if self.params.get("spreading_decay", 0.5) != 0.5:
            step6_args.extend(["--spreading-decay", str(self.params["spreading_decay"])])

        # ── Query normalisation ────────────────────────────────────────────────
        if self.params.get("normalization", "l2") != "l2":
            step6_args.extend(["--normalization", self.params["normalization"]])

        # ── OOV expansion ──────────────────────────────────────────────────────
        if not self.params.get("oov_expansion", True):
            step6_args.append("--no-oov-expansion")

        # ── Synonym weight ─────────────────────────────────────────────────────
        if self.params.get("synonym_weight", 0.5) != 0.5:
            step6_args.extend(["--synonym-weight", str(self.params["synonym_weight"])])

        if llm_query_phrases_path:
            step6_args.extend(["--llm-query-phrases", str(llm_query_phrases_path)])

        # ── Single subprocess call for ALL queries ────────────────────────
        t0 = time.time()
        ok = run_step(STEP_SCRIPTS[6], step6_args, PROJECT_ROOT, "Step 6 query_processor (batch)", timeout=3600)
        batch_elapsed = time.time() - t0

        if not ok:
            logger.error(f"Batch query processor FAILED after {batch_elapsed:.0f}s — {len(batch_entries)} queries lost")
            failed = len(batch_entries)
            update_run_status(bench_dir, self.adapter.dataset_name, "failed_batch")
            return bench_dir

        logger.info(f"Batch query processor completed in {batch_elapsed:.0f}s ({len(batch_entries)} queries)")

        # ── Read batch results ────────────────────────────────────────────
        with open(batch_output, encoding="utf-8") as f:
            all_query_results = json.load(f)

        # ── Pass 2: split results back per-query, compute metrics ────────
        for i, be in enumerate(batch_entries):
            q_idx = be["q_idx"]
            query_text = be["query_text"]
            n_words = be["n_words"]
            spread = be["spread"]
            spread_reason = be["spread_reason"]
            candidate_ids = be["candidate_ids"]
            gold_ids = be["gold_ids"]
            query_out_dir = be["query_out_dir"]

            result_json = query_out_dir / "query_results.json"

            raw = all_query_results[i] if i < len(all_query_results) else {"results": []}
            full_results = raw.get("results", [])

            # Save raw per-query result
            with open(result_json, "w") as f:
                json.dump([raw], f, indent=2)

            candidate_results = filter_results_to_candidates(full_results, candidate_ids)

            with open(query_out_dir / "filtered_results.json", "w") as f:
                json.dump({
                    "query_idx": q_idx,
                    "query": query_text,
                    "query_word_count": n_words,
                    "spreading_steps_used": spread,
                    "spreading_reason": spread_reason,
                    "gold": gold_ids,
                    "candidates": candidate_ids,
                    "filtered_ranked": [(doc_id, float(score)) for doc_id, score in candidate_results],
                    "full_top10": [(doc_id, float(score)) for doc_id, score in full_results[:10]],
                    "elapsed_s": round(batch_elapsed / len(batch_entries), 1),
                }, f, indent=2)

            metrics = compute_metrics(candidate_results, gold_ids,
                                      top_k_list=[1, 2, 3, 5, self.params["top_k"]])
            metrics["spreading_steps"] = spread
            all_metrics.append(metrics)

            logger.info(f"  [{q_idx:04d}/{query_end - 1}] MRR={metrics['mrr']:.3f} AP={metrics['ap']:.3f} "
                        f"P@2={metrics['p@2']:.3f} spread={spread}[{spread_reason[:5]}]  ({i+1}/{len(batch_entries)})")

        # ── Write CSV ────────────────────────────────────────────────────
        with open(results_log, "w", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            header = ["query_idx", "query", "n_words", "spread", "spread_reason",
                      "mrr", "ap", "p@1", "p@2", "p@3", "p@5", "r@2", "ndcg@2",
                      "found_at", "elapsed_s"]
            writer.writerow(header)
            for be, metrics in zip(batch_entries, all_metrics):
                writer.writerow([
                    be["q_idx"], be["query_text"][:60], be["n_words"], be["spread"], be["spread_reason"],
                    f"{metrics['mrr']:.4f}", f"{metrics['ap']:.4f}",
                    f"{metrics['p@1']:.4f}", f"{metrics['p@2']:.4f}",
                    f"{metrics['p@3']:.4f}", f"{metrics['p@5']:.4f}",
                    f"{metrics['r@2']:.4f}", f"{metrics['ndcg@2']:.4f}",
                    metrics.get("found_at", "none"), f"{batch_elapsed / len(batch_entries):.1f}",
                ])

        if all_metrics:
            agg = defaultdict(list)
            for m in all_metrics:
                for k, v in m.items():
                    agg[k].append(v)

            summary = {
                "dataset": self.adapter.dataset_name,
                "display_name": self.adapter.display_name,
                "num_queries": len(all_metrics),
                "failed": failed,
                "batch_elapsed_s": round(batch_elapsed, 1),
            }
            for k, vals in agg.items():
                summary[f"mean_{k}"] = sum(vals) / len(vals)
                summary[f"min_{k}"] = min(vals)
                summary[f"max_{k}"] = max(vals)

            with open(bench_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            # Per-spreading breakdown (only meaningful when dynamic_spreading is on)
            by_spread = defaultdict(list)
            for m in all_metrics:
                by_spread[int(m.get("spreading_steps", self.params["spreading_steps"]))].append(m)
            by_spread_summary = {}
            for spread_val, ms in sorted(by_spread.items()):
                sub_agg = defaultdict(list)
                for m in ms:
                    for k, v in m.items():
                        if k == "spreading_steps":
                            continue
                        sub_agg[k].append(v)
                by_spread_summary[str(spread_val)] = {
                    "n": len(ms),
                    **{f"mean_{k}": sum(vs) / len(vs) for k, vs in sub_agg.items()},
                }
            with open(bench_dir / "summary_by_spreading.json", "w") as f:
                json.dump(by_spread_summary, f, indent=2)

            # Clean, focused params file (for at-a-glance review)
            params_snapshot = {
                "dataset": self.adapter.dataset_name,
                "display_name": self.adapter.display_name,
                "run_dir": str(run_dir),
                "num_queries": len(all_metrics),
                "failed": failed,
                "pipeline": {
                    "grid_size": self.params["grid_size"],
                    "spreading_steps": self.params["spreading_steps"],
                    "dynamic_spreading": self.params.get("dynamic_spreading", False),
                    "short_query_max_words": self.params.get("short_query_max_words", 10),
                    "spreading_steps_long": self.params.get("spreading_steps_long", 2),
                    "top_k": self.params["top_k"],
                    "weighting": self.params["weighting"],
                    "top_percent": self.params["top_percent"],
                    "smoothing_sigma": self.params["smoothing_sigma"],
                    "morton": self.params["morton"],
                    "keep_verbs": self.params["keep_verbs"],
                    "min_word_length": self.params["min_word_length"],
                    "min_freq": self.params["min_freq"],
                },
                "generated": datetime.now().isoformat(timespec="seconds"),
            }
            with open(bench_dir / "params.json", "w") as f:
                json.dump(params_snapshot, f, indent=2)

            update_run_status(bench_dir, self.adapter.dataset_name, "completed")
            logger.success(f"Benchmark complete - {len(all_metrics)} queries, "
                           f"mean MRR={summary['mean_mrr']:.4f}, AP={summary['mean_ap']:.4f}")
            return bench_dir
        else:
            update_run_status(bench_dir, self.adapter.dataset_name, "failed")
            logger.warning("No metrics collected")
            return None

    # ------------------------------------------------------------------
    # Phase 3
    # ------------------------------------------------------------------
    def phase3_report(self, bench_dir: Path) -> None:
        report_path = bench_dir / "benchmark_report.md"

        with open(bench_dir / "config.yml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        with open(bench_dir / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)

        run_dir = Path(config["phase2"]["run_dir"])
        run_config_path = run_dir / "config.yml"
        run_config = {}
        if run_config_path.exists():
            with open(run_config_path, encoding="utf-8") as f:
                run_config = yaml.safe_load(f)

        per_query = sorted(bench_dir.glob("per_query/[0-9]*"))
        queries_data = []
        for qd in per_query:
            fpath = qd / "filtered_results.json"
            if fpath.exists():
                with open(fpath) as f:
                    queries_data.append(json.load(f))

        pipe = config.get("pipeline", {})
        report_lines = [
            f"# {self.adapter.display_name} Benchmark Report\n",
            f"**Dataset:** {self.adapter.display_name} (`{self.adapter.dataset_name}`)",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Benchmark:** `{bench_dir.name}`",
            f"\n**Run:** `{run_dir.name}`\n",
            f"---\n",
            f"## Configuration\n",
            f"| Parameter | Value |",
            f"|-----------|-------|",
        ]
        for k, v in pipe.items():
            report_lines.append(f"| `{k}` | {v} |")
        report_lines += [
            f"\n| Query range | {config['phase2']['query_start']}-{config['phase2']['query_end'] - 1 if config['phase2']['query_end'] is not None else 'N/A'} |",
            f"| Run docs    | {run_config.get('phase1', {}).get('num_docs', '?')} |",
            f"| Queries     | {summary.get('num_queries', '?')} |\n",
            f"---\n",
            f"## Aggregate Results\n",
            f"| Metric | Mean | Min | Max |",
            f"|--------|------|-----|-----|",
        ]
        for metric in ["mrr", "ap", "p@1", "p@2", "p@3", "p@5", "r@2", "r@5", "ndcg@2", "ndcg@5"]:
            mean_k = f"mean_{metric}"; min_k = f"min_{metric}"; max_k = f"max_{metric}"
            if mean_k in summary:
                report_lines.append(
                    f"| **{metric.upper()}** | {summary[mean_k]:.4f} | "
                    f"{summary[min_k]:.4f} | {summary[max_k]:.4f} |"
                )
        report_lines += [
            f"\n**Queries evaluated:** {summary.get('num_queries', '?')}",
            f"\n**Failed:** {summary.get('failed', 0)}\n",
            f"---\n",
            f"## Per-Query Results\n",
            f"| # | Query | MRR | AP | P@1 | P@2 | R@2 | NDCG@2 | Time |",
            f"|---|-------|-----|-----|-----|-----|-----|--------|------|",
        ]

        not_found = found_r1 = found_r2 = 0
        for qd in queries_data:
            q_idx = qd["query_idx"]
            query_short = qd["query"][:50]
            gold = qd["gold"]
            ranked = qd.get("filtered_ranked", [])
            m = compute_metrics(ranked, gold, [1, 2, 3, 5])
            report_lines.append(
                f"| {q_idx:04d} | {query_short}... | "
                f"{m['mrr']:.3f} | {m['ap']:.3f} | {m['p@1']:.3f} | "
                f"{m['p@2']:.3f} | {m['r@2']:.3f} | {m['ndcg@2']:.3f} | "
                f"{qd.get('elapsed_s', '?'):>5}s |"
            )
            fa = m.get("found_at", 0)
            if fa == 0:
                not_found += 1
            elif fa <= 2:
                found_r2 += 1
            if fa == 1:
                found_r1 += 1

        report_lines += [
            f"\n### Distribution\n",
            f"\n**Found at rank 1:** {found_r1}/{len(queries_data)}",
            f"\n**Found at rank <= 2:** {found_r1 + found_r2}/{len(queries_data)}",
            f"\n**Not found:** {not_found}/{len(queries_data)}\n",
            f"---\n",
            f"*Report generated by `generic_benchmark.py --mode report`*",
        ]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.success(f"Report saved -> {report_path}")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def analyze(self, bench_dir: Path) -> Optional[dict]:
        """Run the same analysis as benchmark_analyzer.py, dataset-aware."""
        per_query = sorted(bench_dir.glob("per_query/[0-9]*"))
        results = []
        for qd in per_query:
            fpath = qd / "filtered_results.json"
            if fpath.exists():
                with open(fpath) as f:
                    results.append(json.load(f))
        if not results:
            logger.error("No per-query results found")
            return None

        with open(bench_dir / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)

        analysis = {
            "dataset": self.adapter.dataset_name,
            "display_name": self.adapter.display_name,
            "benchmark": bench_dir.name,
            "generated_at": datetime.now().isoformat(),
            "num_queries": len(results),
            "summary": summary,
            "metrics_distribution": {},
            "found_at_distribution": defaultdict(int),
            "failures": [],
            "top_performers": [],
        }

        mrr_values = []
        ap_values = []
        p1_values = []
        p2_values = []

        for qd in results:
            q_idx = qd["query_idx"]
            gold = qd.get("gold", [])
            ranked = qd.get("filtered_ranked", [])
            retrieved_ids = [doc_id for doc_id, _ in ranked]
            rel_set = set(gold)

            found_at = 0
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in rel_set:
                    found_at = rank
                    break
            analysis["found_at_distribution"][found_at] += 1

            mrr = 1.0 / found_at if found_at > 0 else 0.0
            mrr_values.append(mrr)

            ap = 0.0
            hits = 0
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in rel_set:
                    hits += 1
                    ap += hits / rank
            ap /= len(gold) if gold else 1
            ap_values.append(ap)

            p1 = 1.0 if any(doc_id in rel_set for doc_id in retrieved_ids[:1]) else 0.0
            p2 = sum(1 for d in retrieved_ids[:2] if d in rel_set) / 2
            p1_values.append(p1)
            p2_values.append(p2)

            if found_at == 0:
                analysis["failures"].append({
                    "query_idx": q_idx,
                    "query": qd.get("query", "")[:80],
                    "num_gold": len(gold),
                    "num_candidates": len(qd.get("candidates", [])),
                })

            if mrr >= 0.9:
                analysis["top_performers"].append({
                    "query_idx": q_idx,
                    "query": qd.get("query", "")[:60],
                    "mrr": mrr,
                    "ap": ap,
                })

        for name, vals in [("mrr", mrr_values), ("ap", ap_values),
                           ("p@1", p1_values), ("p@2", p2_values)]:
            if vals:
                mean_v = sum(vals) / len(vals)
                analysis["metrics_distribution"][name] = {
                    "mean": mean_v,
                    "median": sorted(vals)[len(vals) // 2],
                    "min": min(vals),
                    "max": max(vals),
                    "std": (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5,
                    "num_zero": sum(1 for v in vals if v == 0),
                    "num_perfect": sum(1 for v in vals if v >= 1.0),
                }

        out_path = bench_dir / "analysis.json"
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.success(f"Analysis saved -> {out_path}")
        return analysis


# ============================================================================
# CLI
# ============================================================================
def cli_main():
    parser = argparse.ArgumentParser(
        description="Generic benchmark runner (works with any adapter).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # index
    p_idx = sub.add_parser("index", help="Phase 1: build combined corpus + Steps 1-5")
    p_idx.add_argument("--dataset", required=True)
    p_idx.add_argument("--jsonl", type=Path, required=True, help="Converted MuSiQue-like JSONL")
    p_idx.add_argument("--max-queries", type=int, default=None)
    p_idx.add_argument("--registry", type=Path, default=None,
                       help="Path to dataset_registry.yml for per-dataset parameter overrides")
    p_idx.add_argument("--grid-size", type=int, default=PIPELINE_DEFAULTS["grid_size"])
    p_idx.add_argument("--method", default=PIPELINE_DEFAULTS["method"], choices=["tsne", "umap", "pca", "learned"])
    p_idx.add_argument("--umap-n-neighbors", type=int, default=PIPELINE_DEFAULTS["umap_n_neighbors"])
    p_idx.add_argument("--umap-min-dist", type=float, default=PIPELINE_DEFAULTS["umap_min_dist"])
    p_idx.add_argument("--umap-metric", default=PIPELINE_DEFAULTS["umap_metric"])
    p_idx.add_argument("--spreading-steps", type=int, default=PIPELINE_DEFAULTS["spreading_steps"])
    p_idx.add_argument("--top-percent", type=float, default=PIPELINE_DEFAULTS["top_percent"])
    p_idx.add_argument("--weighting", default=PIPELINE_DEFAULTS["weighting"])
    p_idx.add_argument("--smoothing-sigma", type=float, default=PIPELINE_DEFAULTS["smoothing_sigma"])
    p_idx.add_argument("--no-morton", action="store_true")
    p_idx.add_argument("--glossary", type=str, default=None, help="Path to glossary JSON file")
    p_idx.add_argument("--min-freq", type=int, default=1,
                       help="Min document frequency to keep a phrase (default: 1)")
    p_idx.add_argument("--max-doc-freq", type=int, default=0,
                       help="Max document frequency to keep a phrase (0=unlimited, default: 0)")
    p_idx.add_argument("--glossary-corpus-expansion", action="store_true",
                       help="Inject glossary synonyms into corpus at indexing time (P1.3)")
    p_idx.add_argument("--llm-phrases", action="store_true", default=False,
                       help="Enrich phrase vocabulary with LLM-extracted concepts (Step 0.5)")
    p_idx.add_argument("--llm-batch-size", type=int, default=4,
                       help="Docs per LLM API call during phrase extraction (default: 4)")
    p_idx.add_argument("--prebuilt-vocab", type=Path, default=None,
                       help="Path to a pre-built extracted_phrases/ dir to skip Steps 1 + 0.5")

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Phase 2: run Step 6 per query")
    p_bm.add_argument("--dataset", required=True)
    p_bm.add_argument("--jsonl", type=Path, required=True)
    p_bm.add_argument("--run-dir", type=Path, required=True)
    p_bm.add_argument("--query-start", type=int, default=0)
    p_bm.add_argument("--query-end", type=int, default=None)
    p_bm.add_argument("--registry", type=Path, default=None,
                       help="Path to dataset_registry.yml for per-dataset parameter overrides")
    p_bm.add_argument("--spreading-steps", type=int, default=PIPELINE_DEFAULTS["spreading_steps"])
    p_bm.add_argument("--dynamic-spreading", action="store_true",
                      help="If set, use spreading_steps_long for queries with <= short-query-max-words")
    p_bm.add_argument("--spreading-steps-long", type=int,
                      default=PIPELINE_DEFAULTS["spreading_steps_long"])
    p_bm.add_argument("--short-query-max-words", type=int,
                      default=PIPELINE_DEFAULTS["short_query_max_words"])
    p_bm.add_argument("--geometric", action="store_true",
                      help="Apply 3x3 spatial adjacency kernel to query fingerprint before scoring")
    p_bm.add_argument("--hybrid", action="store_true", help="Enable hybrid SF+BM25 scoring")
    p_bm.add_argument("--hybrid-alpha", type=float, default=0.5, help="SF weight in hybrid mode")
    p_bm.add_argument("--splade", action="store_true", default=True, help="Enable hybrid SF+SPLADE scoring (default: True)")
    p_bm.add_argument("--no-splade", dest="splade", action="store_false", help="Disable SPLADE hybrid scoring")
    p_bm.add_argument("--min-freq", type=int, default=1,
                       help="Min document frequency to keep a phrase (default: 1)")
    p_bm.add_argument("--max-doc-freq", type=int, default=0,
                       help="Max document frequency to keep a phrase (0=unlimited, default: 0)")
    p_bm.add_argument("--splade-model", type=str, default="naver/splade-cocondenser-ensembledistil",
                       help="HuggingFace SPLADE model name")
    p_bm.add_argument("--fusion-method", type=str, default=None, choices=["linear", "rrf"],
                       help="Fusion method for SF+SPLADE: linear or rrf. If not set, uses dataset registry default (or 'linear').")
    p_bm.add_argument("--rrf-k", type=int, default=None,
                       help="Rank constant k for RRF fusion. If not set, uses dataset registry default (or 60).")
    p_bm.add_argument("--multi-resolution", action="store_true",
                       help="Apply multi-resolution spreading (spread at multiple radii and combine)")
    p_bm.add_argument("--doc-norm", type=str, default="l2", choices=["sqrt_nnz", "l2", "l1", "max"])
    p_bm.add_argument("--expand-synonyms", dest="expand_synonyms", action="store_true", default=None,
                       help="Expand query with synonyms from glossary")
    p_bm.add_argument("--glossary", type=str, default=None, help="Path to glossary JSON file")
    p_bm.add_argument("--llm-phrases", dest="llm_phrases", action="store_true", default=None,
                       help="Reference flag: index was built with LLM-enriched vocabulary")
    p_bm.add_argument("--llm-query-phrases", dest="llm_query_phrases", action="store_true", default=None,
                       help="Extract query phrases via LLM at query time for improved match quality")
    p_bm.add_argument("--tfidf-rerank", action="store_true", help="Enable TF-IDF re-ranking")
    p_bm.add_argument("--tfidf-alpha", type=float, default=0.3, help="TF-IDF weight in re-ranking")
    p_bm.add_argument("--corpus", type=Path, default=None, help="Path to corpus.txt for hybrid/tfidf")
    p_bm.add_argument("--sim-metric", type=str, default="cosine",
                       choices=["cosine", "dice", "overlap", "jaccard", "idf-weighted", "spatial_jaccard"],
                       help="Similarity metric for document ranking")
    p_bm.add_argument("--asymmetric", action="store_true", help="Use asymmetric containment/coverage scoring")
    p_bm.add_argument("--asym-alpha", type=float, default=0.7, help="Containment weight in asymmetric mode")
    p_bm.add_argument("--score-norm", type=str, default="none",
                       choices=["none", "zscore", "percentile", "minmax"],
                       help="Score normalization method")
    p_bm.add_argument("--rerank", action="store_true", help="Apply LambdaMART re-ranking")
    p_bm.add_argument("--rerank-model", type=Path, default=None, help="Path to trained re-ranker model")
    p_bm.add_argument("--rerank-top-k", type=int, default=100, help="Number of SF candidates for re-ranking")
    p_bm.add_argument("--negation-aware", action="store_true", help="Apply negation penalty to matching documents")
    p_bm.add_argument("--negation-penalty", type=float, default=0.5, help="Negation penalty weight (0-1)")
    p_bm.add_argument("--negation-boost", type=float, default=0.3, help="Boost for properly negated docs (0-1)")
    p_bm.add_argument("--decompose", action="store_true",
                        help="Enable multi-hop query decomposition")
    p_bm.add_argument("--storage", type=str, default="file", choices=["file", "faiss"],
                        help="Storage backend for fingerprints (default: file)")
    p_bm.add_argument("--faiss-path", type=Path, default=None,
                        help="Path to FAISS index directory (required when --storage faiss)")

    # ── Cross-attention scoring (P2.2) ────────────────────────────────
    p_bm.add_argument("--cross-attention", dest="cross_attention", action="store_true", default=False,
                      help="Use block-level cross-attention scoring")
    p_bm.add_argument("--block-size", type=int, default=PIPELINE_DEFAULTS["block_size"],
                      help="Block size for cross-attention")
    p_bm.add_argument("--attention-temperature", type=float, default=PIPELINE_DEFAULTS["attention_temperature"],
                      help="Temperature for softmax in cross-attention")

    # ── Snippet ranking (P2.4) ───────────────────────────────────────
    p_bm.add_argument("--snippet-ranking", dest="snippet_ranking", action="store_true", default=False,
                      help="Use snippet-level fingerprinting with max-pooling")
    p_bm.add_argument("--snippet-window", type=int, default=PIPELINE_DEFAULTS["snippet_window"],
                      help="Snippet window size in sentences")
    p_bm.add_argument("--snippet-stride", type=int, default=PIPELINE_DEFAULTS["snippet_stride"],
                      help="Stride between snippets")
    p_bm.add_argument("--snippet-dir", type=Path, default=None,
                      help="Path to pre-computed snippet fingerprints")

    # ── Spreading options ────────────────────────────────────────────
    p_bm.add_argument("--adaptive-spreading", action="store_true", default=False,
                      help="Adjust spreading radius based on query length")
    p_bm.add_argument("--normalize-after-spreading", action="store_true", default=False,
                      help="L2-normalise fingerprint after spreading")
    p_bm.add_argument("--spreading-decay", type=float, default=PIPELINE_DEFAULTS["spreading_decay"],
                      help="Decay factor per spreading step")

    # ── Query normalisation ──────────────────────────────────────────
    p_bm.add_argument("--normalization", type=str, default=PIPELINE_DEFAULTS["normalization"],
                      choices=["l2", "l1", "binary", "none"],
                      help="Query fingerprint normalisation method")

    # ── OOV expansion ────────────────────────────────────────────────
    p_bm.add_argument("--no-oov-expansion", dest="no_oov_expansion", action="store_true", default=False,
                      help="Skip OOV term expansion")
    p_bm.add_argument("--synonym-weight", type=float, default=PIPELINE_DEFAULTS["synonym_weight"],
                      help="Weight for synonym-expanded phrases")

    # report
    p_rp = sub.add_parser("report", help="Phase 3: generate markdown report")
    p_rp.add_argument("--dataset", required=True)
    p_rp.add_argument("--benchmark-dir", type=Path, required=True)

    # analyze
    p_an = sub.add_parser("analyze", help="Run deep-dive analysis")
    p_an.add_argument("--dataset", required=True)
    p_an.add_argument("--benchmark-dir", type=Path, required=True)

    # all-in-one
    p_all = sub.add_parser("all", help="Run all three phases end-to-end")
    p_all.add_argument("--dataset", required=True)
    p_all.add_argument("--jsonl", type=Path, required=True)
    p_all.add_argument("--max-queries", type=int, default=None)
    p_all.add_argument("--registry", type=Path, default=None,
                       help="Path to dataset_registry.yml for per-dataset parameter overrides")
    p_all.add_argument("--grid-size", type=int, default=PIPELINE_DEFAULTS["grid_size"])
    p_all.add_argument("--method", default=PIPELINE_DEFAULTS["method"], choices=["tsne", "umap", "pca", "learned"])
    p_all.add_argument("--umap-n-neighbors", type=int, default=PIPELINE_DEFAULTS["umap_n_neighbors"])
    p_all.add_argument("--umap-min-dist", type=float, default=PIPELINE_DEFAULTS["umap_min_dist"])
    p_all.add_argument("--umap-metric", default=PIPELINE_DEFAULTS["umap_metric"])
    p_all.add_argument("--spreading-steps", type=int, default=PIPELINE_DEFAULTS["spreading_steps"])
    p_all.add_argument("--top-percent", type=float, default=PIPELINE_DEFAULTS["top_percent"])
    p_all.add_argument("--weighting", default=PIPELINE_DEFAULTS["weighting"])
    p_all.add_argument("--smoothing-sigma", type=float, default=PIPELINE_DEFAULTS["smoothing_sigma"])
    p_all.add_argument("--no-morton", action="store_true")
    p_all.add_argument("--query-start", type=int, default=0)
    p_all.add_argument("--query-end", type=int, default=None)
    p_all.add_argument("--dynamic-spreading", action="store_true",
                       help="If set, use spreading_steps_long for queries with <= short-query-max-words")
    p_all.add_argument("--spreading-steps-long", type=int,
                       default=PIPELINE_DEFAULTS["spreading_steps_long"])
    p_all.add_argument("--short-query-max-words", type=int,
                       default=PIPELINE_DEFAULTS["short_query_max_words"])
    p_all.add_argument("--geometric", action="store_true",
                       help="Apply 3x3 spatial adjacency kernel to query fingerprint before scoring")
    p_all.add_argument("--hybrid", action="store_true", help="Enable hybrid SF+BM25 scoring")
    p_all.add_argument("--hybrid-alpha", type=float, default=0.5, help="SF weight in hybrid mode")
    p_all.add_argument("--splade", action="store_true", default=True, help="Enable hybrid SF+SPLADE scoring (default: True)")
    p_all.add_argument("--no-splade", dest="splade", action="store_false", help="Disable SPLADE hybrid scoring")
    p_all.add_argument("--min-freq", type=int, default=1,
                       help="Min document frequency to keep a phrase (default: 1)")
    p_all.add_argument("--max-doc-freq", type=int, default=0,
                       help="Max document frequency to keep a phrase (0=unlimited, default: 0)")
    p_all.add_argument("--splade-model", type=str, default="naver/splade-cocondenser-ensembledistil",
                        help="HuggingFace SPLADE model name")
    p_all.add_argument("--fusion-method", type=str, default=None, choices=["linear", "rrf"],
                        help="Fusion method for SF+SPLADE: linear or rrf. If not set, uses dataset registry default (or 'linear').")
    p_all.add_argument("--rrf-k", type=int, default=None,
                        help="Rank constant k for RRF fusion. If not set, uses dataset registry default (or 60).")
    p_all.add_argument("--multi-resolution", action="store_true",
                        help="Apply multi-resolution spreading (spread at multiple radii and combine)")
    p_all.add_argument("--doc-norm", type=str, default="l2", choices=["sqrt_nnz", "l2", "l1", "max"])
    p_all.add_argument("--expand-synonyms", dest="expand_synonyms", action="store_true", default=None,
                       help="Expand query with synonyms from glossary")
    p_all.add_argument("--glossary", type=str, default=None, help="Path to glossary JSON file")
    p_all.add_argument("--glossary-corpus-expansion", dest="glossary_corpus_expansion", action="store_true", default=None,
                        help="Inject glossary synonyms into corpus at indexing time (P1.3)")
    p_all.add_argument("--llm-phrases", dest="llm_phrases", action="store_true", default=None,
                        help="Enrich phrase vocabulary with LLM-extracted concepts (Step 0.5)")
    p_all.add_argument("--llm-query-phrases", dest="llm_query_phrases", action="store_true", default=None,
                        help="Extract query phrases via LLM at query time for improved match quality")
    p_all.add_argument("--llm-batch-size", type=int, default=4,
                        help="Docs per LLM API call during phrase extraction (default: 4)")
    p_all.add_argument("--tfidf-rerank", action="store_true", help="Enable TF-IDF re-ranking")
    p_all.add_argument("--tfidf-alpha", type=float, default=0.3, help="TF-IDF weight in re-ranking")
    p_all.add_argument("--corpus", type=Path, default=None, help="Path to corpus.txt for hybrid/tfidf")
    p_all.add_argument("--sim-metric", type=str, default="cosine",
                        choices=["cosine", "dice", "overlap", "jaccard", "idf-weighted", "spatial_jaccard"],
                        help="Similarity metric for document ranking")
    p_all.add_argument("--adaptive-spreading", action="store_true",
                        help="Adjust spreading radius based on query length")
    p_all.add_argument("--decompose", action="store_true",
                        help="Enable multi-hop query decomposition")
    p_all.add_argument("--storage", type=str, default="file", choices=["file", "faiss"],
                        help="Storage backend for fingerprints (default: file)")
    p_all.add_argument("--faiss-path", type=Path, default=None,
                        help="Path to FAISS index directory (required when --storage faiss)")
    p_all.add_argument("--asymmetric", action="store_true", help="Use asymmetric containment/coverage scoring")
    p_all.add_argument("--asym-alpha", type=float, default=0.7, help="Containment weight in asymmetric mode")
    p_all.add_argument("--score-norm", type=str, default="none",
                        choices=["none", "zscore", "percentile", "minmax"],
                        help="Score normalization method")
    p_all.add_argument("--rerank", action="store_true", help="Apply LambdaMART re-ranking")
    p_all.add_argument("--rerank-model", type=Path, default=None, help="Path to trained re-ranker model")
    p_all.add_argument("--rerank-top-k", type=int, default=100, help="Number of SF candidates for re-ranking")
    p_all.add_argument("--negation-aware", action="store_true", help="Apply negation penalty to matching documents")
    p_all.add_argument("--negation-penalty", type=float, default=0.5, help="Negation penalty weight (0-1)")
    p_all.add_argument("--negation-boost", type=float, default=0.3, help="Boost for properly negated docs (0-1)")

    # ── Cross-attention scoring (P2.2) ────────────────────────────────
    p_all.add_argument("--cross-attention", dest="cross_attention", action="store_true", default=False,
                       help="Use block-level cross-attention scoring")
    p_all.add_argument("--block-size", type=int, default=PIPELINE_DEFAULTS["block_size"],
                       help="Block size for cross-attention")
    p_all.add_argument("--attention-temperature", type=float, default=PIPELINE_DEFAULTS["attention_temperature"],
                       help="Temperature for softmax in cross-attention")

    # ── Snippet ranking (P2.4) ───────────────────────────────────────
    p_all.add_argument("--snippet-ranking", dest="snippet_ranking", action="store_true", default=False,
                       help="Use snippet-level fingerprinting with max-pooling")
    p_all.add_argument("--snippet-window", type=int, default=PIPELINE_DEFAULTS["snippet_window"],
                       help="Snippet window size in sentences")
    p_all.add_argument("--snippet-stride", type=int, default=PIPELINE_DEFAULTS["snippet_stride"],
                       help="Stride between snippets")
    p_all.add_argument("--snippet-dir", type=Path, default=None,
                       help="Path to pre-computed snippet fingerprints")

    # ── Spreading options ────────────────────────────────────────────
    p_all.add_argument("--normalize-after-spreading", action="store_true", default=False,
                       help="L2-normalise fingerprint after spreading")
    p_all.add_argument("--spreading-decay", type=float, default=PIPELINE_DEFAULTS["spreading_decay"],
                       help="Decay factor per spreading step")

    # ── Query normalisation ──────────────────────────────────────────
    p_all.add_argument("--normalization", type=str, default=PIPELINE_DEFAULTS["normalization"],
                       choices=["l2", "l1", "binary", "none"],
                       help="Query fingerprint normalisation method")

    # ── OOV expansion ────────────────────────────────────────────────
    p_all.add_argument("--no-oov-expansion", dest="no_oov_expansion", action="store_true", default=False,
                       help="Skip OOV term expansion")
    p_all.add_argument("--synonym-weight", type=float, default=PIPELINE_DEFAULTS["synonym_weight"],
                       help="Weight for synonym-expanded phrases")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    from .adapters import get_adapter
    adapter = get_adapter(args.dataset)
    
    # ── Load registry defaults (CLI overrides take precedence) ─────────────
    registry_params = {}
    if hasattr(args, "registry") and args.registry:
        registry_params = load_dataset_registry(args.registry, args.dataset)
    elif hasattr(args, "dataset"):
        # Try default registry location
        registry_params = load_dataset_registry(dataset=args.dataset)
    
    # Start with registry defaults, then CLI flags override
    params = dict(registry_params)
    if hasattr(args, "grid_size"):
        params["grid_size"] = args.grid_size
        params["method"] = args.method
        params["umap_n_neighbors"] = args.umap_n_neighbors
        params["umap_min_dist"] = args.umap_min_dist
        params["umap_metric"] = args.umap_metric
        params["spreading_steps"] = args.spreading_steps
        params["top_percent"] = args.top_percent
        params["weighting"] = args.weighting
        params["smoothing_sigma"] = args.smoothing_sigma
        params["morton"] = not args.no_morton
        params["tsne_perplexity"] = args.tsne_perplexity if hasattr(args, "tsne_perplexity") else PIPELINE_DEFAULTS["tsne_perplexity"]
        params["min_freq"] = args.min_freq if hasattr(args, "min_freq") else PIPELINE_DEFAULTS["min_freq"]
        # Only override max_doc_freq if user explicitly passed --max-doc-freq
        # (otherwise preserve registry value which may be > 0)
        if hasattr(args, "max_doc_freq") and args.max_doc_freq != PIPELINE_DEFAULTS["max_doc_freq"]:
            params["max_doc_freq"] = args.max_doc_freq
        params["keep_verbs"] = args.keep_verbs if hasattr(args, "keep_verbs") else PIPELINE_DEFAULTS["keep_verbs"]
    if hasattr(args, "dynamic_spreading"):
        params["dynamic_spreading"] = args.dynamic_spreading
        params["spreading_steps_long"] = args.spreading_steps_long
        params["short_query_max_words"] = args.short_query_max_words
    if hasattr(args, "geometric"):
        params["geometric"] = args.geometric
    if hasattr(args, "hybrid"):
        params["hybrid"] = args.hybrid
        params["hybrid_alpha"] = args.hybrid_alpha
    if "hybrid_alpha" not in params:
        params["hybrid_alpha"] = registry_params.get("splade_alpha", params.get("hybrid_alpha", 0.3))
    if hasattr(args, "splade"):
        # CLI always wins; registry is merely a base default
        params["splade"] = args.splade
        params["splade_model"] = args.splade_model
    if hasattr(args, "fusion_method") and args.fusion_method is not None:
        params["fusion_method"] = args.fusion_method
    if hasattr(args, "rrf_k") and args.rrf_k is not None:
        params["rrf_k"] = args.rrf_k
    if hasattr(args, "multi_resolution"):
        params["multi_resolution"] = args.multi_resolution
    if hasattr(args, "doc_norm"):
        params["doc_norm"] = args.doc_norm
    if hasattr(args, "sim_metric"):
        params["sim_metric"] = args.sim_metric
    if hasattr(args, "asymmetric"):
        params["asymmetric"] = args.asymmetric
        params["asym_alpha"] = args.asym_alpha
    if hasattr(args, "score_norm"):
        params["score_norm"] = args.score_norm
    if hasattr(args, "rerank"):
        params["rerank"] = args.rerank
        params["rerank_model"] = args.rerank_model
        params["rerank_top_k"] = args.rerank_top_k
    if hasattr(args, "negation_aware"):
        params["negation_aware"] = args.negation_aware
        params["negation_penalty"] = args.negation_penalty
        params["negation_boost"] = args.negation_boost
    if hasattr(args, "decompose"):
        params["decompose"] = args.decompose
    if hasattr(args, "storage"):
        params["storage"] = args.storage
    if hasattr(args, "faiss_path") and args.faiss_path:
        params["faiss_path"] = str(args.faiss_path)
    if hasattr(args, "expand_synonyms") and args.expand_synonyms is not None:
        params["expand_synonyms"] = args.expand_synonyms
    if hasattr(args, "glossary_corpus_expansion") and args.glossary_corpus_expansion is not None:
        params["glossary_corpus_expansion"] = args.glossary_corpus_expansion
    if hasattr(args, "llm_phrases") and args.llm_phrases is not None:
        params["llm_phrases"] = args.llm_phrases
    if hasattr(args, "llm_query_phrases") and args.llm_query_phrases is not None:
        params["llm_query_phrases"] = args.llm_query_phrases
    if hasattr(args, "llm_batch_size") and args.llm_batch_size is not None:
        params["llm_batch_size"] = args.llm_batch_size
    if hasattr(args, "prebuilt_vocab") and args.prebuilt_vocab is not None:
        params["prebuilt_vocab"] = str(args.prebuilt_vocab)
    if hasattr(args, "glossary") and args.glossary:
        params["glossary_path"] = args.glossary
    # Also check registry for glossary_path
    if "glossary_path" not in params and "glossary_path" in registry_params:
        params["glossary_path"] = registry_params["glossary_path"]
    if "glossary_corpus_expansion" not in params and "glossary_corpus_expansion" in registry_params:
        params["glossary_corpus_expansion"] = registry_params["glossary_corpus_expansion"]
    if hasattr(args, "tfidf_rerank"):
        params["tfidf_rerank"] = args.tfidf_rerank
        params["tfidf_alpha"] = args.tfidf_alpha
    if hasattr(args, "cross_attention"):
        params["cross_attention"] = args.cross_attention
        params["block_size"] = args.block_size
        params["attention_temperature"] = args.attention_temperature
    if hasattr(args, "snippet_ranking"):
        params["snippet_ranking"] = args.snippet_ranking
        params["snippet_window"] = args.snippet_window
        params["snippet_stride"] = args.snippet_stride
    if hasattr(args, "snippet_dir") and args.snippet_dir:
        params["snippet_dir"] = str(args.snippet_dir)
    if hasattr(args, "adaptive_spreading"):
        params["adaptive_spreading"] = args.adaptive_spreading
    if hasattr(args, "normalize_after_spreading"):
        params["normalize_after_spreading"] = args.normalize_after_spreading
    if hasattr(args, "spreading_decay"):
        params["spreading_decay"] = args.spreading_decay
    if hasattr(args, "normalization"):
        params["normalization"] = args.normalization
    if hasattr(args, "no_oov_expansion") and args.no_oov_expansion:
        params["oov_expansion"] = False
    if hasattr(args, "synonym_weight"):
        params["synonym_weight"] = args.synonym_weight
    if hasattr(args, "corpus") and args.corpus:
        params["corpus_path"] = str(args.corpus)
    elif args.command == "benchmark" and hasattr(args, "run_dir"):
        # Load corpus_path from run config for benchmark subcommand
        run_config_path = Path(args.run_dir) / "config.yml"
        if run_config_path.exists():
            import yaml as _yaml
            with open(run_config_path) as f:
                run_cfg = _yaml.safe_load(f)
            run_pipeline = run_cfg.get("pipeline", {})
            if "corpus_path" in run_pipeline:
                params["corpus_path"] = run_pipeline["corpus_path"]

    runner = GenericBenchmarkRunner(adapter, params)

    if args.command == "index":
        run_dir = runner.phase1_index(args.jsonl, max_queries=args.max_queries)
        if run_dir is None:
            sys.exit(1)
        print(f"\nINDEX_OK:{run_dir}")
    elif args.command == "benchmark":
        bench_dir = runner.phase2_benchmark(
            args.run_dir, args.jsonl,
            query_start=args.query_start, query_end=args.query_end,
        )
        if bench_dir is None:
            sys.exit(1)
        print(f"\nBENCH_OK:{bench_dir}")
    elif args.command == "report":
        runner.phase3_report(args.benchmark_dir)
    elif args.command == "analyze":
        analysis = runner.analyze(args.benchmark_dir)
        if analysis is None:
            sys.exit(1)
        print(json.dumps(analysis["metrics_distribution"], indent=2))
    elif args.command == "all":
        run_dir = runner.phase1_index(args.jsonl, max_queries=args.max_queries)
        if run_dir is None:
            sys.exit(1)
        bench_dir = runner.phase2_benchmark(
            run_dir, args.jsonl,
            query_start=args.query_start, query_end=args.query_end,
        )
        if bench_dir is None:
            sys.exit(1)
        runner.phase3_report(bench_dir)
        runner.analyze(bench_dir)
        print(f"\nALL_OK:{bench_dir}")


if __name__ == "__main__":
    cli_main()
