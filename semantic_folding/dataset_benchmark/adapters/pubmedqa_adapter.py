"""
PubMedQA Adapter

Source: HuggingFace qiaojin/PubMedQA (parquet) or
        GitHub pubmedqa/pubmedqa (raw json)
Paper:  Jin et al., 2019 (arXiv:1909.06146)

Format conversion:
  Input row:  { pubid, question, context: {contexts:[...], labels:[...]}, long_answer, final_decision }
  Output entry (MuSiQue-like):
    {
      "id": f"pubmedqa_{pubid}",
      "question": question,
      "answer": long_answer or final_decision,
      "final_decision": "yes" | "no" | "maybe",
      "paragraphs": [
        { "idx": i, "title": labels[i], "paragraph_text": contexts[i], "is_supporting": bool },
        ...
      ]
    }

Relevance convention:
  - For "yes" queries: ALL passages are marked is_supporting=True (the abstract
    supports the answer; passage retrieval is the task).
  - For "no" / "maybe" queries: passages are still candidates but not supporting.

This mirrors the MuSiQue benchmark: the gold standard is the full set of
passages that the query is answerable from. The retrieval task is to rank
these above distractors (which come from other queries, via the combined
corpus strategy of phase 1).

Per-dataset parameter overrides (if any) come from get_recommended_params().
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from .base_adapter import BaseDatasetAdapter

# Force UTF-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class PubMedQAAdapter(BaseDatasetAdapter):
    dataset_name = "pubmedqa"
    display_name = "PubMedQA"
    default_subset = "pqa_labeled"

    PARQUET_FILES = {
        "pqa_labeled": "pqa_labeled/train-00000-of-00001.parquet",
        "pqa_artificial": "pqa_artificial/train-00000-of-00001.parquet",
        "pqa_unlabeled": "pqa_unlabeled/train-00000-of-00001.parquet",
    }

    def _resolve_parquet(self, raw_path: Path, subset: str) -> Path:
        """Return the parquet path for a given subset (or default)."""
        if subset not in self.PARQUET_FILES:
            raise ValueError(
                f"Unknown PubMedQA subset: {subset}. "
                f"Available: {list(self.PARQUET_FILES)}"
            )
        p = raw_path / self.PARQUET_FILES[subset]
        if not p.exists():
            raise FileNotFoundError(
                f"Parquet not found: {p}\n"
                f"Place it under data/pubmedqa/raw/{self.PARQUET_FILES[subset]}"
            )
        return p

    def download(self, output_dir: Path) -> Path:
        """
        PubMedQA parquets are expected to be placed manually under output_dir.
        Validates that they exist; raises FileNotFoundError otherwise.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        found = []
        for sub, rel in self.PARQUET_FILES.items():
            p = output_dir / rel
            if p.exists() and p.stat().st_size > 1000:
                found.append(sub)
        if not found:
            raise FileNotFoundError(
                f"No PubMedQA parquet files found under {output_dir}.\n"
                f"Download from https://huggingface.co/datasets/qiaojin/PubMedQA "
                f"and place into: {output_dir}/pqa_labeled/, pqa_artificial/, pqa_unlabeled/"
            )
        return output_dir

    def convert_to_musique_format(
        self,
        raw_path: Path,
        output_dir: Path,
        max_queries: int = 500,
        subset: str = None,
    ) -> Path:
        """
        Convert PubMedQA parquet to MuSiQue-like JSONL.

        Parameters
        ----------
        raw_path : Path
            Directory containing pqa_<subset>/train-...parquet
        output_dir : Path
            Where to write the JSONL.
        max_queries : int
            Cap the number of queries to include (for benchmark).
        subset : str, optional
            Which subset to use. Default: pqa_labeled (gold standard, smallest).
        """
        import pyarrow.parquet as pq

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        subset = subset or self.default_subset
        parquet = self._resolve_parquet(raw_path, subset)
        table = pq.read_table(parquet)
        rows = table.to_pylist()
        print(f"  loaded {len(rows)} rows from {parquet.name}")

        # Filter to "yes" queries for the gold set (these are the answerable
        # ones with full human-annotated context). We also keep "no" / "maybe"
        # as distractors in the candidate pool later.
        # For binary retrieval: gold = yes; no/maybe = negative examples.
        out_path = output_dir / f"pubmedqa_{subset}.jsonl"

        n_written = 0
        n_skipped = 0
        n_yes = 0
        n_no = 0
        n_maybe = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                ctx = row.get("context", {})
                contexts: List[str] = ctx.get("contexts", [])
                labels: List[str] = ctx.get("labels", [])
                decision: str = row.get("final_decision", "").strip().lower()
                if not contexts or not decision:
                    n_skipped += 1
                    continue
                if n_written >= max_queries:
                    break
                # Build MuSiQue-like paragraphs
                # All sections are "supporting" because the abstract IS the
                # supporting context for the query. For "no"/"maybe" we
                # still include the query but mark non-supporting (the answer
                # is not derivable from the text alone).
                is_supporting = decision == "yes"
                paragraphs = []
                for i, txt in enumerate(contexts):
                    title = labels[i] if i < len(labels) else f"section_{i}"
                    paragraphs.append({
                        "idx": i,
                        "title": title,
                        "paragraph_text": txt,
                        "is_supporting": is_supporting,
                    })
                entry = {
                    "id": f"pubmedqa_{row['pubid']}",
                    "question": row["question"],
                    "answer": row.get("long_answer", "") or decision,
                    "final_decision": decision,
                    "paragraphs": paragraphs,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written += 1
                if decision == "yes":
                    n_yes += 1
                elif decision == "no":
                    n_no += 1
                elif decision == "maybe":
                    n_maybe += 1

        stats = {
            "subset": subset,
            "num_queries": n_written,
            "num_yes": n_yes,
            "num_no": n_no,
            "num_maybe": n_maybe,
            "num_skipped": n_skipped,
            "out_path": str(out_path),
        }
        # Sidecar stats file for the orchestrator / final-datasets.md
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  wrote {n_written} entries -> {out_path}")
        print(f"  yes={n_yes}, no={n_no}, maybe={n_maybe}, skipped={n_skipped}")
        return out_path
