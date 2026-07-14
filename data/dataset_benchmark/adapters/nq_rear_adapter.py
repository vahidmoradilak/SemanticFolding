"""
NQ-REaR Adapter

Source: HippoRAG2 datasets (originally from Google Natural Questions)
Paper:  Karpukhin et al., 2020 (DPR paper, arXiv:2004.04906)

Format conversion:
  Input row:  { question, reference, contexts: [{title, text, is_supporting}], ... }
  Output entry (MuSiQue-like):
    {
      "id": "nq_rear_000",
      "question": question,
      "answer": reference list,
      "paragraphs": [
        { "idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": true },
        ...
      ]
    }

Relevance convention:
  NQ-REaR provides contexts with is_supporting flags.
  The retrieval task: find the supporting passage(s) given the question.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from .base_adapter import BaseDatasetAdapter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class NQRearAdapter(BaseDatasetAdapter):
    dataset_name = "nq_rear"
    display_name = "NQ-REaR"
    default_subset = "default"

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if (output_dir / "nq_rear.json").exists() and (output_dir / "nq_rear_corpus.json").exists():
            print(f"  NQ-REaR data already exists at {output_dir}")
            return output_dir

        raise FileNotFoundError(
            f"NQ-REaR data not found in {output_dir}.\n"
            f"Place nq_rear.json and nq_rear_corpus.json in {output_dir}/\n"
            f"Source: brain_approaches/hipporag2/datasets/"
        )

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "nq_rear.jsonl"

        queries_path = raw_path / "nq_rear.json"
        with open(queries_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        entries = []
        n_written = 0
        n_skipped = 0

        for sample in samples:
            if n_written >= max_queries:
                break

            question = sample.get("question", "").strip()
            reference = sample.get("reference", [])
            raw_contexts = sample.get("contexts", [])

            if not question or not raw_contexts:
                n_skipped += 1
                continue

            paragraphs = []
            for i, ctx in enumerate(raw_contexts):
                text = ctx.get("text", "").strip()
                if not text:
                    continue
                paragraphs.append({
                    "idx": i,
                    "title": ctx.get("title", ""),
                    "paragraph_text": text,
                    "is_supporting": ctx.get("is_supporting", False),
                })

            if not paragraphs:
                n_skipped += 1
                continue

            answer = reference if isinstance(reference, list) else [reference]

            entries.append({
                "id": f"nq_rear_{n_written:04d}",
                "question": question,
                "answer": answer,
                "paragraphs": paragraphs,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": n_written, "num_skipped": n_skipped, "total_rows": len(samples)}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  NQ-REaR: wrote {n_written} queries -> {out_path} (skipped {n_skipped})")
        return out_path
