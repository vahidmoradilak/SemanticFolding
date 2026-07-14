"""
PopQA Adapter

Source: HippoRAG2 datasets (originally from Wikidata KG)
Paper:  Sciavolino et al., 2021 (arXiv:2109.04298)

Format conversion:
  Input row:  { question, answer, paragraphs: [{title, text, is_supporting}], ... }
  Output entry (MuSiQue-like):
    {
      "id": "popqa_000",
      "question": question,
      "answer": answer string,
      "paragraphs": [
        { "idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": true },
        ...
      ]
    }

Relevance convention:
  PopQA provides 2 supporting passages per query (subject entity + object entity).
  Both are marked is_supporting=true.
  The retrieval task: find the gold passage(s) given the question.
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


class PopQAAdapter(BaseDatasetAdapter):
    dataset_name = "popqa"
    display_name = "PopQA"
    default_subset = "default"

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if (output_dir / "popqa.json").exists() and (output_dir / "popqa_corpus.json").exists():
            print(f"  PopQA data already exists at {output_dir}")
            return output_dir

        raise FileNotFoundError(
            f"PopQA data not found in {output_dir}.\n"
            f"Place popqa.json and popqa_corpus.json in {output_dir}/\n"
            f"Source: brain_approaches/hipporag2/datasets/"
        )

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "popqa.jsonl"

        queries_path = raw_path / "popqa.json"
        with open(queries_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        entries = []
        n_written = 0
        n_skipped = 0

        for sample in samples:
            if n_written >= max_queries:
                break

            question = sample.get("question", "").strip()
            answer = sample.get("answer", "").strip()
            raw_paragraphs = sample.get("paragraphs", [])

            if not question or not raw_paragraphs:
                n_skipped += 1
                continue

            paragraphs = []
            for i, p in enumerate(raw_paragraphs):
                text = p.get("text", "").strip()
                if not text:
                    continue
                paragraphs.append({
                    "idx": i,
                    "title": p.get("title", ""),
                    "paragraph_text": text,
                    "is_supporting": p.get("is_supporting", False),
                })

            if not paragraphs:
                n_skipped += 1
                continue

            entries.append({
                "id": f"popqa_{n_written:04d}",
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
        print(f"  PopQA: wrote {n_written} queries -> {out_path} (skipped {n_skipped})")
        return out_path
