"""
Belebele Adapter

Source: HuggingFace facebook/belebele (english split)
Paper:  Malayi et al., 2023 (arXiv:2308.16884)

Format conversion:
  Input row:  { flores_passage, question, mc_answer1..4, correct_answer_num, ... }
  Output entry (MuSiQue-like):
    {
      "id": "belebele_000",
      "question": question,
      "answer": mc_answer text,
      "paragraphs": [
        { "idx": 0, "title": "passage", "paragraph_text": flores_passage, "is_supporting": True },
        { "idx": 1..19, "title": "distractor_N", "paragraph_text": ..., "is_supporting": False }
      ]
    }

Relevance convention:
  The gold passage is the one the question was written about.
  Distractor passages come from other questions.
  The retrieval task: find the correct passage given the question.
"""

import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Any

from .base_adapter import BaseDatasetAdapter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class BelebeleAdapter(BaseDatasetAdapter):
    dataset_name = "belebele"
    display_name = "Belebele"
    default_subset = "english"

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_path = output_dir / "belebele_cache.jsonl"
        if cache_path.exists() and cache_path.stat().st_size > 100:
            print(f"  Belebele data already cached at {cache_path}")
            return output_dir

        extracted = output_dir / "extracted" / "eng_Latn.jsonl"
        if extracted.exists():
            import shutil
            shutil.copy2(extracted, cache_path)
            print(f"  Copied extracted data -> {cache_path}")
            return output_dir

        try:
            from datasets import load_dataset
            print("  Downloading facebook/belebele (english) from HuggingFace...")
            ds = load_dataset("facebook/belebele", self.default_subset, split="test")
            with open(cache_path, "w", encoding="utf-8") as f:
                for row in ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  Cached {len(ds)} rows -> {cache_path}")
            return output_dir
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download Belebele: {e}\n"
                f"Either:\n"
                f"  1. Place eng_Latn.jsonl in {output_dir}/extracted/\n"
                f"  2. Or download from https://huggingface.co/datasets/facebook/belebele"
            )

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "belebele.jsonl"

        cache_path = raw_path / "belebele_cache.jsonl"
        if not cache_path.exists():
            extracted = raw_path / "extracted" / "eng_Latn.jsonl"
            if extracted.exists():
                cache_path = extracted
            else:
                raise FileNotFoundError(
                    f"Belebele data not found. Run download first."
                )

        rows = []
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        random.seed(42)
        all_passages = [r.get("flores_passage", "") for r in rows]

        answer_map = {1: "mc_answer1", 2: "mc_answer2", 3: "mc_answer3", 4: "mc_answer4"}

        entries = []
        n_written = 0
        n_skipped = 0

        for row in rows:
            if n_written >= max_queries:
                break

            passage = row.get("flores_passage", "").strip()
            question = row.get("question", "").strip()
            correct_num = row.get("correct_answer_num", 0)

            if not passage or not question or not correct_num:
                n_skipped += 1
                continue

            correct_key = answer_map.get(correct_num, "mc_answer1")
            mc_answer = row.get(correct_key, "").strip()
            if not mc_answer:
                n_skipped += 1
                continue

            distractor_indices = random.sample(
                range(len(all_passages)),
                min(19, len(all_passages) - 1),
            )
            distractor_indices = [i for i in distractor_indices if all_passages[i] != passage][:19]

            paragraphs = [{
                "idx": 0,
                "title": "passage",
                "paragraph_text": passage,
                "is_supporting": True,
            }]
            for i, di in enumerate(distractor_indices):
                paragraphs.append({
                    "idx": i + 1,
                    "title": f"distractor_{i:04d}",
                    "paragraph_text": all_passages[di],
                    "is_supporting": False,
                })

            entries.append({
                "id": f"belebele_{n_written:04d}",
                "question": question,
                "answer": mc_answer,
                "answer_num": correct_num,
                "dialect": row.get("dialect", ""),
                "paragraphs": paragraphs,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": n_written, "num_skipped": n_skipped, "total_rows": len(rows)}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Belebele: wrote {n_written} queries -> {out_path} (skipped {n_skipped})")
        return out_path
