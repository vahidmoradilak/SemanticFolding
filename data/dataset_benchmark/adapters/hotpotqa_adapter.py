"""
HotpotQA Adapter

Source: HippoRAG2 datasets (originally from HotpotQA)
Paper:  Yang et al., 2018 (arXiv:1808.09060)

Supporting facts: 2-7 per query (multi-hop)
We evaluate retrieval of ALL gold passages, not just the first hop.
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


class HotpotQAAdapter(BaseDatasetAdapter):
    dataset_name = "hotpotqa"
    display_name = "HotpotQA"
    default_subset = "distractor"

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "hotpotqa.json").exists():
            print(f"  HotpotQA data already exists at {output_dir}")
            return output_dir
        raise FileNotFoundError(
            f"HotpotQA data not found in {output_dir}.\n"
            f"Place hotpotqa.json and hotpotqa_corpus.json in {output_dir}/"
        )

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "hotpotqa.jsonl"

        with open(raw_path / "hotpotqa.json", "r", encoding="utf-8") as f:
            samples = json.load(f)

        entries = []
        n_written = 0

        for sample in samples:
            if n_written >= max_queries:
                break

            question = sample.get("question", "").strip()
            answer = sample.get("answer", "").strip()
            context = sample.get("context", [])
            supporting_facts = sample.get("supporting_facts", [])

            if not question or not context:
                continue

            gold_titles = set(sf[0] for sf in supporting_facts)

            paragraphs = []
            for i, (title, sentences) in enumerate(context):
                text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                paragraphs.append({
                    "idx": i,
                    "title": title,
                    "paragraph_text": text,
                    "is_supporting": title in gold_titles,
                })

            if not paragraphs:
                continue

            entries.append({
                "id": f"hotpotqa_{n_written:04d}",
                "question": question,
                "answer": answer,
                "type": sample.get("type", ""),
                "level": sample.get("level", ""),
                "paragraphs": paragraphs,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": n_written, "total_rows": len(samples)}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  HotpotQA: wrote {n_written} queries -> {out_path}")
        return out_path
