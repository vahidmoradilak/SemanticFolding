"""
BioASQ Adapter

Supports two sources:
  1. BigBIO HuggingFace (bigbio/bioasq_task_b) — preferred, no registration
  2. Official BioASQ (https://participants-area.bioasq.org/datasets/) — requires registration

BigBIO schema:
  Input row:  { id, question, type, choices, context: [{text, sections}], answer }

Official BioASQ Task B schema:
  Input: JSON with "data" list of questions. Each question:
    { "body": "...", "type": "yesno|list|factoid",
      "documents": [ {"pmid": "...", "snippets": [{"offset": ..., "text": "..."}]} ],
      "exact_answer": [...], "ideal_answer": "..." }

Output entry (MuSiQue-like):
  Each context/snippet passage becomes a candidate paragraph.
  Passages that match the answer are marked as gold.
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


class BioASQAdapter(BaseDatasetAdapter):
    dataset_name = "bioasq"
    display_name = "BioASQ"
    default_subset = "bioasq_task_b_source"

    CONFIG_CANDIDATES = [
        "bioasq_task_b_source",
        "bioasq_task_b_abstracts",
        "bioasq_task_b_large",
    ]

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_path = output_dir / "bioasq_cache.jsonl"
        if cache_path.exists() and cache_path.stat().st_size > 100:
            print(f"  BioASQ data already cached at {cache_path}")
            return output_dir

        # Check for official BioASQ format files in raw dir
        official_files = list(output_dir.glob("*.json")) + list(output_dir.glob("*.jsonl"))
        if official_files:
            print(f"  Found official BioASQ files: {[f.name for f in official_files]}")
            return output_dir

        # Try BigBIO HuggingFace
        try:
            from datasets import load_dataset
            last_error = None
            for config in self.CONFIG_CANDIDATES:
                try:
                    print(f"  Trying bigbio/bioasq_task_b with config={config}...")
                    ds = load_dataset("bigbio/bioasq_task_b", config, split="test")
                    print(f"  Success with config={config}, {len(ds)} rows")
                    with open(cache_path, "w", encoding="utf-8") as f:
                        for row in ds:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"  Cached {len(ds)} rows -> {cache_path}")
                    return output_dir
                except Exception as e:
                    last_error = e
                    print(f"  Config {config} failed: {e}")
                    continue
            raise FileNotFoundError(f"All BigBIO configs failed. Last: {last_error}")
        except FileNotFoundError:
            raise
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download BioASQ: {e}\n"
                f"Options:\n"
                f"  1. Register at https://participants-area.bioasq.org/accounts/register/\n"
                f"     and download Task B training data to {output_dir}/\n"
                f"  2. Or download from https://huggingface.co/datasets/bigbio/bioasq_task_b"
            )

    def _load_official_format(self, json_path: Path) -> List[dict]:
        """Load official BioASQ Task B JSON format."""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        questions = data if isinstance(data, list) else data.get("data", data.get("questions", []))
        rows = []
        for q in questions:
            question_text = q.get("body", q.get("question", "")).strip()
            if not question_text:
                continue

            answer_parts = q.get("exact_answer", [])
            if isinstance(answer_parts, str):
                answer_parts = [answer_parts] if answer_parts else []
            ideal_answer = q.get("ideal_answer", "")
            answer_text = "; ".join(str(a) for a in answer_parts) if answer_parts else str(ideal_answer)

            passages = []
            seen = set()
            for doc in q.get("documents", []):
                pmid = doc.get("pmid", "")
                for snippet in doc.get("snippets", []):
                    text = snippet.get("text", "").strip()
                    if text and text not in seen:
                        seen.add(text)
                        passages.append({
                            "title": f"PMID:{pmid}",
                            "text": text,
                        })

            if not passages:
                abstract = q.get("abstract", "")
                if abstract:
                    passages.append({"title": "abstract", "text": abstract})

            if passages:
                rows.append({
                    "id": q.get("id", q.get("_id", "")),
                    "question": question_text,
                    "type": q.get("type", ""),
                    "answer": answer_text,
                    "context": passages,
                })
        return rows

    def _load_bigbio_format(self, cache_path: Path) -> List[dict]:
        """Load BigBIO JSONL format."""
        rows = []
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "bioasq.jsonl"

        cache_path = raw_path / "bioasq_cache.jsonl"
        official_files = list(raw_path.glob("*.json"))

        if cache_path.exists():
            rows = self._load_bigbio_format(cache_path)
            print(f"  Loaded {len(rows)} rows from BigBIO cache")
        elif official_files:
            rows = []
            for jf in official_files:
                rows.extend(self._load_official_format(jf))
            print(f"  Loaded {len(rows)} rows from official BioASQ files")
        else:
            raise FileNotFoundError("No BioASQ data found. Run download first.")

        if not rows:
            raise ValueError("No valid rows found in BioASQ data")

        sample = rows[0]
        print(f"  Sample fields: {list(sample.keys())}")

        random.seed(42)
        all_passages = []
        for row in rows:
            ctx = row.get("context", [])
            if isinstance(ctx, list):
                for p in ctx:
                    if isinstance(p, dict):
                        text = p.get("text", "")
                    elif isinstance(p, str):
                        text = p
                    else:
                        continue
                    if text.strip():
                        all_passages.append(text.strip())

        entries = []
        n_written = 0
        n_skipped = 0

        for row in rows:
            if n_written >= max_queries:
                break

            question = row.get("question", "").strip()
            if not question:
                n_skipped += 1
                continue

            answer_text = row.get("answer", "")
            ctx = row.get("context", [])
            if not isinstance(ctx, list) or not ctx:
                n_skipped += 1
                continue

            candidate_passages = []
            for p in ctx:
                if isinstance(p, dict):
                    text = p.get("text", "").strip()
                    title = p.get("title", "context")
                elif isinstance(p, str):
                    text = p.strip()
                    title = "context"
                else:
                    continue
                if text:
                    candidate_passages.append((title, text))

            if not candidate_passages:
                n_skipped += 1
                continue

            answer_lower = answer_text.lower()
            paragraphs = []
            for i, (title, text) in enumerate(candidate_passages):
                is_gold = answer_lower and answer_lower in text.lower()
                paragraphs.append({
                    "idx": i,
                    "title": title,
                    "paragraph_text": text,
                    "is_supporting": is_gold,
                })

            if not any(p["is_supporting"] for p in paragraphs) and answer_text:
                for p in paragraphs:
                    if answer_text.lower() in p["paragraph_text"].lower():
                        p["is_supporting"] = True
                        break

            n_existing = len(paragraphs)
            if n_existing < 20 and all_passages:
                distractor_texts = random.sample(
                    all_passages, min(20 - n_existing, len(all_passages))
                )
                for dt in distractor_texts:
                    paragraphs.append({
                        "idx": len(paragraphs),
                        "title": "distractor",
                        "paragraph_text": dt,
                        "is_supporting": False,
                    })

            entries.append({
                "id": row.get("id", f"bioasq_{n_written:04d}"),
                "question": question,
                "answer": answer_text,
                "qa_type": row.get("type", ""),
                "paragraphs": paragraphs,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": n_written, "num_skipped": n_skipped, "total_rows": len(rows)}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  BioASQ: wrote {n_written} queries -> {out_path} (skipped {n_skipped})")
        return out_path
