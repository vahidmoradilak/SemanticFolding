"""
NarrativeQA Adapter

Source: HippoRAG2 datasets (originally from DeepMind NarrativeQA)
Paper:  Kočiský et al., 2018 (arXiv:1712.07040)

Format:
  Input: { document: {id, text, summary, ...}, question, answer }
  Output: MuSiQue-like JSONL with document chunks as passages

Task: Given a question about a movie script, find the relevant passage.
Unlike other datasets, each question belongs to one document.
The corpus contains chunks of all documents.
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


class NarrativeQAAdapter(BaseDatasetAdapter):
    dataset_name = "narrativeqa"
    display_name = "NarrativeQA"
    default_subset = "dev"

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        queries_file = output_dir / "narrativeqa_dev_10_doc.json"
        corpus_file = output_dir / "narrativeqa_dev_10_doc_corpus.json"

        if queries_file.exists() and corpus_file.exists():
            print(f"  NarrativeQA data already exists at {output_dir}")
            return output_dir

        raise FileNotFoundError(
            f"NarrativeQA data not found in {output_dir}.\n"
            f"Place narrativeqa_dev_10_doc.json and corpus in {output_dir}/\n"
            f"Source: brain_approaches/hipporag2/datasets/"
        )

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "narrativeqa.jsonl"

        # Load queries
        queries_path = raw_path / "narrativeqa_dev_10_doc.json"
        with open(queries_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        # Load corpus to get passage texts
        corpus_path = raw_path / "narrativeqa_dev_10_doc_corpus.json"
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        # Build doc_id -> passages mapping
        doc_passages = {}
        for passage in corpus:
            idx = passage.get("idx", "")
            # idx format: "doc_hash_N" where N is chunk number
            doc_id = "_".join(idx.split("_")[:-1]) if "_" in idx else idx
            if doc_id not in doc_passages:
                doc_passages[doc_id] = []
            doc_passages[doc_id].append({
                "idx": len(doc_passages[doc_id]),
                "title": passage.get("title", ""),
                "paragraph_text": passage.get("text", ""),
                "is_supporting": False,  # Will mark relevant ones below
            })

        entries = []
        n_written = 0
        n_skipped = 0

        for sample in samples:
            if n_written >= max_queries:
                break

            question = sample.get("question", "").strip()
            answer = sample.get("answer", [])
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            
            doc_info = sample.get("document", {})
            doc_id = doc_info.get("id", "")

            if not question or not doc_id:
                n_skipped += 1
                continue

            # Get passages for this document
            passages = doc_passages.get(doc_id, [])
            if not passages:
                # Try to find by partial match
                for did, ps in doc_passages.items():
                    if did.startswith(doc_id[:20]):
                        passages = ps
                        break

            if not passages:
                n_skipped += 1
                continue

            # For NarrativeQA, we don't have per-passage gold labels.
            # Mark the first passage as "supporting" (approximation)
            # In reality, the answer could be in any chunk of the document.
            # We'll mark passages that contain answer keywords as supporting.
            answer_lower = answer.lower() if answer else ""
            
            for p in passages:
                text_lower = p["paragraph_text"].lower()
                # Simple heuristic: if answer words appear in passage
                if answer_lower and any(w in text_lower for w in answer_lower.split() if len(w) > 3):
                    p["is_supporting"] = True

            # If no passage marked as supporting, mark first one
            if not any(p["is_supporting"] for p in passages):
                passages[0]["is_supporting"] = True

            # Add distractor passages from other documents
            import random
            random.seed(42)
            other_passages = []
            for other_id, other_ps in doc_passages.items():
                if other_id != doc_id:
                    other_passages.extend(other_ps[:2])  # Take up to 2 from each

            if other_passages:
                n_distractors = min(5, len(other_passages))
                distractors = random.sample(other_passages, n_distractors)
                for i, d in enumerate(distractors):
                    passages.append({
                        "idx": len(passages),
                        "title": d["title"],
                        "paragraph_text": d["paragraph_text"],
                        "is_supporting": False,
                    })

            entries.append({
                "id": f"narrativeqa_{n_written:04d}",
                "question": question,
                "answer": answer,
                "document_id": doc_id,
                "paragraphs": passages,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": n_written, "num_skipped": n_skipped, "total_rows": len(samples)}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  NarrativeQA: wrote {n_written} queries -> {out_path} (skipped {n_skipped})")
        return out_path
