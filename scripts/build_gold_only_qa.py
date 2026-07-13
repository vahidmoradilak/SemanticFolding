#!/usr/bin/env python3
"""Create gold-only QA files: keep only reference ayah (relevance=2) as the single relevant doc."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "quran"

for base_name in ["quran_qa_semantic", "quran_random_300_qa"]:
    src = DATA_DIR / f"{base_name}.jsonl"
    dst = DATA_DIR / f"{base_name}_gold_only.jsonl"
    entries = []
    with open(src, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qa = json.loads(line)
            # Find the reference ayah (the one with relevance=2)
            ref_ayah = None
            for doc_id, rel in qa["relevance"].items():
                if rel == 2:
                    ref_ayah = doc_id
                    break
            if ref_ayah is None:
                continue
            entries.append({
                "id": qa["id"],
                "category": qa["category"],
                "question": qa["question"],
                "relevant": [ref_ayah],
                "relevance": {ref_ayah: 2},
            })
    with open(dst, "w", encoding="utf8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} entries to {dst}")

print("Done.")
