#!/usr/bin/env python3
"""Build English-text QA file: 300 random English ayah translations as queries, gold-only."""
import csv
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "quran"
CORPUS_PATH = DATA_DIR / "quran_ayahs_clean.txt"
OUT_PATH = DATA_DIR / "quran_random_300_en_qa.jsonl"
TXT_PATH = DATA_DIR / "quran_random_300_en.txt"
NUM = 300

# Load corpus
corpus = []
with open(CORPUS_PATH, "r", encoding="utf8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 3:
            corpus.append((row[0].strip(), row[1].strip(), row[2].strip()))

print(f"Loaded {len(corpus)} ayahs")

# Pick same 300 random ayahs (seed=42)
random.seed(42)
chosen = random.sample(corpus, min(NUM, len(corpus)))

# Write English texts to file
with open(TXT_PATH, "w", encoding="utf8") as f:
    for idx, arabic, english in chosen:
        f.write(english + "\n")
print(f"Wrote {len(chosen)} English texts to {TXT_PATH}")

# Build QA entries
entries = []
for i, (idx, arabic, english) in enumerate(chosen):
    entries.append({
        "id": f"RE{i+1:03d}",
        "category": "random_en",
        "question": english,
        "relevant": [idx],
        "relevance": {idx: 2},
    })

with open(OUT_PATH, "w", encoding="utf8") as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"Wrote {len(entries)} entries to {OUT_PATH}")
print("Done.")
