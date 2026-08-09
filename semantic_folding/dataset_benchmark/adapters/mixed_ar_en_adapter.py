import csv, json, random, re, sys
from pathlib import Path
from typing import List, Dict, Any

from .base_adapter import BaseDatasetAdapter
from lib import detect_language, extract_raw_phrases_ar_fa, normalize_arabic_phrase

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    nlp = None

_SEP = " | "


def _clean_ar_phrase(p: str) -> str:
    p = normalize_arabic_phrase(p)
    if not p:
        return ""
    p = re.sub(r'[^\w\s\u0600-\u06FF]', '', p).strip()
    return p


def _extract_ar_phrases(text: str) -> List[str]:
    raw = extract_raw_phrases_ar_fa(text)
    valid = []
    for p in raw:
        clean = _clean_ar_phrase(p)
        if clean and len(clean) >= 2:
            valid.append(clean)
    return valid


def _extract_en_phrases(text: str) -> List[str]:
    if not SPACY_OK or not nlp:
        return text.split()[:10]
    from phrase_extractor import extract_raw_phrases_spacy, normalize_hyphens
    clean = normalize_hyphens(text)
    doc = nlp(clean)
    return extract_raw_phrases_spacy(doc)


def _make_mixed_query(question: str) -> str:
    parts = question.split(_SEP, maxsplit=1)
    if len(parts) < 2:
        return question
    ar_q, en_q = parts[0].strip(), parts[1].strip()
    ar_phrases = _extract_ar_phrases(ar_q)
    en_phrases = _extract_en_phrases(en_q)
    mixed = " ".join(ar_phrases + en_phrases)
    return mixed if mixed.strip() else question


class MixedArEnAdapter(BaseDatasetAdapter):
    dataset_name = "mixed_ar_en"
    display_name = "Mixed Arabic-English (Bilingual)"

    @property
    def default_subset(self) -> str:
        return "all"

    def get_recommended_params(self) -> Dict[str, Any]:
        return {
            "grid_size": 64,
            "spreading_steps": 1,
            "top_percent": 0.10,
            "weighting": "idf",
            "smoothing_sigma": 1.5,
            "morton": True,
            "min_word_length": 2,
            "min_freq": 1,
            "keep_verbs": True,
            "top_k": 20,
            "tsne_perplexity": 50,
            "tsne_iter": 1000,
            "method": "umap",
        }

    def download(self, output_dir: Path) -> Path:
        src = output_dir / "corpus_belebele_ar_en_deduped.csv"
        if not src.exists():
            raise FileNotFoundError(f"Deduped CSV not found at {src}")
        return output_dir

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "mixed_ar_en.jsonl"
        stats_path = output_dir / "mixed_ar_en.stats.json"

        csv_path = raw_path / "corpus_belebele_ar_en_deduped.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Deduped CSV not found: {csv_path}")

        rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        random.seed(42)
        all_passages = [r["passage"] for r in rows]
        n = min(len(rows), max_queries)

        entries = []
        n_skipped = 0

        for i in range(n):
            row = rows[i]
            passage = row["passage"]
            question = row["question_1"]
            if not passage or not question:
                n_skipped += 1
                continue

            mixed_q = _make_mixed_query(question)

            # 20 distractors (other passages)
            others = [p for j, p in enumerate(all_passages) if j != i]
            distractors = random.sample(others, min(20, len(others)))

            paragraphs = [{
                "idx": 0, "title": "passage",
                "paragraph_text": passage,
                "is_supporting": True,
            }]
            for j, dp in enumerate(distractors):
                paragraphs.append({
                    "idx": j + 1, "title": f"distractor_{j:04d}",
                    "paragraph_text": dp,
                    "is_supporting": False,
                })

            entries.append({
                "id": f"mixed_ar_en_{i:04d}",
                "question": mixed_q,
                "answer": passage,
                "paragraphs": paragraphs,
            })

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": len(entries), "n_skipped": n_skipped, "total_rows": n}
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  MixedArEn: wrote {len(entries)} queries -> {out_path} (skipped {n_skipped})")
        return out_path
