"""
2WikiMultihopQA Adapter

Source: HippoRAG2 datasets
Paper:  Ho et al., 2020 (arXiv:2009.06056)

Multi-hop compositional QA with supporting facts.
"""
import json, sys
from pathlib import Path
from .base_adapter import BaseDatasetAdapter
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class TwoWikiMultihopQAAdapter(BaseDatasetAdapter):
    dataset_name = "2wikimultihopqa"
    display_name = "2WikiMultihopQA"
    default_subset = "default"

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "2wikimultihopqa.json").exists():
            return output_dir
        raise FileNotFoundError(f"2WikiMultihopQA data not found in {output_dir}")

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "2wikimultihopqa.jsonl"

        with open(raw_path / "2wikimultihopqa.json", "r", encoding="utf-8") as f:
            samples = json.load(f)

        entries = []
        n_written = 0

        for sample in samples:
            if n_written >= max_queries:
                break

            question = sample.get("question", "").strip()
            answer = sample.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
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
                "id": f"2wiki_{n_written:04d}",
                "question": question,
                "answer": str(answer),
                "type": sample.get("type", ""),
                "paragraphs": paragraphs,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": n_written, "total_rows": len(samples)}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  2WikiMultihopQA: wrote {n_written} queries -> {out_path}")
        return out_path
