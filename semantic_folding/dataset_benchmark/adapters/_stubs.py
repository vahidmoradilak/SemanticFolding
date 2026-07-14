"""Stub adapters for datasets not yet implemented. See final-datasets.md."""

from .base_adapter import BaseDatasetAdapter
from pathlib import Path


class _StubAdapter(BaseDatasetAdapter):
    dataset_name = ""
    display_name = ""
    default_subset = "dev"

    def download(self, output_dir: Path) -> Path:
        raise NotImplementedError(
            f"{self.display_name} adapter is not yet implemented."
        )

    def convert_to_musique_format(self, raw_path: Path, output_dir: Path, max_queries: int = 500) -> Path:
        raise NotImplementedError(
            f"{self.display_name} adapter is not yet implemented."
        )


class SciDQAAdapter(_StubAdapter):
    dataset_name = "scidqa"
    display_name = "SciDQA"


class DropAdapter(_StubAdapter):
    dataset_name = "drop"
    display_name = "DROP"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._converted_path = None

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        f = output_dir / "drop.jsonl"
        if not f.exists():
            raise FileNotFoundError(
                f"DROP raw data not found at {f}. "
                f"Download from huggingface.co/datasets/ucinlp/drop and place there."
            )
        return output_dir

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 200
    ) -> Path:
        import random, json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "drop.jsonl"

        src = raw_path / "drop.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"DROP JSONL not found: {src}")

        sections = {}
        with open(src, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                sid = d["section_id"]
                if sid not in sections:
                    sections[sid] = {"passage": d["passage"], "queries": []}
                sections[sid]["queries"].append(d)

        random.seed(42)
        n_written = 0
        entries = []
        for sid, info in sections.items():
            for qd in info["queries"]:
                if n_written >= max_queries:
                    break
                gold_passage = info["passage"]
                distractor_ids = [x for x in sections if x != sid]
                random.shuffle(distractor_ids)
                distractors = distractor_ids[:19]

                paragraphs = [{
                    "idx": 0,
                    "title": f"section_{sid}",
                    "paragraph_text": gold_passage,
                    "is_supporting": True,
                }]
                for i, dsid in enumerate(distractors):
                    paragraphs.append({
                        "idx": i + 1,
                        "title": f"section_{dsid}",
                        "paragraph_text": sections[dsid]["passage"],
                        "is_supporting": False,
                    })
                entries.append({
                    "id": qd["query_id"],
                    "question": qd["question"],
                    "answer": "; ".join(qd["answers_spans"]["spans"]),
                    "paragraphs": paragraphs,
                })
                n_written += 1
            if n_written >= max_queries:
                break

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": len(entries)}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"DROP: wrote {len(entries)} queries -> {out_path}")
        self._converted_path = out_path
        return out_path


class MultiMedQAAdapter(_StubAdapter):
    dataset_name = "multimedqa"
    display_name = "MultiMedQA"


class DocFinQAAdapter(_StubAdapter):
    dataset_name = "docfinqa"
    display_name = "DocFinQA"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._converted_path = None

    def get_recommended_params(self):
        p = super().get_recommended_params()
        p["grid_size"] = 128
        return p

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        f = output_dir / "dev.json"
        if not f.exists():
            raise FileNotFoundError(
                f"DocFinQA raw data not found at {f}. "
                f"Download dev.json from huggingface.co/datasets/kensho/DocFinQA"
            )
        return output_dir

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 100
    ) -> Path:
        import json, re
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "docfinqa.jsonl"

        src = raw_path / "dev.json"
        if not src.exists():
            raise FileNotFoundError(f"DocFinQA dev.json not found: {src}")

        with open(src, encoding="utf-8") as f:
            data = json.load(f)

        def chunk_text(text, size=5000):
            return [text[i:i+size] for i in range(0, len(text), size)]

        def find_answer_chunks(chunks, ans):
            result = []
            for i, chunk in enumerate(chunks):
                clean = chunk.replace('\t', ' ').replace('\n', ' ')
                if ans in clean:
                    result.append(i)
            return result

        import random
        random.seed(42)
        entries = []
        n_written = 0
        skipped = 0

        for e in data:
            if n_written >= max_queries:
                break
            ctx = e["Context"]
            ans = e["Answer"].strip()
            if not ans or ans not in ctx.replace('\t', ' '):
                skipped += 1
                continue

            chunks = chunk_text(ctx)
            answer_chunk_idxs = find_answer_chunks(chunks, ans)
            if not answer_chunk_idxs:
                skipped += 1
                continue

            paragraphs = []
            for i, chunk in enumerate(chunks):
                paragraphs.append({
                    "idx": i,
                    "title": f"chunk_{i:04d}",
                    "paragraph_text": chunk,
                    "is_supporting": i in answer_chunk_idxs,
                })
            entries.append({
                "id": f"docfinqa_{n_written:04d}",
                "question": e["Question"],
                "answer": ans,
                "paragraphs": paragraphs,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {"num_queries": len(entries), "skipped_no_answer": skipped}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"DocFinQA: wrote {len(entries)} queries -> {out_path} (skipped {skipped})")
        self._converted_path = out_path
        return out_path


class MedReadMeAdapter(_StubAdapter):
    dataset_name = "medreadme"
    display_name = "MedReadMe"

    def get_recommended_params(self):
        p = super().get_recommended_params()
        p["grid_size"] = 32
        return p


class CflueAdapter(_StubAdapter):
    dataset_name = "cflue"
    display_name = "CFLUE"
