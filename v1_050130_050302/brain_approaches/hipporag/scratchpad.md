# HippoRAG Scratchpad

This scratchpad collects the exact HippoRAG triple-extraction prompt, plus notes on adapting it to OpenRouter and running the end-to-end pipeline from datasets → triples → Memgraph.

---

## Triple Extraction Prompt (HippoRAG Original)

This is the original triple-extraction prompt template from the HippoRAG repository (`triple_extraction.py`). It conditions relation extraction on named entities and returns an RDF-style graph as JSON triples.

```python
from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{"triples": [
 ["Radio City", "located in", "India"],
 ["Radio City", "is", "private FM radio station"],
 ["Radio City", "started on", "3 July 2001"],
 ["Radio City", "plays songs in", "Hindi"],
 ["Radio City", "plays songs in", "English"],
 ["Radio City", "forayed into", "New Media"],
 ["Radio City", "launched", "PlanetRadiocity.com"],
 ["PlanetRadiocity.com", "launched in", "May 2008"],
 ["PlanetRadiocity.com", "is", "music portal"],
 ["PlanetRadiocity.com", "offers", "news"],
 ["PlanetRadiocity.com", "offers", "videos"],
 ["PlanetRadiocity.com", "offers", "songs"]
 ]
}
"""


prompt_template = [
 {"role": "system", "content": ner_conditioned_re_system},
 {"role": "user", "content": ner_conditioned_re_input},
 {"role": "assistant", "content": ner_conditioned_re_output},
 {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]
```

---

## Notes for OpenRouter Adaptation

- **Messages format**: Use `prompt_template` directly as the `messages` payload for OpenRouter's `chat/completions` endpoint; only the final `user` message should have `{passage}` and `{named_entity_json}` dynamically filled in.
- **API key**: Read `OPENROUTER_API_KEY` from the environment; do not hardcode secrets.
- **Endpoint**: `https://openrouter.ai/api/v1/chat/completions`.
- **Headers** (minimum):
  - `Authorization: Bearer <OPENROUTER_API_KEY>`
  - Optionally, `HTTP-Referer` and `X-Title` to identify your app.
- **Response shape**: Expect JSON in the form:
  - `{"triples": [[subject, relation, object], ...]}`
  - Optionally include `source_id` or passage index on your side when storing triples.

Example OpenRouter request body (conceptual):

```json
{
  "model": "openai/gpt-4.1-mini",
  "messages": [
    {"role": "system", "content": "...ner_conditioned_re_system..."},
    {"role": "user", "content": "...example input with passage and named entities..."},
    {"role": "assistant", "content": "...example triples JSON..."},
    {
      "role": "user",
      "content": "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\nParagraph:\n```\n{passage}\n```\n\n{named_entity_json}\n"
    }
  ]
}
```

When calling from code, substitute `{passage}` and `{named_entity_json}` in the last message before sending.

---

## Per-Dataset Workflow (Data → Triples → Memgraph → Query)

For each dataset under `data/HippoRAG2/dataset`:

1. **Load dataset corpus**
   - Read `<dataset>_corpus.json` (e.g. `sample_corpus.json`, `musique_corpus.json`).
   - Each entry should at least contain `idx`, `title`, `text`.
2. **(Optional) Load dataset queries**
   - Read `<dataset>.json` (e.g. `sample.json`, `musique.json`) if you want question-answer pairs.
3. **Chunk passages using HippoRAG chunker**
   - Uses the same chunking logic as `src/agents/chunker_node.py`:
     - Target chunk size: 1000 tokens (configurable via `config.chunk_size`)
     - Overlap ratio: 15% (configurable via `config.chunk_overlap`)
     - Splits at sentence boundaries
     - Validates chunk sizes (min 200, max 2000 tokens)
   - Creates `ChunkedPassage` objects with unique `chunk_id` (e.g., `doc_0_chunk_0`)
4. **Extract triples with OpenRouter** (with progress tracking)
   - For each chunk, call the triple-extraction prompt via OpenRouter.
   - **WAL-style progress tracking** (`<dataset>_progress.wal`):
     - Before API call: logs `chunk_id`, timestamp, status="pending"
     - After API call: logs `chunk_id`, timestamp, status="completed|failed", triple_count, error (if any)
   - **Individual API results** saved to `<dataset>_api_results/<chunk_id>.json`:
     - Contains triples, raw response (truncated), status, error (if any)
   - **Incremental CSV export**: Triples appended to `<dataset>_triples.csv` as they are extracted
   - Parse the `"triples"` JSON list into in-memory structures with `subject`, `relation`, `object`, and `source_chunk_id`.
5. **Normalize and deduplicate**
   - Normalize text (strip whitespace, consistent casing).
   - Remove exact duplicate `(subject, relation, object)` triples.
6. **Export triples** (already done incrementally during extraction)
   - Final CSV at `data/HippoRAG2/output/<dataset>_triples.csv` with headers:
     - `subject,predicate,object,source_chunk_id`
7. **Load into Memgraph**
   - Connect via Bolt (e.g. `bolt://localhost:7687`) using default or configured credentials.
   - For each triple, `MERGE` nodes and relationships:
     - `(:Entity {name: subject})-[:RELATION {type: relation, source_id: source_id}]->(:Entity {name: object})`
8. **Query in Memgraph**
   - Example simple inspection queries:
     - `MATCH (s:Entity)-[r:RELATION]->(o:Entity) RETURN s,r,o LIMIT 25;`
     - `MATCH (e:Entity {name: $name})-[r]->(o) RETURN e,r,o LIMIT 25;`

---

## Dataset Checklist Template & Status

Use this checklist to track progress for each dataset under `data/HippoRAG2/dataset` (sample, musique, hotpotqa, 2wikimultihopqa, etc.).

### Generic Checklist (copy per dataset)

- [x] Load corpus JSON (**code implemented in** `load_dataset.py`)
- [x] (Optional) Load query JSON (**code implemented in** `load_dataset.py`)
- [x] Chunk passages using HippoRAG chunker (**code implemented in** `chunker.py`)
- [x] Extract triples with OpenRouter (**code implemented in** `extract_triples.py`)
  - Uses chunking before extraction
  - WAL-style progress tracking per dataset (`<dataset>_progress.wal`)
  - Individual API call results saved to `<dataset>_api_results/` folder
  - Incremental CSV export during extraction
- [ ] Normalize & deduplicate triples (optional extra step)
- [x] Export triples CSV to `data/HippoRAG2/output` (**incremental export during extraction**, final file: `<dataset>_triples.csv`)
- [x] Load triples into Memgraph (**code implemented in** `load_to_memgraph.py` and wired in `run_pipeline.py`)
- [ ] Run basic Memgraph queries to verify graph

### Example: `sample`

- [ ] `sample_corpus.json` loaded (via `HippoRagConfig(dataset="sample")`)
- [ ] `sample.json` (queries) loaded
- [ ] Triples extracted and parsed
- [ ] `sample_triples.csv` written
- [ ] Triples loaded into Memgraph
- [ ] Queries run in Memgraph Lab

### Example: `musique`

- [ ] `musique_corpus.json` loaded
- [ ] `musique.json` loaded
- [ ] Triples extracted and parsed
- [ ] `musique_triples.csv` written
- [ ] Triples loaded into Memgraph
- [ ] Queries run in Memgraph Lab

### Example: `hotpotqa`

- [ ] `hotpotqa_corpus.json` loaded
- [ ] `hotpotqa.json` loaded
- [ ] Triples extracted and parsed
- [ ] `hotpotqa_triples.csv` written
- [ ] Triples loaded into Memgraph
- [ ] Queries run in Memgraph Lab

### Example: `2wikimultihopqa`

- [ ] `2wikimultihopqa_corpus.json` loaded
- [ ] `2wikimultihopqa.json` loaded
- [ ] Triples extracted and parsed
- [ ] `2wikimultihopqa_triples.csv` written
- [ ] Triples loaded into Memgraph
- [ ] Queries run in Memgraph Lab

---

## Quick Notes

- You can refine the triple-extraction prompt here (e.g. require only factual relations, filter out weak edges) and then mirror changes into the code.
- When testing, start with the `sample` dataset for fast runs before scaling up to `musique`, `hotpotqa`, and `2wikimultihopqa`.
- To run the current pipeline entry point, make sure the **base project virtual environment** is active and then run, for example:
  - Interactive dataset selection:
    - `uv run python -m src.hipporag.run_pipeline`
  - Non-interactive for a specific dataset (e.g. `musique` = dataset 2):
    - `uv run python -m src.hipporag.run_pipeline --datasets musique`
  - Non-interactive for all datasets:
    - `uv run python -m src.hipporag.run_pipeline --datasets all`
- Architectural overview and run instructions for this submodule are documented in `src/hipporag/README.md`. Whenever you complete a new task here (e.g. triple extraction, Memgraph loading), update both this scratchpad and that README.

---

## Loguru Logging Pattern for New Modules

All new `src/hipporag` modules should integrate with the existing **loguru** setup used by the base project.

### Import and logger setup

Use this pattern at the top of each module:

```python
from loguru import logger
```

Do **not** reconfigure loguru in these modules — configuration is already handled centrally on startup (see the base `src` entrypoint). Simply use `logger` for structured logs.

### Example usage in a HippoRAG module

```python
from loguru import logger


def some_step(...):
    logger.info("Starting some_step for dataset='{}'", dataset_name)

    try:
        # core logic
        logger.debug("Loaded {} passages from {}", len(corpus), corpus_path)
    except Exception as exc:
        logger.exception("Error while running some_step: {}", exc)
        raise
```

Recommended conventions:

- Use `logger.info` for high-level progress (dataset selection, step start/finish, counts).
- Use `logger.debug` for detailed diagnostics (file paths, exact counts, intermediate sizes).
- Use `logger.warning` when proceeding with degraded behavior (e.g. skipping a passage).
- Use `logger.error` / `logger.exception` when aborting or surfacing an unexpected failure.


