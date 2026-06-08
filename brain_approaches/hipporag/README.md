## HippoRAG2 → Triples → Memgraph Pipeline

This folder contains a **minimal HippoRAG-style pipeline** that:

- reads **HippoRAG2-format datasets** from `data/HippoRAG2/dataset`,
- uses **OpenRouter** to extract RDF-style triples from text, and
- loads those triples into **Memgraph** so you can explore the knowledge graph.

The implementation is intentionally split into **small, single-responsibility modules** to match your scratchpad tasks.

---

### Module overview

- **`config.py`**
  - Central configuration for:
    - Dataset selection (`sample`, `musique`, `hotpotqa`, `2wikimultihopqa`, …).
    - Paths to corpus / query JSON files in `data/HippoRAG2/dataset`.
    - Output directory in `data/HippoRAG2/output`.
    - Default **OpenRouter model** for HippoRAG triple extraction.
    - **Free models list**: Loads free OpenRouter models from `data/HippoRAG2/free-models.yml` for automatic rotation.
  - Provides:
    - `HippoRagConfig(dataset: str)` → exposes `corpus_path`, `queries_path`.
    - `list_available_datasets()` → returns all known dataset keys.
    - `DEFAULT_OPENROUTER_MODEL` → default model string, aligned with free/dev models in `src/config.py` (e.g. `deepseek/deepseek-v3.1:free`).
    - `get_default_openrouter_model()` → resolves the model, preferring `OPENROUTER_MODEL` env var when set.
    - `load_free_models(active_only=True)` → loads only models with `status: "active"` from `free-models.yml` for rotation.

- **`load_dataset.py`**
  - Handles **only data loading** from HippoRAG2-style JSON:
    - `load_corpus(config)` → list of typed `CorpusEntry(idx, title, text)`.
    - `load_queries(config)` → raw query dicts or `None`.
    - `summarize_dataset(config)` → one-line summary for CLI.

- **`run_pipeline.py`**
  - Orchestrator / entry point for the pipeline:
    - Lists available datasets and asks **in the command line** which ones to process.
    - For each selected dataset, executes the full pipeline:
      - Loads the corpus JSON.
      - Extracts triples with OpenRouter (`extract_triples.py`).
      - Exports triples to CSV under `data/HippoRAG2/output` (`export_triples.py`).
      - Loads triples into Memgraph (`load_to_memgraph.py`).
    - Prints a short summary per dataset: number of triples, CSV path, and confirmation of Memgraph load.

- **`chunker.py`**
  - Adapts the original HippoRAG chunker logic (`src/agents/chunker_node.py`) for HippoRAG2 passages:
    - **Chunking strategy**:
      - Splits passages at sentence boundaries (regex-based sentence detection)
      - Target chunk size: **1000 tokens** (configurable via `config.chunk_size`, defaults from `src/config.py`)
      - Overlap ratio: **15%** (configurable via `config.chunk_overlap`, defaults from `src/config.py`)
      - Token counting: Uses `tiktoken` with `cl100k_base` encoding (GPT-4 tokenizer), falls back to character-based estimation (1 token ≈ 4 chars)
    - **Chunk validation**:
      - Minimum size: 200 tokens (smaller chunks are merged with the next chunk if possible)
      - Maximum size: 2000 tokens (warns but allows larger chunks)
    - **Output**: Creates `ChunkedPassage` dataclass objects with:
      - `chunk_id`: Unique identifier (e.g., `"doc_0_chunk_0"`)
      - `passage_idx`: Original passage index from corpus
      - `chunk_index`: Index within the passage (0, 1, 2, ...)
      - `content`: The chunked text content
      - `token_count`: Estimated token count for the chunk
      - `overlap_with_next`: Number of characters overlapping with the next chunk
    - **Functions**:
      - `chunk_passage(passage, target_size, overlap_ratio)`: Chunk a single `CorpusEntry`
      - `chunk_corpus(corpus, target_size, overlap_ratio)`: Chunk an entire corpus (reads defaults from `get_config()`)

- **`extract_triples.py`**
  - Uses `OpenRouterClient` and the HippoRAG triple-extraction prompt (see `scratchpad.md`) to:
    - **Chunk passages** using `chunker.py` before extraction
    - **Resume capability**: Automatically detects already-processed chunks from `<dataset>_progress.wal` and skips them
      - On resume: loads existing triples from CSV, only processes remaining chunks
      - Logs resume status: "Resume: skipping N already-processed chunks, M chunks left to run"
    - **Model rotation with credit error handling**:
      - Loads free models from `data/HippoRAG2/free-models.yml`
      - Automatically detects credit/affordability errors in API responses
      - Rotates to the next free model when credit errors occur
      - Prompts user to confirm rotation when all models are exhausted
      - Limits `max_tokens` to 4096 to reduce credit consumption
    - Call OpenRouter for each chunk in the corpus (only unprocessed chunks when resuming)
    - **Track progress** with WAL-style logging (`progress_tracker.py`)
    - **Batched API results**: Saves API results in batches of 100 calls per file (`batch_0000.json`, `batch_0001.json`, ...)
    - **Incrementally export** triples to CSV as they are extracted
    - Parse the JSON `{"triples": [[s, p, o], ...]}` into `Triple`-like structures
  - Model selection:
    - Reads `OPENROUTER_API_KEY` from the environment.
    - Loads free models from `free-models.yml` via `load_free_models()`.
    - Falls back to `DEFAULT_OPENROUTER_MODEL` if YAML not available.

- **`progress_tracker.py`**
  - WAL-style progress tracking for triple extraction:
    - Maintains `<dataset>_progress.wal` file with before/after entries for each API call
    - **Batched API results**: Saves API call results in batches of 100 per file (`batch_0000.json`, `batch_0001.json`, ...) instead of individual files
      - Each batch file contains a JSON array of 100 API call results
      - Automatically flushes batches when reaching 100 calls or at the end of extraction
    - Incrementally appends triples to final CSV output file
    - Provides progress summary statistics (pending, completed, failed, total_triples)
    - **Resume support**: `get_processed_chunk_ids()` returns set of chunk_ids that have been completed or failed, enabling automatic resume on next run

- **`export_triples.py`**
  - Writes extracted triples to CSV:
    - Location: `data/HippoRAG2/output/<dataset>_triples.csv`.
    - Columns: `subject,predicate,object,source_chunk_id`.

- **`load_to_memgraph.py`**
  - Uses the shared `MemgraphClient` to:
    - Bulk insert triples into Memgraph as nodes and relationships.
    - Report simple graph statistics (node and edge counts).

- **Reused core components from the base project**
  - `src/utils/openrouter_client.py`:
    - Async `OpenRouterClient` that wraps `chat/completions` calls with retries, logging, and flexible response parsing.
    - The HippoRAG triple-extraction prompt from `scratchpad.md` will be wired through this client.
  - `src/storage/memgraph_client.py`:
    - `MemgraphClient` built on the Neo4j driver (compatible with Memgraph Bolt).
    - Supports single-node operations, bulk triple inserts, stats, and clearing the graph.
  - `src/models/data_models.py`:
    - Defines the `Triple` dataclass used by `MemgraphClient.bulk_insert_triples`.

As we add `extract_triples.py`, `build_graph.py`, and `export_triples.py`, they will sit here and use the shared OpenRouter and Memgraph clients above.

---

### Architectural flow

High-level data flow from dataset to Memgraph:

1. **Dataset selection (CLI)**
   - `run_pipeline.py` → `_prompt_dataset_selection()`:
     - Lists dataset keys from `config.list_available_datasets()`.
     - Asks the user to choose indices or `all`.

2. **Dataset loading**
   - For each selected dataset:
     - `HippoRagConfig(dataset=name)` resolves the correct JSON paths.
     - `load_corpus(config)` reads `<dataset>_corpus.json`.
     - (Optional) `load_queries(config)` reads `<dataset>.json`.

3. **Chunking**
   - `chunker.py`:
     - Chunks passages into optimal sizes (default 1000 tokens) using sentence boundaries
     - Applies overlap (default 15%) between consecutive chunks
     - Creates `ChunkedPassage` objects with unique `chunk_id` identifiers

4. **Triple extraction** (with progress tracking)
   - `extract_triples.py`:
     - Builds prompts based on the **HippoRAG triple-extraction template** in `scratchpad.md`.
     - Calls `OpenRouterClient.generate(...)` from `src/utils/openrouter_client.py` for each chunk.
     - **Progress tracking** (`progress_tracker.py`):
       - Logs before request: `chunk_id`, timestamp, status="pending" to `<dataset>_progress.wal`
       - Logs after response: `chunk_id`, timestamp, status="completed|failed", triple_count to WAL
       - Saves API results in batches of **100 records per file** to `<dataset>_api_results/batch_0000.json`, `batch_0001.json`, … (no per-chunk files; legacy `doc_*_chunk_*.json` are removed on run)
     - **Incremental CSV export**: Appends triples to `<dataset>_triples.csv` as they are extracted
     - Parses the returned JSON `{"triples": [[s, p, o], ...]}` into `Triple`-like structures.

5. **Graph export** (incremental during extraction)
   - Triples are written incrementally to `data/HippoRAG2/output/<dataset>_triples.csv` during extraction
   - Final CSV has columns: `subject, predicate, object, source_chunk_id`

6. **Memgraph loading**
   - `load_to_memgraph.py`:
     - Uses `MemgraphClient` to bulk insert triples as nodes and edges.
     - Logs simple graph statistics (node and edge counts) after insertion.

---

### How to run (using the base project `.venv` and `uv`)

1. **Activate the base project virtual environment**

   From the project root (where `pyproject.toml` and `uv.lock` live), activate your existing `.venv`:

   - On **Windows PowerShell**:
     - `.\.venv\Scripts\Activate.ps1`
   - On **Windows cmd**:
     - `.\.venv\Scripts\activate.bat`
   - On **Linux/macOS**:
     - `source .venv/bin/activate`

2. **Use `uv run` to execute the pipeline entry point**

   Always run the pipeline through `uv` so it uses the project’s configured environment:

   - **Interactive dataset selection**:

     ```bash
     uv run python -m src.hipporag.run_pipeline
     ```

   - **Non-interactive for a specific dataset** (e.g. `musique` = dataset 2):

     ```bash
     uv run python -m src.hipporag.run_pipeline --datasets musique
     ```

   - **Non-interactive for all datasets**:

     ```bash
     uv run python -m src.hipporag.run_pipeline --datasets all
     ```

   - The script will:
     - For each dataset, chunk passages, extract triples (with progress tracking), export CSV incrementally, and load into Memgraph.
     - Print a one-line summary per dataset (triple count, CSV path, progress file, Memgraph confirmation).
     - **Output files** created in `data/HippoRAG2/output/`:
       - `<dataset>_triples.csv`: Final CSV with all extracted triples
       - `<dataset>_progress.wal`: WAL-style progress log (before/after each API call)
       - `<dataset>_api_results/`: Folder with batched API call results (`batch_0000.json`, `batch_0001.json`, ...) — 100 calls per file

3. **Resume capability**

   The pipeline automatically supports **resume** from interruptions:
   - If a previous run was interrupted (e.g., network error, Ctrl+C), simply run the pipeline again with the same dataset
   - The system will:
     - Read `<dataset>_progress.wal` to identify already-processed chunks
     - Skip chunks that have status `"completed"` or `"failed"` in the WAL
     - Only process remaining chunks (those with `"pending"` status or not in WAL)
     - Load existing triples from `<dataset>_triples.csv` and merge with new results
   - Example log output on resume:
     ```
     Resume: skipping 150 already-processed chunks, 50 chunks left to run
     Triple extraction complete for dataset 'musique': 2000 total triples in CSV (this run: 500, resumed from 150 processed chunks)
     ```
   - **Note**: Failed chunks are skipped on resume. To retry failed chunks, delete their entries from the WAL file or remove the entire progress file to start fresh.

4. **Active models and test_models**

   Only models marked **`status: "active"`** in `data/HippoRAG2/free-models.yml` are used for extraction. To refresh which models work:
   ```bash
   uv run python -m src.hipporag.test_models
   ```
   This calls the OpenRouter API for each free model, sets `status: "active"` or `status: "inactive"` (and optional `last_error`) in `free-models.yml`, and prints a summary. Re-run after OpenRouter changes or to recover from credit/rate issues.

5. **Model rotation and credit error handling**

   The pipeline automatically handles credit/affordability errors by rotating through **active** free models:
   - **Free models list**: `data/HippoRAG2/free-models.yml`; only entries with `status: "active"` are used (see **test_models** above)
   - **Automatic rotation**: When a credit error is detected, the pipeline switches to the next active model
   - **Credit error detection**: Detects errors containing keywords like "credit", "afford", "insufficient", "requires more credits"
   - **User prompt on exhaustion**: When all models are exhausted, the pipeline prompts you to:
     - Rotate and retry from the first model (credits may have reset)
     - Stop and wait/upgrade your account
   - **Example log output**:
     ```
     Credit error with model 'tngtech/deepseek-r1t2-chimera:free', rotating to model 2: 'tngtech/deepseek-r1t-chimera:free'
     ```
   - **Batched results**: API results are saved in batches of 100 calls per file (`batch_0000.json`, `batch_0001.json`, ...) to reduce file system overhead

6. **Memgraph side**

   - Ensure your **Docker Compose for Memgraph** is running and exposes Bolt on `bolt://localhost:7687` (the default used by `MemgraphClient`).

   #### Loading Triples into Memgraph

   To load triples from a JSON file into Memgraph for visualization:

   ```bash
   # Load triples from a JSON file (array of triple objects)
   uv run python -c "
   import json
   from src.storage.memgraph_client import MemgraphClient
   from src.models.data_models import Triple

   # Load triples from JSON
   with open('data/output/triples_20251228_234452.json', 'r') as f:
       raw_triples = json.load(f)

   triples = []
   for item in raw_triples:
       triple = Triple(
           subject=item['subject'],
           predicate=item['predicate'],
           object=item['object'],
           confidence=item.get('confidence', 1.0),
           source_chunk_id=item.get('source_chunk_id', ''),
           metadata=item.get('metadata', {})
       )
       triples.append(triple)

   # Load into Memgraph
   with MemgraphClient() as client:
       client.clear_graph()  # Clear existing graph
       client.bulk_insert_triples(triples)
       stats = client.get_stats()
       print(f'Loaded {stats[\"node_count\"]} nodes and {stats[\"edge_count\"]} relationships')
   "
   ```

   #### Visualization with Memgraph Lab

   - **Memgraph Lab Web Interface**: Open `http://localhost:3000` in your browser
   - **Interactive Graph Exploration**: Visual graph representations with Cypher queries
   - **Query Editor**: Write and execute Cypher queries with syntax highlighting

   #### Common Cypher Queries for Exploration

   **View the entire graph:**
   ```cypher
   MATCH (s)-[r]->(o) RETURN s, r, o LIMIT 50;
   ```

   **Find highly connected entities (most relationships):**
   ```cypher
   MATCH (n)-[r]-()
   RETURN n.name, count(r) AS connections
   ORDER BY connections DESC LIMIT 10;
   ```

   **Explore relationships for specific entities:**
   ```cypher
   MATCH (s)-[r]->(o)
   WHERE s.name CONTAINS "RAG"
   RETURN s, r, o;
   ```

   **Get graph statistics:**
   ```cypher
   MATCH (n) RETURN count(n) AS node_count;
   MATCH ()-[r]->() RETURN count(r) AS relationship_count;
   ```

   **Find entities by relationship type:**
   ```cypher
   MATCH (s)-[r]->(o)
   WHERE type(r) = "COMBINE"
   RETURN s, r, o LIMIT 20;
   ```

   **Explore confidence scores:**
   ```cypher
   MATCH (s)-[r]->(o)
   RETURN s.name, type(r), o.name, r.confidence
   ORDER BY r.confidence DESC LIMIT 25;
   ```

   These correspond to the stats logged by `MemgraphClient.get_stats()` after loading triples.

---

### Keeping README and scratchpad in sync

Whenever a new step is implemented (e.g. triple extraction, export, Memgraph load):

- **Scratchpad**:
  - Update the checklist in `scratchpad.md` (mark tasks as done, add notes).
- **This README**:
  - Add/upate a short note under the relevant section (e.g. “Triple extraction is now implemented in `extract_triples.py`”).

This keeps both the **architectural view** (README) and the **fine-grained task/progress view** (scratchpad) aligned.

