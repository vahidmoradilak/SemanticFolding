I'll search for information about the datasets and corpuses used in HippoRAG2.Let me search for more specific information about the HippoRAG2 paper and the datasets used.Based on my search, I can provide you with information about the datasets (corpuses) used in HippoRAG2. Here are the key datasets evaluated in the HippoRAG2 paper:

## Main Datasets Used in HippoRAG2:

### 1. **Factual Memory Tasks:**
- **NaturalQuestions (NQ)** - Contains real user questions with a wide range of topics
- **PopQA** - Derived from the December 2021 Wikipedia dump, particularly entity-centric

### 2. **Multi-hop QA (Associativity) Tasks:**
- **MuSiQue** - Multi-hop reasoning dataset
- **2WikiMultiHopQA** - Multi-passage reasoning
- **HotpotQA** - Multi-hop question answering
- **LV-Eval (hotpotwikiqa-mixup 256k)** - A challenging dataset designed to minimize knowledge leakage through keyword and phrase replacements

### 3. **Discourse Understanding (Sense-Making) Tasks:**
- **NarrativeQA** - Questions requiring cohesive understanding of full-length novels

## Dataset Access:

**HuggingFace Repository:** The complete set of datasets is available on their HuggingFace dataset at: **`osunlp/HippoRAG_2`**

**GitHub Repository:** Some sample datasets are included in the `reproduce/dataset` directory of the HippoRAG2 GitHub repository at: **`github.com/ianliuwd/HippoRAG2`**

The datasets follow a naming convention where corpus files end with `_corpus.json` (e.g., `sample_corpus.json`, `narrativeqa_dev_10_doc_corpus.json`).