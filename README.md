# Advanced RAG Assistant

A production-style **Retrieval-Augmented Generation** pipeline that combines dense vector search, cross-encoder reranking, and local LLM inference — all runnable on your machine.

```text
  Documents (PDF / TXT / DOCX)
           │
           ▼
    ┌──────────────┐
    │   Ingest     │  Recursive chunking (500 tokens, 100 overlap)
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  ChromaDB    │  BGE-small-en-v1.5 embeddings
    └──────┬───────┘
           │
    Query  ▼
    ┌──────────────┐
    │  Retrieve    │  Top-5 semantic search
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   Rerank     │  MS MARCO cross-encoder
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  Generate    │  Llama 3 via Ollama
    └──────────────┘
```

## Features

| Stage | Model / Tool | Purpose |
|-------|-------------|---------|
| **Embedding** | `BAAI/bge-small-en-v1.5` | Fast, high-quality dense retrieval |
| **Vector store** | ChromaDB | Persistent local index |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision scoring over bi-encoder candidates |
| **Generation** | `llama3` (Ollama) | Grounded answers from retrieved context |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with `llama3` pulled (`ollama pull llama3`)

### Setup

```bash
git clone https://github.com/VipraSharma270/advanced-rag-assistant.git
cd advanced-rag-assistant

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### Index documents

Place your files in `data/` (PDF, TXT, or DOCX), then run:

```bash
python ingest.py
```

### Launch the UI

```bash
streamlit run app.py
```

## Project Structure

```text
├── app.py              # Streamlit chat interface
├── config.py           # Centralized pipeline settings
├── ingest.py           # Document ingestion & indexing
├── rag_pipeline.py     # Retrieve → rerank → generate
├── utils/
│   ├── loader.py       # Multi-format document loading
│   ├── chunker.py      # Text splitting
│   ├── retriever.py    # ChromaDB semantic search
│   └── reranker.py     # Cross-encoder reranking
├── data/               # Source documents
└── vectordb/           # Generated vector index (gitignored)
```

## Configuration

All tunables live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `LLM_MODEL` | `llama3` | Ollama model name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Candidates fetched before reranking |
| `TOP_K_CONTEXT` | `3` | Chunks passed to the LLM |

## How It Works

1. **Ingest** — Documents are loaded, split into overlapping chunks, and embedded into a persistent ChromaDB index.
2. **Retrieve** — A user query is embedded and matched against the top-*k* most similar chunks (bi-encoder search).
3. **Rerank** — A cross-encoder scores each *(query, chunk)* pair jointly, reordering results for higher precision.
4. **Generate** — The top-ranked chunks form the LLM context; the model answers strictly from that context.

## License

MIT
