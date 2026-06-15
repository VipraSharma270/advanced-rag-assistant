from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vectordb"

# Embedding & retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
RETRIEVAL_K = 5
TOP_K_CONTEXT = 3
