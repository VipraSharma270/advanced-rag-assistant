from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, VECTOR_DB_DIR
from utils.chunker import chunk_documents
from utils.loader import load_documents


def main() -> None:
    print("Loading documents...")
    docs = load_documents()

    print(f"Chunking {len(docs)} document(s)...")
    chunks = chunk_documents(docs)

    print(f"Indexing {len(chunks)} chunks into ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DB_DIR),
    )
    vectorstore.persist()

    print(f"Done — vector store saved to {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
