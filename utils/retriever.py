from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, RETRIEVAL_K, VECTOR_DB_DIR

_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

_vectorstore = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=_embedding_model,
)

_retriever = _vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def retrieve(query: str) -> list[Document]:
    """Return the top-k semantically similar document chunks."""
    return _retriever.invoke(query)
