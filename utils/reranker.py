from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

_model = CrossEncoder(RERANKER_MODEL)


def rerank(query: str, docs: list[Document]) -> list[Document]:
    """Re-score retrieved chunks with a cross-encoder for higher precision."""
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = _model.predict(pairs)

    ranked = sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])
    return [doc for _, doc in ranked]
