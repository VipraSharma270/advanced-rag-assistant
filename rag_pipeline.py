from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

from config import LLM_MODEL, TOP_K_CONTEXT
from utils.reranker import rerank
from utils.retriever import retrieve

_llm = OllamaLLM(model=LLM_MODEL)

_SYSTEM_PROMPT = """You are a precise research assistant.

Answer the question using ONLY the context below.
If the context does not contain enough information, respond exactly with:
"I could not find the answer in the provided documents."

Be concise, factual, and cite relevant details from the context."""


def _build_prompt(query: str, context: str) -> str:
    return f"""{_SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:"""


def ask(query: str) -> dict[str, str | list[Document]]:
    """Run the full RAG pipeline: retrieve → rerank → generate."""
    docs = retrieve(query)
    ranked_docs = rerank(query, docs)
    top_docs = ranked_docs[:TOP_K_CONTEXT]

    context = "\n\n".join(doc.page_content for doc in top_docs)
    answer = _llm.invoke(_build_prompt(query, context))

    return {"answer": answer, "sources": top_docs}
