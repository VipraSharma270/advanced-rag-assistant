"""Smoke tests for individual pipeline components."""

from utils.loader import load_documents
from utils.chunker import chunk_documents
from utils.retriever import retrieve


def test_loader():
    docs = load_documents()
    assert len(docs) > 0
    assert docs[0].page_content


def test_chunker():
    docs = load_documents()
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)


def test_retriever():
    results = retrieve("What is machine learning?")
    assert len(results) > 0
    assert results[0].page_content
