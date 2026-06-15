import os
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from config import DATA_DIR

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(data_folder: Path | str = DATA_DIR) -> list[Document]:
    """Load supported documents from a directory."""
    data_path = Path(data_folder)
    documents: list[Document] = []

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    for file in sorted(data_path.iterdir()):
        if not file.is_file():
            continue

        loader_cls = LOADERS.get(file.suffix.lower())
        if loader_cls is None:
            continue

        documents.extend(loader_cls(str(file)).load())

    if not documents:
        raise ValueError(f"No supported documents found in {data_path}")

    return documents
