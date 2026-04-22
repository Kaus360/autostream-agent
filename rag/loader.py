"""
rag/loader.py
Loads knowledge_base.md and splits it into overlapping chunks
for embedding and retrieval.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "knowledge_base.md"
)


def load_documents():
    """Load the knowledge base markdown file and return raw Document objects."""
    loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding="utf-8")
    documents = loader.load()
    return documents


def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split documents into smaller overlapping chunks suitable for embedding.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n- ", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def load_and_chunk():
    """Convenience wrapper: load + chunk in one call."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    return chunks
