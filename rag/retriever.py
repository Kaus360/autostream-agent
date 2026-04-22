"""
rag/retriever.py
Builds a FAISS vector store from knowledge-base chunks and exposes
a retriever that can be queried for semantically similar passages.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rag.loader import load_and_chunk

# Singleton — built once per process lifetime
_vectorstore = None

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build_vectorstore():
    """
    Embed all knowledge-base chunks and store them in FAISS.
    Uses HuggingFace all-MiniLM-L6-v2 (runs locally, no API key needed).
    """
    print("[RAG Retriever] Building FAISS vector store …")
    chunks = load_and_chunk()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[RAG Retriever] Vector store ready.")
    return vectorstore


def get_vectorstore():
    """Return (and lazily build) the singleton FAISS vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore


def retrieve(query: str, k: int = 3) -> str:
    """
    Retrieve the top-k most relevant chunks for *query*.

    Args:
        query: The user's question / search string.
        k:     Number of chunks to return (default 3).

    Returns:
        A single string with all retrieved chunks joined by separator lines,
        ready to be injected into an LLM prompt.
    """
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    if not docs:
        return "No relevant information found in the knowledge base."

    sections = []
    for i, doc in enumerate(docs, 1):
        sections.append(f"[Chunk {i}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(sections)
