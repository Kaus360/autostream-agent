"""rag/__init__.py — exposes top-level helpers for the RAG pipeline."""

from rag.retriever import retrieve, get_vectorstore

__all__ = ["retrieve", "get_vectorstore"]
