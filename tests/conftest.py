"""Shared test setup. Tests never call the OpenAI API: query embeddings are replaced with vectors
already stored in Chroma, so the suite runs offline, for free, and gives the same results every time."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

import recommender  # noqa: E402


@pytest.fixture(scope="session")
def stored_vectors():
    """isbn13 -> the book's stored OpenAI embedding."""
    stored = recommender.db_books._collection.get(include=["embeddings", "documents"])
    return {int(doc.strip().strip('"').split()[0]): np.array(vec)
            for doc, vec in zip(stored["documents"], stored["embeddings"])}


@pytest.fixture
def query_like(monkeypatch, stored_vectors):
    """Make the 'query' embed to the same vector as a given book, so that book should rank first."""
    def use_book(isbn13):
        # Patched on the class: LangChain embeddings are Pydantic models, which reject instance attributes
        monkeypatch.setattr(type(recommender.embeddings), "embed_query",
                            lambda self, text: stored_vectors[isbn13].tolist())
        return isbn13
    return use_book
