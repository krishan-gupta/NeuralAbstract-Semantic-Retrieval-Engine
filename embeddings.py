"""Swappable text embedding providers with a common normalized output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class BaseEmbeddingProvider(ABC):
    """Interface implemented by all embedding backends."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts into an L2-normalized float32 matrix."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the output embedding dimension."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query, allowing models to apply query-specific instructions."""
        return self.embed_texts([query])


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors


class LocalSentenceTransformerProvider(BaseEmbeddingProvider):
    """Embedding provider backed by a local Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        query_prefix: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to use the local provider"
            ) from exc
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        self.batch_size = batch_size
        get_dimension = getattr(self.model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self.model.get_sentence_embedding_dimension
        self._dimension = int(get_dimension())
        self.query_prefix = query_prefix
        if self.query_prefix is None and "bge-small-en-v1.5" in model_name.lower():
            self.query_prefix = (
                "Represent this sentence for searching relevant passages: "
            )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return _normalize(vectors)

    def get_dimension(self) -> int:
        return self._dimension

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query, applying BGE's retrieval instruction only to queries."""
        return self.embed_texts([(self.query_prefix or "") + query])


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by the OpenAI embeddings API."""

    _KNOWN_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 128,
        api_key: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use the OpenAI provider") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        if model not in self._KNOWN_DIMENSIONS:
            raise ValueError(
                "Unknown embedding dimension for model; add it to _KNOWN_DIMENSIONS"
            )
        self._dimension = self._KNOWN_DIMENSIONS[model]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise RuntimeError("OpenAI returned an unexpected number of embeddings")
            rows.extend(item.embedding for item in ordered)
        vectors = _normalize(np.asarray(rows, dtype=np.float32))
        if vectors.shape[1] != self._dimension:
            raise RuntimeError(
                f"Expected dimension {self._dimension}, got {vectors.shape[1]}"
            )
        return vectors

    def get_dimension(self) -> int:
        return self._dimension
