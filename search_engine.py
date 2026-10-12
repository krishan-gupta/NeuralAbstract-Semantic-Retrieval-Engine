"""Dense, sparse, and reciprocal-rank-fused paper retrieval."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from threading import RLock
from typing import List
from urllib.parse import quote

import bm25s
import faiss
import numpy as np

from embeddings import BaseEmbeddingProvider, LocalSentenceTransformerProvider

LOGGER = logging.getLogger(__name__)


class HybridSearchEngine:
    """Search the persisted FAISS and BM25 indexes and load paper metadata."""

    def __init__(
        self,
        database_path: str | Path = "papers.db",
        dense_index_path: str | Path = "faiss_local.index",
        sparse_index_path: str | Path = "bm25_index",
        embedder: BaseEmbeddingProvider | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        dense_path = Path(dense_index_path)
        sparse_path = Path(sparse_index_path)
        for path in (self.database_path, dense_path, sparse_path):
            if not path.exists():
                raise FileNotFoundError(f"Required search data not found: {path}")

        self.embedder = embedder or LocalSentenceTransformerProvider()
        self.dense_index = faiss.read_index(str(dense_path))
        if self.dense_index.d != self.embedder.get_dimension():
            raise ValueError(
                f"FAISS dimension {self.dense_index.d} does not match embedder dimension "
                f"{self.embedder.get_dimension()}"
            )
        self.sparse_index = bm25s.BM25.load(str(sparse_path), load_corpus=False)

        # SQLite URI mode=ro guarantees this connection cannot modify the database.
        database_uri = f"file:{quote(str(self.database_path.resolve().as_posix()), safe='/:')}?mode=ro"
        self._connection = sqlite3.connect(
            database_uri, uri=True, check_same_thread=False, timeout=10.0
        )
        self._connection.row_factory = sqlite3.Row
        self._connection_lock = RLock()
        with self._connection_lock:
            self._sqlite_ids = [
                int(row[0])
                for row in self._connection.execute("SELECT id FROM papers ORDER BY id ASC")
            ]
        if self.dense_index.ntotal != len(self._sqlite_ids):
            self.close()
            raise ValueError(
                "FAISS index and database have different document counts; rebuild the indexes"
            )
        if self.sparse_index.scores["num_docs"] != len(self._sqlite_ids):
            self.close()
            raise ValueError(
                "BM25 index and database have different document counts; rebuild the indexes"
            )

    def close(self) -> None:
        """Close the read-only SQLite connection."""
        connection = getattr(self, "_connection", None)
        if connection is not None:
            with self._connection_lock:
                connection.close()
            self._connection = None

    @staticmethod
    def _validate_query(query: str, top_k: int) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

    def search_dense(self, query: str, top_k: int) -> List[int]:
        """Return nearest SQLite IDs using cosine similarity via inner product."""
        self._validate_query(query, top_k)
        if self.dense_index.ntotal == 0:
            return []
        vector = np.asarray(self.embedder.embed_texts([query]), dtype=np.float32)
        expected_shape = (1, self.embedder.get_dimension())
        if vector.shape != expected_shape or not np.isfinite(vector).all():
            raise ValueError(f"Embedder returned invalid query vector with shape {vector.shape}")
        _, found_ids = self.dense_index.search(
            np.ascontiguousarray(vector), min(top_k, self.dense_index.ntotal)
        )
        return [int(doc_id) for doc_id in found_ids[0] if doc_id >= 0]

    def search_sparse(self, query: str, top_k: int) -> List[int]:
        """Return SQLite IDs from the positional BM25 index."""
        self._validate_query(query, top_k)
        document_count = len(self._sqlite_ids)
        if document_count == 0:
            return []
        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        positions, _ = self.sparse_index.retrieve(
            query_tokens,
            k=min(top_k, document_count),
            sorted=True,
            return_as="tuple",
            show_progress=False,
        )
        return [self._sqlite_ids[int(position)] for position in positions[0] if int(position) >= 0]

    def search_hybrid(self, query: str, top_k: int) -> List[int]:
        """Fuse the top 50 dense and sparse results using standard RRF."""
        self._validate_query(query, top_k)
        dense_ids = self.search_dense(query, 50)
        sparse_ids = self.search_sparse(query, 50)
        scores: dict[int, float] = {}
        for ranked_ids in (dense_ids, sparse_ids):
            for rank, doc_id in enumerate(ranked_ids, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]

    def get_metadata(self, doc_ids: List[int]) -> List[dict]:
        """Fetch paper metadata in exactly the order requested."""
        if not doc_ids:
            return []
        placeholders = ",".join("?" for _ in set(doc_ids))
        sql = (
            "SELECT id, title, abstract, authors, categories, published, url "
            f"FROM papers WHERE id IN ({placeholders})"
        )
        with self._connection_lock:
            rows = self._connection.execute(sql, tuple(dict.fromkeys(doc_ids))).fetchall()
        by_id = {int(row["id"]): dict(row) for row in rows}
        return [
            {key: value for key, value in by_id[doc_id].items() if key != "id"}
            for doc_id in doc_ids
            if doc_id in by_id
        ]
