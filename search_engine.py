"""Dense, sparse, hybrid, and metadata retrieval for the paper collection."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from threading import RLock
from urllib.parse import quote

import bm25s
import faiss
import numpy as np

from embeddings import BaseEmbeddingProvider, LocalSentenceTransformerProvider

LOGGER = logging.getLogger(__name__)


class HybridSearchEngine:
    """Search persisted indexes and resolve their results to paper metadata."""

    def __init__(
        self,
        database_path: str | Path = "papers.db",
        dense_index_path: str | Path = "faiss_local.index",
        sparse_index_path: str | Path = "bm25_index",
        embedder: BaseEmbeddingProvider | None = None,
        dense_indexes: dict[str, str | Path] | None = None,
        embedders: dict[str, BaseEmbeddingProvider] | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.sparse_index_path = Path(sparse_index_path)
        selected_paths = dict(dense_indexes or {"local": dense_index_path})
        selected_paths.setdefault("local", dense_index_path)
        for path in (
            self.database_path,
            self.sparse_index_path,
            *map(Path, selected_paths.values()),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Required search data not found: {path}")

        self.embedders: dict[str, BaseEmbeddingProvider] = dict(embedders or {})
        if embedder is not None:
            self.embedders.setdefault("local", embedder)
        if "local" not in self.embedders:
            self.embedders["local"] = LocalSentenceTransformerProvider(
                local_files_only=True
            )

        self.dense_indexes: dict[str, faiss.Index] = {
            tag: faiss.read_index(str(path)) for tag, path in selected_paths.items()
        }
        for tag, index in self.dense_indexes.items():
            if tag not in self.embedders:
                model = "BAAI/bge-small-en-v1.5" if tag == "bge" else tag
                self.embedders[tag] = LocalSentenceTransformerProvider(
                    model, local_files_only=True
                )
            if index.d != self.embedders[tag].get_dimension():
                raise ValueError(
                    f"FAISS index '{tag}' dimension {index.d} does not match its embedder "
                    f"dimension {self.embedders[tag].get_dimension()}"
                )
        self.dense_index = self.dense_indexes["local"]  # Backward-compatible attribute.
        self.embedder = self.embedders["local"]

        self._bm25_doc_ids_path = self.sparse_index_path / "doc_ids.npy"
        if not self._bm25_doc_ids_path.is_file():
            raise FileNotFoundError(
                f"BM25 SQLite ID map not found: {self._bm25_doc_ids_path}; rebuild the sparse index"
            )
        self.sparse_index = bm25s.BM25.load(
            str(self.sparse_index_path), load_corpus=False
        )
        self.sparse_doc_ids = np.load(
            self._bm25_doc_ids_path, allow_pickle=False
        ).astype(np.int64)
        if len(self.sparse_doc_ids) != int(self.sparse_index.scores["num_docs"]):
            raise ValueError(
                "BM25 index and doc_ids.npy have different document counts"
            )

        database_uri = f"file:{quote(str(self.database_path.resolve().as_posix()), safe='/:')}?mode=ro"
        self._connection = sqlite3.connect(
            database_uri, uri=True, check_same_thread=False, timeout=10.0
        )
        self._connection.row_factory = sqlite3.Row
        self._connection_lock = RLock()
        try:
            with self._connection_lock:
                self._sqlite_ids = [
                    int(row[0])
                    for row in self._connection.execute(
                        "SELECT id FROM papers ORDER BY id ASC"
                    )
                ]
            database_id_set = set(self._sqlite_ids)
            if len(self.sparse_doc_ids) != len(self._sqlite_ids):
                raise ValueError(
                    "BM25 index and database have different document counts; rebuild indexes"
                )
            if len(set(map(int, self.sparse_doc_ids))) != len(self.sparse_doc_ids):
                raise ValueError("BM25 doc_ids.npy contains duplicate SQLite IDs")
            if not set(map(int, self.sparse_doc_ids)).issubset(database_id_set):
                raise ValueError(
                    "BM25 doc_ids.npy contains IDs absent from the papers database"
                )
            for tag, index in self.dense_indexes.items():
                if index.ntotal != len(self._sqlite_ids):
                    raise ValueError(
                        f"FAISS index '{tag}' and database have different document counts; rebuild indexes"
                    )
                if not hasattr(index, "id_map"):
                    raise ValueError(
                        f"FAISS index '{tag}' has no explicit SQLite ID map"
                    )
                faiss_ids = set(map(int, faiss.vector_to_array(index.id_map)))
                if faiss_ids != database_id_set:
                    raise ValueError(
                        f"FAISS index '{tag}' IDs do not match papers.db; rebuild indexes"
                    )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close the read-only SQLite connection."""
        connection = getattr(self, "_connection", None)
        if connection is not None:
            lock = getattr(self, "_connection_lock", RLock())
            with lock:
                connection.close()
            self._connection = None

    @staticmethod
    def _validate_query(query: str, top_k: int) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

    def search_dense(
        self, query: str, top_k: int, embedder: str = "local"
    ) -> list[int]:
        """Return nearest SQLite IDs using cosine similarity via inner product."""
        self._validate_query(query, top_k)
        if embedder not in self.dense_indexes:
            raise ValueError(f"Dense embedder '{embedder}' is not loaded")
        index = self.dense_indexes[embedder]
        if index.ntotal == 0:
            return []
        vector = np.asarray(
            self.embedders[embedder].embed_query(query), dtype=np.float32
        )
        expected_shape = (1, self.embedders[embedder].get_dimension())
        if vector.shape != expected_shape or not np.isfinite(vector).all():
            raise ValueError(
                f"Embedder returned invalid query vector with shape {vector.shape}"
            )
        _, found_ids = index.search(
            np.ascontiguousarray(vector), min(top_k, index.ntotal)
        )
        return [int(doc_id) for doc_id in found_ids[0] if doc_id >= 0]

    def search_sparse(self, query: str, top_k: int) -> list[int]:
        """Return positive-scoring BM25 results mapped through saved SQLite IDs."""
        self._validate_query(query, top_k)
        if len(self.sparse_doc_ids) == 0:
            return []
        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results = self.sparse_index.retrieve(
            query_tokens,
            k=min(top_k, len(self.sparse_doc_ids)),
            sorted=True,
            return_as="tuple",
            show_progress=False,
        )
        positions = results.documents[0]
        scores = results.scores[0]
        output: list[int] = []
        for position, score in zip(positions, scores):
            position = int(position)
            if (
                position < 0
                or float(score) <= 0
                or position >= len(self.sparse_doc_ids)
            ):
                continue
            output.append(int(self.sparse_doc_ids[position]))
        return output

    def search_hybrid(
        self, query: str, top_k: int, embedder: str = "local"
    ) -> list[int]:
        """Fuse dense and sparse top-50 results using reciprocal rank fusion."""
        self._validate_query(query, top_k)
        dense_ids = self.search_dense(query, 50, embedder=embedder)
        sparse_ids = self.search_sparse(query, 50)
        scores: dict[int, float] = {}
        for ranked_ids in (dense_ids, sparse_ids):
            for rank, doc_id in enumerate(ranked_ids, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
        return [
            doc_id
            for doc_id, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )[:top_k]
        ]

    def get_metadata(self, doc_ids: list[int]) -> list[dict]:
        """Fetch metadata in requested order, including SQLite and arXiv IDs."""
        if not doc_ids:
            return []
        unique_ids = tuple(dict.fromkeys(int(doc_id) for doc_id in doc_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        sql = (
            "SELECT id, arxiv_id, title, abstract, authors, categories, published, url "
            f"FROM papers WHERE id IN ({placeholders})"
        )
        with self._connection_lock:
            rows = self._connection.execute(sql, unique_ids).fetchall()
        by_id = {int(row["id"]): dict(row) for row in rows}
        return [by_id[int(doc_id)] for doc_id in doc_ids if int(doc_id) in by_id]

    def has_arxiv_id(self, arxiv_id: str) -> bool:
        """Check that a stable arXiv identifier belongs to this corpus."""
        with self._connection_lock:
            return (
                self._connection.execute(
                    "SELECT 1 FROM papers WHERE arxiv_id = ? LIMIT 1", (arxiv_id,)
                ).fetchone()
                is not None
            )

    def get_metadata_by_arxiv_ids(self, arxiv_ids: list[str]) -> list[dict]:
        """Fetch paper metadata in the requested stable arXiv-ID order."""
        if not arxiv_ids:
            return []
        unique_ids = tuple(dict.fromkeys(str(value) for value in arxiv_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        sql = (
            "SELECT id, arxiv_id, title, abstract, authors, categories, published, url "
            f"FROM papers WHERE arxiv_id IN ({placeholders})"
        )
        with self._connection_lock:
            rows = self._connection.execute(sql, unique_ids).fetchall()
        by_arxiv_id = {str(row["arxiv_id"]): dict(row) for row in rows}
        return [by_arxiv_id[value] for value in arxiv_ids if value in by_arxiv_id]
