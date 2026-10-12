"""Build dense FAISS and sparse BM25 indexes from papers.db."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import bm25s
import faiss
import numpy as np

from embeddings import BaseEmbeddingProvider, LocalSentenceTransformerProvider

LOGGER = logging.getLogger(__name__)


def load_papers(database_path: Path) -> list[tuple[int, str, str]]:
    """Load paper identifiers and text in stable SQLite ID order."""
    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT id, title, abstract FROM papers ORDER BY id ASC"
        ).fetchall()
    if not rows:
        raise ValueError(f"No papers found in {database_path}; run ingest.py first")
    return [(int(row[0]), row[1] or "", row[2] or "") for row in rows]


def build_indexes(
    database_path: Path = Path("papers.db"),
    dense_index_path: Path = Path("faiss_local.index"),
    sparse_index_path: Path = Path("bm25_index"),
    provider: BaseEmbeddingProvider | None = None,
) -> None:
    """Build and persist both indexes; BM25 document IDs follow SQLite IDs."""
    rows = load_papers(database_path)
    ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    documents = [f"{title}. {abstract}".strip() for _, title, abstract in rows]
    embedder = provider or LocalSentenceTransformerProvider()

    embeddings = np.asarray(embedder.embed_texts(documents), dtype=np.float32)
    expected = (len(documents), embedder.get_dimension())
    if embeddings.shape != expected:
        raise ValueError(f"Embedder returned shape {embeddings.shape}; expected {expected}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    dense_index = faiss.IndexIDMap(faiss.IndexFlatIP(embedder.get_dimension()))
    dense_index.add_with_ids(np.ascontiguousarray(embeddings), ids)
    dense_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(dense_index, str(dense_index_path))

    tokenized = bm25s.tokenize(documents, stopwords="en", show_progress=True)
    sparse_index = bm25s.BM25(corpus=[{"id": int(paper_id), "text": doc}
                                      for paper_id, doc in zip(ids, documents)])
    sparse_index.index(tokenized)
    sparse_index_path.mkdir(parents=True, exist_ok=True)
    sparse_index.save(str(sparse_index_path), corpus=[{"id": int(paper_id), "text": doc}
                                                       for paper_id, doc in zip(ids, documents)])
    LOGGER.info("Built indexes for %d papers", len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("papers.db"))
    parser.add_argument("--faiss-index", type=Path, default=Path("faiss_local.index"))
    parser.add_argument("--bm25-index", type=Path, default=Path("bm25_index"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        build_indexes(args.db, args.faiss_index, args.bm25_index)
    except Exception:
        LOGGER.exception("Index build failed")
        raise


if __name__ == "__main__":
    main()
