"""Build resumable FAISS and BM25 indexes from papers.db."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import bm25s
import faiss
import numpy as np
from tqdm import tqdm

from embeddings import (
    BaseEmbeddingProvider,
    LocalSentenceTransformerProvider,
    OpenAIEmbeddingProvider,
)

LOGGER = logging.getLogger(__name__)


def load_papers(database_path: Path) -> list[tuple[int, str, str]]:
    """Load papers in stable SQLite ID order."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT id, title, abstract FROM papers ORDER BY id ASC"
        ).fetchall()
    if not rows:
        raise ValueError(f"No papers found in {database_path}")
    return [(int(row[0]), row[1] or "", row[2] or "") for row in rows]


def make_provider(
    provider_name: str, model_name: str | None, local_files_only: bool = False
) -> tuple[BaseEmbeddingProvider, str]:
    """Create the selected provider and resolve its effective model name."""
    if provider_name == "local":
        effective_model = model_name or "all-MiniLM-L6-v2"
        return LocalSentenceTransformerProvider(
            effective_model, batch_size=64, local_files_only=local_files_only
        ), effective_model
    effective_model = model_name or "text-embedding-3-small"
    return OpenAIEmbeddingProvider(effective_model, batch_size=128), effective_model


def cache_fingerprint(
    database_path: Path,
    provider_name: str,
    model_name: str,
    tag: str,
    dimension: int,
    row_count: int,
) -> str:
    stat = database_path.stat()
    payload = {
        "provider": provider_name,
        "model": model_name,
        "tag": tag,
        "dimension": dimension,
        "row_count": row_count,
        "db_size": stat.st_size,
        "db_mtime_ns": stat.st_mtime_ns,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def save_chunk(path: Path, vectors: np.ndarray) -> None:
    """Atomically persist a completed embedding chunk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, vectors.astype(np.float32, copy=False), allow_pickle=False)
    temporary_path.replace(path)


def load_or_embed_chunks(
    documents: list[str],
    ids: np.ndarray,
    provider: BaseEmbeddingProvider,
    cache_dir: Path,
    chunk_size: int,
    sample_positions: np.ndarray,
) -> tuple[list[Path], dict[int, np.ndarray]]:
    """Reuse completed chunks or embed and atomically cache missing chunks."""
    chunk_paths: list[Path] = []
    sample_vectors: dict[int, np.ndarray] = {}
    starts = range(0, len(documents), chunk_size)
    for start in tqdm(
        starts,
        total=(len(documents) + chunk_size - 1) // chunk_size,
        desc="Embedding chunks",
        unit="chunk",
    ):
        stop = min(start + chunk_size, len(documents))
        path = cache_dir / f"chunk_{start}.npy"
        vectors: np.ndarray | None = None
        if path.is_file():
            try:
                cached = np.load(path, mmap_mode="r", allow_pickle=False)
                if (
                    cached.shape == (stop - start, provider.get_dimension())
                    and cached.dtype == np.float32
                ):
                    cached_array = np.asarray(cached)
                    cached_norms = np.linalg.norm(cached_array, axis=1)
                    normalized = np.all(
                        (cached_norms == 0) | np.isclose(cached_norms, 1.0, atol=1e-3)
                    )
                    if np.isfinite(cached_array).all() and normalized:
                        vectors = cached_array
                        LOGGER.info("Reusing cached embedding chunk %s", path.name)
            except (OSError, ValueError):
                vectors = None
        if vectors is None:
            vectors = np.asarray(
                provider.embed_texts(documents[start:stop]), dtype=np.float32
            )
            expected = (stop - start, provider.get_dimension())
            if vectors.shape != expected:
                raise ValueError(
                    f"Embedder returned shape {vectors.shape}; expected {expected}"
                )
            if not np.isfinite(vectors).all():
                raise ValueError("Embedding chunk contains NaN or infinite values")
            norms = np.linalg.norm(vectors, axis=1)
            if np.any((norms > 0) & ~np.isclose(norms, 1.0, atol=1e-3)):
                raise ValueError("Embedding provider must return L2-normalized vectors")
            vectors = np.ascontiguousarray(vectors)
            save_chunk(path, vectors)
        chunk_paths.append(path)
        local_positions = sample_positions[
            (sample_positions >= start) & (sample_positions < stop)
        ]
        for position in local_positions:
            sample_vectors[int(position)] = np.array(
                vectors[int(position) - start], dtype=np.float32
            )
    return chunk_paths, sample_vectors


def build_indexes(
    database_path: Path = Path("papers.db"),
    provider_name: str = "local",
    model_name: str | None = None,
    tag: str = "local",
    chunk_size: int = 2000,
    skip_bm25: bool = False,
    provider: BaseEmbeddingProvider | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Build indexes in chunks, resuming cached embeddings after interruption."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    rows = load_papers(database_path)
    ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    documents = [f"{title}. {abstract}" for _, title, abstract in rows]
    titles = [title for _, title, _ in rows]

    if provider is None:
        embedder, effective_model = make_provider(
            provider_name, model_name, local_files_only
        )
        effective_provider = provider_name
    else:
        embedder = provider
        effective_model = model_name or type(provider).__name__
        effective_provider = f"custom:{type(provider).__name__}"
    dimension = embedder.get_dimension()
    fingerprint = cache_fingerprint(
        database_path, effective_provider, effective_model, tag, dimension, len(rows)
    )
    cache_dir = Path("emb_cache") / fingerprint
    index_path = Path(f"faiss_{tag}.index")
    index_started = time.perf_counter()

    rng = np.random.default_rng(42)
    sample_positions = np.sort(
        rng.choice(len(rows), size=min(5, len(rows)), replace=False)
    )
    chunk_paths, sample_vectors = load_or_embed_chunks(
        documents, ids, embedder, cache_dir, chunk_size, sample_positions
    )

    dense_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
    for start, chunk_path in zip(range(0, len(rows), chunk_size), chunk_paths):
        vectors = np.load(chunk_path, allow_pickle=False)
        stop = start + len(vectors)
        dense_index.add_with_ids(
            np.ascontiguousarray(vectors, dtype=np.float32), ids[start:stop]
        )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(dense_index, str(index_path))

    dense_queries = np.ascontiguousarray(
        np.stack([sample_vectors[int(position)] for position in sample_positions]),
        dtype=np.float32,
    )
    _, dense_hits = dense_index.search(dense_queries, 1)
    dense_passed = all(
        int(hit) == int(ids[position])
        for hit, position in zip(dense_hits[:, 0], sample_positions)
    )
    if not dense_passed:
        raise RuntimeError(
            f"Dense self-check failed: expected own vector at rank 1, got {dense_hits[:, 0].tolist()}"
        )

    bm25_hits = 0
    if not skip_bm25:
        sparse_path = Path("bm25_index")
        sparse_path.mkdir(parents=True, exist_ok=True)
        tokenized = bm25s.tokenize(documents, stopwords="en", show_progress=True)
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        retriever.save(str(sparse_path), show_progress=True)
        doc_ids_path = sparse_path / "doc_ids.npy"
        temporary_doc_ids_path = sparse_path / "doc_ids.npy.tmp"
        with temporary_doc_ids_path.open("wb") as handle:
            np.save(handle, ids, allow_pickle=False)
        temporary_doc_ids_path.replace(doc_ids_path)

        title_tokens = bm25s.tokenize(
            [titles[int(position)] for position in sample_positions],
            stopwords="en",
            show_progress=False,
        )
        retrieved_positions, retrieved_scores = retriever.retrieve(
            title_tokens,
            k=min(5, len(rows)),
            sorted=True,
            return_as="tuple",
            show_progress=False,
        )
        for row_num, position in enumerate(sample_positions):
            top_position = int(retrieved_positions[row_num, 0])
            top_score = float(retrieved_scores[row_num, 0])
            if (
                top_position >= 0
                and top_score > 0
                and int(ids[top_position]) == int(ids[position])
            ):
                bm25_hits += 1
        if bm25_hits < max(0, len(sample_positions) - 1):
            LOGGER.warning(
                "BM25 title self-check: %d/%d returned the source paper at rank 1",
                bm25_hits,
                len(sample_positions),
            )

    elapsed = time.perf_counter() - index_started
    index_size = index_path.stat().st_size
    sparse_size = (
        sum(
            path.stat().st_size
            for path in Path("bm25_index").glob("*")
            if path.is_file()
        )
        if not skip_bm25
        else 0
    )
    stats_path = Path("index_stats.json")
    try:
        stats = (
            json.loads(stats_path.read_text(encoding="utf-8"))
            if stats_path.exists()
            else {}
        )
    except (json.JSONDecodeError, OSError):
        stats = {}
    stats[tag] = {
        "provider": effective_provider,
        "model": effective_model,
        "dimension": dimension,
        "vectors": len(rows),
        "index_path": str(index_path),
        "build_seconds": round(elapsed, 3),
        "faiss_size_bytes": index_size,
        "bm25_size_bytes": sparse_size
        if not skip_bm25
        else stats.get(tag, {}).get("bm25_size_bytes"),
        "cache_fingerprint": fingerprint,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nBuilt {dense_index.ntotal:,} dense vectors ({dimension} dimensions).")
    print(f"FAISS: {index_path} ({index_size / (1024 * 1024):.2f} MiB); {elapsed:.1f}s")
    print(f"Dense self-check: {len(sample_positions)}/{len(sample_positions)} passed.")
    if not skip_bm25:
        print(
            f"BM25: bm25_index/ ({sparse_size / (1024 * 1024):.2f} MiB); title self-check {bm25_hits}/{len(sample_positions)}."
        )
        if bm25_hits < max(0, len(sample_positions) - 1):
            raise RuntimeError(
                "BM25 title self-check failed: more than one sampled paper missed rank 1"
            )
    return stats[tag]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=Path("papers.db"), help="SQLite database path"
    )
    parser.add_argument("--provider", choices=("local", "openai"), default="local")
    parser.add_argument("--model", help="Embedding model name")
    parser.add_argument(
        "--tag", default="local", help="Dense index tag (default: local)"
    )
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the embedding model to already exist in the local Hugging Face cache",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        build_indexes(
            args.db,
            args.provider,
            args.model,
            args.tag,
            args.chunk_size,
            args.skip_bm25,
            local_files_only=args.local_files_only,
        )
    except Exception:
        LOGGER.exception("Index build failed")
        raise


if __name__ == "__main__":
    main()
