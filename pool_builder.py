"""Persistent, blind candidate pools shared by every retrieval system."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from labels_store import add_pool_candidates, list_labels, list_pool

if TYPE_CHECKING:
    from search_engine import HybridSearchEngine


def build_query_pool(
    engine: HybridSearchEngine, connection: sqlite3.Connection, query_id: int
) -> tuple[str, int]:
    """Add top-20 candidates from BM25 and every loaded dense/hybrid system."""
    query_row = connection.execute(
        "SELECT text FROM benchmark_queries WHERE id = ?", (query_id,)
    ).fetchone()
    if query_row is None:
        raise KeyError(f"Benchmark query {query_id} does not exist")

    query_text = str(query_row["text"])
    candidate_ids = engine.search_sparse(query_text, 20)
    for tag in engine.dense_indexes:
        candidate_ids.extend(engine.search_dense(query_text, 20, embedder=tag))
        candidate_ids.extend(engine.search_hybrid(query_text, 20, embedder=tag))
    candidates = engine.get_metadata(list(dict.fromkeys(candidate_ids)))
    added = add_pool_candidates(
        connection,
        query_id,
        [str(paper["arxiv_id"]) for paper in candidates],
    )
    return query_text, added


def read_query_pool(
    engine: HybridSearchEngine, connection: sqlite3.Connection, query_id: int
) -> dict[str, Any]:
    """Read the pool and saved judgments in its persisted stable order."""
    query_row = connection.execute(
        "SELECT text FROM benchmark_queries WHERE id = ?", (query_id,)
    ).fetchone()
    if query_row is None:
        raise KeyError(f"Benchmark query {query_id} does not exist")
    papers = engine.get_metadata_by_arxiv_ids(list_pool(connection, query_id))
    labels = list_labels(connection, query_id)
    for paper in papers:
        state = labels.get(str(paper["arxiv_id"]))
        paper["labeled"] = state is not None
        paper["relevant"] = bool(state) if state is not None else None
    return {
        "query_id": query_id,
        "query": str(query_row["text"]),
        "papers": papers,
        "labeled_count": len(labels),
        "pool_count": len(papers),
        "pool_built": bool(papers),
    }


def check_pool_coverage(
    engine: HybridSearchEngine, labels_connection: sqlite3.Connection
) -> dict[str, Any]:
    """Report labeled-query top-10 results absent from the persisted pools."""
    query_rows = labels_connection.execute(
        """SELECT q.id, q.text FROM benchmark_queries q
           WHERE EXISTS (SELECT 1 FROM labels l WHERE l.query_id = q.id)
           ORDER BY q.id"""
    ).fetchall()
    systems = {"bm25": None}
    for tag in engine.dense_indexes:
        systems[f"dense:{tag}"] = tag
        systems[f"hybrid:{tag}"] = tag

    pool_exists = labels_connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pool'"
    ).fetchone()
    warnings: list[dict[str, Any]] = []
    if query_rows and not pool_exists:
        warnings = [
            {
                "query_id": int(row["id"]),
                "query": str(row["text"]),
                "method": method,
                "count": None,
            }
            for row in query_rows
            for method in systems
        ]
        outside_count = len(warnings)
    else:
        for query_row in query_rows:
            query_id = int(query_row["id"])
            query_text = str(query_row["text"])
            pool_rows = labels_connection.execute(
                "SELECT arxiv_id FROM pool WHERE query_id = ?", (query_id,)
            ).fetchall()
            pool_ids = {str(row[0]) for row in pool_rows}
            for method, tag in systems.items():
                if method == "bm25":
                    sqlite_ids = engine.search_sparse(query_text, 10)
                elif method.startswith("dense:"):
                    sqlite_ids = engine.search_dense(query_text, 10, embedder=tag)
                else:
                    sqlite_ids = engine.search_hybrid(query_text, 10, embedder=tag)
                outside = [
                    str(paper["arxiv_id"])
                    for paper in engine.get_metadata(sqlite_ids)
                    if str(paper["arxiv_id"]) not in pool_ids
                ]
                if outside:
                    warnings.append(
                        {
                            "query_id": query_id,
                            "query": query_text,
                            "method": method,
                            "count": len(outside),
                            "arxiv_ids": outside,
                        }
                    )
        outside_count = sum(int(item["count"] or 0) for item in warnings)

    return {
        "status": "out_of_date" if outside_count else "up_to_date",
        "out_of_pool_count": outside_count,
        "warning": (
            f"{outside_count} top-10 system results are outside the label pools. "
            "Refresh affected pools before trusting benchmark metrics."
            if outside_count
            else None
        ),
        "warnings": warnings,
    }
