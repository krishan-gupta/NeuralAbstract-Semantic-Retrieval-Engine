"""Shared human-labeled benchmark runner used by the API and CLI."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metrics import bootstrap_mean_ci, ndcg_at_k, recall_at_k
from pool_builder import check_pool_coverage

if TYPE_CHECKING:
    from search_engine import HybridSearchEngine


def _ranked_arxiv_ids(engine: HybridSearchEngine, sqlite_ids: list[int]) -> list[str]:
    return [str(paper["arxiv_id"]) for paper in engine.get_metadata(sqlite_ids)]


def run_labeled_benchmark(
    engine: HybridSearchEngine,
    labels_connection: sqlite3.Connection,
    k_values: list[int] | tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Compute macro Recall/NDCG and latency from explicit human judgments."""
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    k_values = list(dict.fromkeys(int(k) for k in k_values))
    query_rows = labels_connection.execute(
        "SELECT id, text FROM benchmark_queries ORDER BY id"
    ).fetchall()
    judgment_count = int(
        labels_connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    )
    if judgment_count == 0:
        return {
            "status": "no_labels",
            "message": "No human labels yet. Open the Label tab and judge a few papers first.",
            "num_queries": 0,
            "num_judgments": 0,
            "methods": {},
            "lift_hybrid_vs_bm25": {},
            "per_query": [],
            "pool_coverage": check_pool_coverage(engine, labels_connection),
            "timestamp": datetime.now(UTC).isoformat(),
            "zero_relevance_query_count": 0,
        }

    labels_by_query: dict[int, dict[str, int]] = {}
    relevant_by_query: dict[int, set[str]] = {}
    eligible: list[tuple[int, str, set[str]]] = []
    for query_row in query_rows:
        query_id = int(query_row["id"])
        judgments = labels_connection.execute(
            "SELECT arxiv_id, relevant FROM labels WHERE query_id = ?", (query_id,)
        ).fetchall()
        if not judgments:
            continue
        mapping = {str(row["arxiv_id"]): int(row["relevant"]) for row in judgments}
        relevant = {arxiv_id for arxiv_id, value in mapping.items() if value == 1}
        labels_by_query[query_id] = mapping
        relevant_by_query[query_id] = relevant
        if relevant:
            eligible.append((query_id, str(query_row["text"]), relevant))
    zero_relevance_query_count = sum(
        1 for mapping in labels_by_query.values() if not any(mapping.values())
    )
    if not eligible:
        return {
            "status": "no_relevant_labels",
            "message": "No papers are marked Relevant yet. Label at least one relevant paper to compute metrics.",
            "num_queries": 0,
            "num_judgments": judgment_count,
            "methods": {},
            "lift_hybrid_vs_bm25": {},
            "per_query": [],
            "pool_coverage": check_pool_coverage(engine, labels_connection),
            "timestamp": datetime.now(UTC).isoformat(),
            "zero_relevance_query_count": zero_relevance_query_count,
        }

    max_k = max(k_values)
    tags = list(engine.dense_indexes)
    method_names = ["bm25"] + [
        name for tag in tags for name in (f"dense:{tag}", f"hybrid:{tag}")
    ]
    sums: dict[str, dict[str, float]] = {
        name: {
            **{f"recall@{k}": 0.0 for k in k_values},
            **{f"ndcg@{k}": 0.0 for k in k_values},
            "latency_ms": 0.0,
        }
        for name in method_names
    }
    per_query: list[dict[str, Any]] = []
    for query_id, query_text, relevant in eligible:
        item: dict[str, Any] = {"query": query_text, "methods": {}}
        rankings: dict[str, list[str]] = {}
        searches = {"bm25": lambda query=query_text: engine.search_sparse(query, max_k)}
        for tag in tags:
            searches[f"dense:{tag}"] = lambda tag=tag, query=query_text: (
                engine.search_dense(query, max_k, embedder=tag)
            )
            searches[f"hybrid:{tag}"] = lambda tag=tag, query=query_text: (
                engine.search_hybrid(query, max_k, embedder=tag)
            )
        for method_name, search in searches.items():
            started = time.perf_counter()
            sqlite_ids = search()
            latency_ms = (time.perf_counter() - started) * 1000.0
            ranked = _ranked_arxiv_ids(engine, sqlite_ids)
            rankings[method_name] = ranked
            method_result: dict[str, float] = {"latency_ms": round(latency_ms, 2)}
            sums[method_name]["latency_ms"] += latency_ms
            for k in k_values:
                recall = recall_at_k(ranked, relevant, k)
                ndcg = ndcg_at_k(ranked, relevant, k)
                method_result[f"recall@{k}"] = round(recall, 6)
                method_result[f"ndcg@{k}"] = round(ndcg, 6)
                sums[method_name][f"recall@{k}"] += recall
                sums[method_name][f"ndcg@{k}"] += ndcg
            item["methods"][method_name] = method_result
        per_query.append(item)

    query_count = len(eligible)
    methods: dict[str, dict[str, float | int | str]] = {}
    for method_name in method_names:
        methods[method_name] = {
            metric: round(value / query_count, 6)
            for metric, value in sums[method_name].items()
        }

    lift_by_embedder: dict[str, dict[str, float | str]] = {}
    if 10 in k_values:
        for tag in tags:
            method_name = f"hybrid:{tag}"
            lift_by_embedder[method_name] = {}
            for metric in ("recall@10", "ndcg@10"):
                baseline = float(methods["bm25"][metric])
                hybrid = float(methods[method_name][metric])
                lift_by_embedder[method_name][metric] = (
                    round((hybrid - baseline) / baseline * 100.0, 1)
                    if baseline
                    else "n/a"
                )
    # Keep the original single-local-embedder fields at the top level while
    # exposing every model's lift in an additive map.
    lift: dict[str, Any] = dict(lift_by_embedder.get("hybrid:local", {}))
    lift["by_embedder"] = lift_by_embedder

    stats_path = Path("index_stats.json")
    try:
        stats = (
            json.loads(stats_path.read_text(encoding="utf-8"))
            if stats_path.exists()
            else {}
        )
    except (OSError, json.JSONDecodeError):
        stats = {}
    bm25_path = Path("bm25_index")
    bm25_size_bytes = (
        sum(file.stat().st_size for file in bm25_path.rglob("*") if file.is_file())
        if bm25_path.is_dir()
        else int(stats.get("local", {}).get("bm25_size_bytes") or 0)
    )
    methods["bm25"]["index_size_bytes"] = bm25_size_bytes
    for tag in tags:
        for method_name in (f"dense:{tag}", f"hybrid:{tag}"):
            methods[method_name]["index_size_bytes"] = int(
                stats.get(tag, {}).get("faiss_size_bytes", 0)
            )
            methods[method_name]["index_build_seconds"] = float(
                stats.get(tag, {}).get("build_seconds", 0.0)
            )

    comparison: dict[str, Any] = {
        "metric": "ndcg@10",
        "hybrid_wins": 0,
        "ties": 0,
        "bm25_wins": 0,
        "mean_difference": None,
        "confidence_interval_95": None,
        "bootstrap_resamples": 1000,
        "seed": 42,
    }
    if 10 in k_values and "hybrid:local" in methods:
        differences: list[float] = []
        for item in per_query:
            bm25_ndcg = float(item["methods"]["bm25"]["ndcg@10"])
            hybrid_ndcg = float(item["methods"]["hybrid:local"]["ndcg@10"])
            difference = hybrid_ndcg - bm25_ndcg
            differences.append(difference)
            if difference > 1e-12:
                comparison["hybrid_wins"] += 1
                outcome = "hybrid"
            elif difference < -1e-12:
                comparison["bm25_wins"] += 1
                outcome = "bm25"
            else:
                comparison["ties"] += 1
                outcome = "tie"
            item["ndcg10_comparison"] = {
                "hybrid_local_minus_bm25": round(difference, 6),
                "outcome": outcome,
            }
        mean, lower, upper = bootstrap_mean_ci(
            differences, n_resamples=1000, confidence=0.95, seed=42
        )
        comparison["mean_difference"] = round(mean, 6)
        comparison["confidence_interval_95"] = [round(lower, 6), round(upper, 6)]

    return {
        "status": "ok",
        "num_queries": query_count,
        "num_judgments": judgment_count,
        "methods": methods,
        "lift_hybrid_vs_bm25": lift,
        "per_query": per_query,
        "pool_coverage": check_pool_coverage(engine, labels_connection),
        "timestamp": datetime.now(UTC).isoformat(),
        "zero_relevance_query_count": zero_relevance_query_count,
        "ndcg10_comparison": comparison,
    }
