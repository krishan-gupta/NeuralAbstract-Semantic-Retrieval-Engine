"""Evaluate sparse, dense, and hybrid retrieval against keyword-derived labels.

Run from the project root with: python evaluate.py --db papers.db
Ground truth IDs are resolved from title/abstract keyword matches in the local DB,
so the benchmark remains usable with different arXiv snapshots.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from search_engine import HybridSearchEngine


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    relevant_keywords: tuple[str, ...]


BENCHMARK: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("diffusion models for image synthesis", ("diffusion model", "denoising diffusion", "stable diffusion")),
    BenchmarkCase("low rank adaptation LoRA fine tuning", ("low-rank adaptation", "lora", "parameter-efficient fine-tuning")),
    BenchmarkCase("mixture of experts routing for large language models", ("mixture of experts", "expert routing", "sparse expert")),
    BenchmarkCase("graph neural networks for molecular property prediction", ("molecular property prediction", "graph neural network", "molecular graph")),
    BenchmarkCase("FlashAttention GPU optimized attention", ("flashattention", "flash attention", "io-aware attention")),
    BenchmarkCase("vision transformers patch projection image classification", ("vision transformer", "vision transformers", "patch embedding")),
    BenchmarkCase("retrieval augmented generation with external knowledge", ("retrieval-augmented generation", "retrieval augmented generation", "retriever-generator")),
    BenchmarkCase("reinforcement learning from human feedback language models", ("reinforcement learning from human feedback", "rlhf", "preference model")),
    BenchmarkCase("contrastive learning visual language representation", ("contrastive language-image", "contrastive learning", "image-text representation")),
    BenchmarkCase("long context efficient transformer attention", ("long-context", "linear attention", "efficient transformer")),
)


def resolve_relevant_ids(database_path: Path) -> dict[str, set[int]]:
    """Resolve each case's labels to unique IDs using title/abstract phrases."""
    with sqlite3.connect(database_path) as connection:
        papers = connection.execute("SELECT id, title, abstract FROM papers ORDER BY id").fetchall()
    if not papers:
        raise ValueError(f"No papers found in {database_path}")

    labels: dict[str, set[int]] = {}
    for case in BENCHMARK:
        phrases = tuple(keyword.casefold() for keyword in case.relevant_keywords)
        labels[case.query] = {
            int(paper_id)
            for paper_id, title, abstract in papers
            if any(phrase in f"{title or ''} {abstract or ''}".casefold() for phrase in phrases)
        }
    return labels


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    relevances = np.asarray([1.0 if paper_id in relevant_ids else 0.0 for paper_id in ranked_ids[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2, dtype=np.float64))
    dcg = float(np.sum(relevances * discounts))
    ideal_count = min(len(relevant_ids), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_count + 2, dtype=np.float64))))
    return dcg / idcg if idcg else 0.0


def _format_lift(hybrid: float, baseline: float, metric: str) -> str:
    if baseline == 0:
        return f"N/A {metric} (BM25 baseline is 0)"
    return f"{(hybrid - baseline) / baseline * 100:+.1f}% {metric}"


def evaluate(database_path: Path, top_ks: tuple[int, ...] = (5, 10)) -> None:
    """Run each benchmark query in all modes and print aggregate scores."""
    labels = resolve_relevant_ids(database_path)
    usable = [(case, labels[case.query]) for case in BENCHMARK if labels[case.query]]
    skipped = len(BENCHMARK) - len(usable)
    if skipped:
        print(f"Note: skipped {skipped} benchmark queries with no keyword matches in this DB.")
    if not usable:
        raise ValueError("No benchmark cases matched papers.db; expand the keyword labels for this corpus")

    engine = HybridSearchEngine(database_path=database_path)
    methods: dict[str, Callable[[str, int], list[int]]] = {
        "BM25 Sparse": engine.search_sparse,
        "FAISS Dense": engine.search_dense,
        "Hybrid RRF": engine.search_hybrid,
    }
    totals = {method: {k: {"recall": 0.0, "ndcg": 0.0} for k in top_ks} for method in methods}
    try:
        for case, relevant_ids in usable:
            rankings = {name: search(case.query, max(top_ks)) for name, search in methods.items()}
            for name, ranked_ids in rankings.items():
                for k in top_ks:
                    totals[name][k]["recall"] += recall_at_k(ranked_ids, relevant_ids, k)
                    totals[name][k]["ndcg"] += ndcg_at_k(ranked_ids, relevant_ids, k)
    finally:
        engine.close()

    count = len(usable)
    print(f"\nEvaluation over {count} of {len(BENCHMARK)} benchmark queries\n")
    headings = ["Method"] + [f"Recall@{k}" for k in top_ks] + [f"NDCG@{k}" for k in top_ks]
    values: dict[str, list[float]] = {}
    for name in methods:
        values[name] = ([totals[name][k]["recall"] / count for k in top_ks]
                        + [totals[name][k]["ndcg"] / count for k in top_ks])
    widths = [max(len(headings[0]), max(map(len, methods)))] + [max(len(h), 9) for h in headings[1:]]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(border)
    print("| " + " | ".join(h.ljust(width) for h, width in zip(headings, widths)) + " |")
    print(border)
    for name, scores in values.items():
        print("| " + " | ".join([name.ljust(widths[0])] + [f"{score:.4f}".rjust(width) for score, width in zip(scores, widths[1:])]) + " |")
    print(border)
    hybrid = values["Hybrid RRF"]
    sparse = values["BM25 Sparse"]
    recall10_index = headings.index("Recall@10") - 1
    ndcg10_index = headings.index("NDCG@10") - 1
    print("Hybrid lift over BM25: " + _format_lift(hybrid[recall10_index], sparse[recall10_index], "Recall@10")
          + "; " + _format_lift(hybrid[ndcg10_index], sparse[ndcg10_index], "NDCG@10"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("papers.db"))
    args = parser.parse_args()
    evaluate(args.db)


if __name__ == "__main__":
    main()
