"""Print retrieval metrics from human judgments in labels.db."""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path
from typing import Any

from benchmark_runner import run_labeled_benchmark
from labels_store import connect_labels
from search_engine import HybridSearchEngine


def _print_table(result: dict[str, Any], k_values: list[int]) -> None:
    if result["status"] != "ok":
        print(f"Benchmark pending: {result['message']}")
        return
    headings = (
        ["Method"]
        + [f"Recall@{k}" for k in k_values]
        + [f"NDCG@{k}" for k in k_values]
        + ["Latency ms"]
    )
    method_names = list(result["methods"])
    widths = [max(len(headings[0]), max(map(len, method_names)))] + [
        max(10, len(item)) for item in headings[1:]
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(
        f"\nHuman-labeled benchmark: {result['num_queries']} queries, {result['num_judgments']} judgments"
    )
    print(border)
    print(
        "| "
        + " | ".join(value.ljust(width) for value, width in zip(headings, widths))
        + " |"
    )
    print(border)
    for name, scores in result["methods"].items():
        values = [scores[f"recall@{k}"] for k in k_values]
        values += [scores[f"ndcg@{k}"] for k in k_values]
        cells = [name.ljust(widths[0])] + [
            f"{value:.4f}".rjust(width) for value, width in zip(values, widths[1:-1])
        ]
        cells.append(f"{scores['latency_ms']:.1f}".rjust(widths[-1]))
        print("| " + " | ".join(cells) + " |")
    print(border)
    for method, metrics in result["lift_hybrid_vs_bm25"].get("by_embedder", {}).items():
        print(
            f"{method} lift over BM25: "
            + ", ".join(
                f"{metric} {value if value == 'n/a' else f'{value:+.1f}%'}"
                for metric, value in metrics.items()
            )
        )
    comparison = result.get("ndcg10_comparison", {})
    interval = comparison.get("confidence_interval_95")
    if interval:
        print(
            "NDCG@10 wins/ties/losses (Hybrid local vs BM25): "
            f"{comparison['hybrid_wins']}/{comparison['ties']}/{comparison['bm25_wins']}"
        )
        print(
            "Mean NDCG@10 difference (95% bootstrap CI): "
            f"{comparison['mean_difference']:+.4f} "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
        )
    coverage = result.get("pool_coverage", {})
    if coverage.get("warning"):
        print(
            f"Pool coverage warning: {coverage['warning']} "
            f"({coverage['out_of_pool_count']} results)"
        )
    if result.get("zero_relevance_query_count"):
        print(
            "Queries excluded for having no Relevant judgments: "
            f"{result['zero_relevance_query_count']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("papers.db"))
    parser.add_argument("--faiss-index", type=Path, default=Path("faiss_local.index"))
    parser.add_argument("--bge-index", type=Path, default=Path("faiss_bge.index"))
    parser.add_argument("--bm25-index", type=Path, default=Path("bm25_index"))
    parser.add_argument("--labels-db", type=Path, default=Path("labels.db"))
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10])
    parser.add_argument(
        "--save",
        type=Path,
        help="Write the full benchmark JSON only after at least 12 labeled queries qualify",
    )
    args = parser.parse_args()

    dense_indexes = {"local": args.faiss_index}
    if args.bge_index.is_file():
        dense_indexes["bge"] = args.bge_index

    with (
        closing(connect_labels(args.labels_db)) as labels_connection,
        closing(
            HybridSearchEngine(
                database_path=args.db,
                dense_index_path=args.faiss_index,
                sparse_index_path=args.bm25_index,
                dense_indexes=dense_indexes,
            )
        ) as engine,
    ):
        result = run_labeled_benchmark(engine, labels_connection, args.k_values)

    if result["status"] != "ok":
        _print_table(result, args.k_values)
        if args.save:
            print("No benchmark file written; real human labels are required first.")
        return 2 if args.save else 0
    if args.save and int(result["num_queries"]) < 12:
        print(
            f"Benchmark has {result['num_queries']} positively labeled queries; "
            "at least 12 are required before saving benchmark_results.json."
        )
        return 2

    _print_table(result, args.k_values)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved benchmark results to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
