"""Regenerate the README results block from benchmark_results.json only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
PENDING = (
    "Results pending: run the labeling workflow and "
    "python evaluate.py --save benchmark_results.json."
)


def _format_percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%" if isinstance(value, (int, float)) else "—"


def _format_number(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "—"


def render_results(result_path: Path) -> str:
    if not result_path.is_file():
        return PENDING
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok" or int(result.get("num_queries", 0)) < 12:
        raise ValueError(
            "benchmark_results.json must contain at least 12 evaluated queries"
        )

    lines = [
        "| Method / embedder | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Latency (ms) | Index size (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    methods = result.get("methods", {})
    for name, metrics in methods.items():
        size_mib = float(metrics.get("index_size_bytes", 0)) / (1024 * 1024)
        lines.append(
            "| {name} | {r5} | {r10} | {n5} | {n10} | {latency} | {size:.2f} |".format(
                name=name,
                r5=_format_percent(metrics.get("recall@5")),
                r10=_format_percent(metrics.get("recall@10")),
                n5=_format_number(metrics.get("ndcg@5")),
                n10=_format_number(metrics.get("ndcg@10")),
                latency=(
                    f"{float(metrics['latency_ms']):.2f}"
                    if isinstance(metrics.get("latency_ms"), (int, float))
                    else "—"
                ),
                size=size_mib,
            )
        )

    lift = (
        result.get("lift_hybrid_vs_bm25", {})
        .get("by_embedder", {})
        .get("hybrid:local", {})
        .get("recall@10", "n/a")
    )
    lift_text = f"{float(lift):+.1f}%" if isinstance(lift, (int, float)) else "n/a"
    comparison = result.get("ndcg10_comparison", {})
    interval = comparison.get("confidence_interval_95")
    if isinstance(interval, list) and len(interval) == 2:
        interval_text = (
            f"[{_format_number(interval[0])}, {_format_number(interval[1])}]"
        )
    else:
        interval_text = "not available"
    wins = int(comparison.get("hybrid_wins", 0))
    ties = int(comparison.get("ties", 0))
    losses = int(comparison.get("bm25_wins", 0))
    queries = int(result["num_queries"])
    judgments = int(result["num_judgments"])
    sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    lines.extend(
        [
            "",
            f"Hybrid local Recall@10 lift over BM25: **{lift_text}**.",
            "",
            (
                "Mean paired NDCG@10 difference (Hybrid local − BM25), 95% bootstrap CI "
                f"(1,000 resamples): **{interval_text}**."
            ),
            "",
            f"Evaluation used **{queries} queries, {judgments} judgments, 1 labeler, pooled judgments**.",
            "",
            f"NDCG@10 wins/ties/losses (Hybrid local vs BM25): **{wins}/{ties}/{losses}**.",
            "",
            (
                f"On these {queries} queries, Hybrid local Recall@10 was "
                f"{_format_percent(methods.get('hybrid:local', {}).get('recall@10'))} "
                f"and BM25 Recall@10 was {_format_percent(methods.get('bm25', {}).get('recall@10'))}; "
                f"the mean NDCG@10 difference was {_format_number(comparison.get('mean_difference'))} "
                f"with a 95% interval of {interval_text}."
            ),
            "",
            f"<!-- BENCHMARK_RESULTS_SHA256:{sha256} -->",
        ]
    )
    return "\n".join(lines)


def update_readme(readme_path: Path, result_path: Path) -> str:
    text = readme_path.read_text(encoding="utf-8")
    block = render_results(result_path)
    if START not in text or END not in text or text.index(START) > text.index(END):
        heading = "## Results"
        if heading in text:
            position = text.index(heading) + len(heading)
            text = text[:position] + f"\n\n{START}\n{END}\n" + text[position:]
        else:
            text = text.rstrip() + f"\n\n{heading}\n\n{START}\n{END}\n"
    start_at = text.index(START) + len(START)
    end_at = text.index(END)
    updated = text[:start_at] + "\n" + block + "\n" + text[end_at:]
    readme_path.write_text(updated, encoding="utf-8")
    return block


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--results", type=Path, default=ROOT / "benchmark_results.json")
    args = parser.parse_args()
    try:
        block = update_readme(args.readme, args.results)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
