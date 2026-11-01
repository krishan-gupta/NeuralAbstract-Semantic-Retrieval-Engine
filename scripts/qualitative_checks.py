"""Write qualitative dense/sparse comparison evidence for representative queries."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from search_engine import HybridSearchEngine

QUERIES = (
    "medical",
    "LoRA",
    "making large models run on phones",
    "teaching AI to say it doesn't know",
    "reducing hallucinations in chatbots",
)


def _markdown_title(title: str) -> str:
    return " ".join(title.split()).replace("|", "\\|")


def write_qualitative_checks(
    database: Path = Path("papers.db"),
    dense_index: Path = Path("faiss_local.index"),
    sparse_index: Path = Path("bm25_index"),
    output: Path = Path("docs/qualitative_checks.md"),
    top_k: int = 10,
) -> Path:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qualitative retrieval checks",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Corpus: `{database}`; dense index: `{dense_index}`; sparse index: `{sparse_index}`.",
        f"Cutoff: top {top_k} per method.",
        "",
        (
            "Overlap is agreement between ranked lists, not retrieval accuracy. "
            "Review the titles below to judge whether Hybrid adds useful results."
        ),
        "",
    ]

    engine = HybridSearchEngine(
        database_path=database,
        dense_index_path=dense_index,
        sparse_index_path=sparse_index,
    )
    try:
        for query in QUERIES:
            hybrid_ids = engine.search_hybrid(query, top_k)
            bm25_ids = engine.search_sparse(query, top_k)
            hybrid_rows = engine.get_metadata(hybrid_ids)
            bm25_rows = engine.get_metadata(bm25_ids)
            bm25_set = {int(row["id"]) for row in bm25_rows}
            hybrid_set = {int(row["id"]) for row in hybrid_rows}
            overlap = len(hybrid_set & bm25_set)
            denominator = max(len(hybrid_ids), len(bm25_ids), 1)
            overlap_percent = round(100.0 * overlap / denominator, 1)
            hybrid_only = [row for row in hybrid_rows if int(row["id"]) not in bm25_set]
            bm25_only = [row for row in bm25_rows if int(row["id"]) not in hybrid_set]

            lines.extend(
                [
                    f"## {query}",
                    "",
                    f"- Overlap: **{overlap_percent}%** ({overlap} shared; {len(hybrid_ids)} Hybrid results; {len(bm25_ids)} BM25 results).",
                    "- Titles found only by Hybrid:",
                ]
            )
            lines.extend(
                f"  - {_markdown_title(str(row['title']))}" for row in hybrid_only
            )
            if not hybrid_only:
                lines.append("  - None")
            lines.append("- Titles found only by BM25:")
            lines.extend(
                f"  - {_markdown_title(str(row['title']))}" for row in bm25_only
            )
            if not bm25_only:
                lines.append("  - None")
            lines.append("")

    finally:
        engine.close()
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("papers.db"))
    parser.add_argument("--faiss-index", type=Path, default=Path("faiss_local.index"))
    parser.add_argument("--bm25-index", type=Path, default=Path("bm25_index"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/qualitative_checks.md")
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    output = write_qualitative_checks(
        database=args.db,
        dense_index=args.faiss_index,
        sparse_index=args.bm25_index,
        output=args.output,
        top_k=args.top_k,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
