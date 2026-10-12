"""Fetch arXiv papers and store them in a local SQLite database."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Iterator

import arxiv
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"


def initialize_database(database_path: Path) -> None:
    """Create the papers table if it does not already exist."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL,
                categories TEXT NOT NULL,
                published TEXT NOT NULL,
                url TEXT NOT NULL
            )"""
        )


def clean_text(value: str) -> str:
    """Remove line breaks and collapse repeated whitespace."""
    return " ".join(value.split())


def iter_results(max_results: int) -> Iterator[arxiv.Result]:
    client = arxiv.Client()
    search = arxiv.Search(query=QUERY, max_results=max_results,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    yield from client.results(search)


def ingest(database_path: Path, max_results: int = 1000) -> int:
    """Fetch up to ``max_results`` and insert newly seen papers."""
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    initialize_database(database_path)
    inserted = 0
    insert_sql = """INSERT OR IGNORE INTO papers
        (arxiv_id, title, abstract, authors, categories, published, url)
        VALUES (?, ?, ?, ?, ?, ?, ?)"""

    try:
        with sqlite3.connect(database_path) as connection:
            for result in tqdm(iter_results(max_results), total=max_results,
                               desc="Fetching arXiv papers", unit="paper"):
                cursor = connection.execute(
                    insert_sql,
                    (
                        result.entry_id.rsplit("/", 1)[-1],
                        clean_text(result.title),
                        clean_text(result.summary),
                        ", ".join(author.name for author in result.authors),
                        ", ".join(result.categories),
                        result.published.isoformat(),
                        result.entry_id,
                    ),
                )
                inserted += cursor.rowcount
    except Exception:
        LOGGER.exception("Ingestion failed; successfully inserted %d papers", inserted)
        raise
    LOGGER.info("Inserted %d new papers into %s", inserted, database_path)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("papers.db"), help="SQLite database path")
    parser.add_argument("--max-results", type=int, default=1000, help="Maximum number of arXiv results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    ingest(args.db, args.max_results)


if __name__ == "__main__":
    main()
