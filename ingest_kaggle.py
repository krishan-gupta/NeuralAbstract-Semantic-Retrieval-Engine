"""Stream an arXiv Kaggle JSONL snapshot into the project's SQLite schema."""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

PUBLISHED_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"


def clean_text(value: Any) -> str:
    """Convert a metadata value to one whitespace-normalized string."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value, flags=re.UNICODE).strip()


def published_date(record: dict[str, Any]) -> str | None:
    """Return the first version's creation date, falling back to update_date."""
    versions = record.get("versions")
    if isinstance(versions, list) and versions and isinstance(versions[0], dict):
        created = versions[0].get("created")
        if isinstance(created, str):
            try:
                return (
                    datetime.strptime(created.strip(), PUBLISHED_FORMAT)
                    .date()
                    .isoformat()
                )
            except ValueError:
                pass
    update_date = record.get("update_date")
    if isinstance(update_date, str):
        try:
            return datetime.strptime(update_date[:10], "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None
    return None


def iter_matching_rows(
    input_path: Path,
    wanted_categories: set[str],
    min_year: int,
    counts: dict[str, int],
) -> Iterator[tuple[str, str, str, str, str, str, str]]:
    """Yield qualifying rows while reading the source strictly one line at a time."""
    with input_path.open("r", encoding="utf-8") as source:
        for line in tqdm(source, desc="Scanning Kaggle metadata", unit="line"):
            counts["lines_scanned"] += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict):
                continue

            raw_categories = record.get("categories", "")
            category_values = (
                set(raw_categories.split())
                if isinstance(raw_categories, str)
                else set()
            )
            if not category_values.intersection(wanted_categories):
                continue

            arxiv_id = clean_text(record.get("id"))
            title = clean_text(record.get("title"))
            abstract = clean_text(record.get("abstract"))
            authors = clean_text(record.get("authors"))[:500]
            date = published_date(record)
            if not arxiv_id or not title or not abstract or not date:
                continue
            try:
                if int(date[:4]) < min_year:
                    continue
            except ValueError:
                continue

            url = f"https://arxiv.org/abs/{arxiv_id}"
            yield (
                arxiv_id,
                title,
                abstract,
                authors,
                clean_text(raw_categories),
                date,
                url,
            )


def sample_rows(
    rows: Iterator[tuple[str, str, str, str, str, str, str]],
    max_papers: int,
    seed: int,
) -> tuple[list[tuple[str, str, str, str, str, str, str]], int]:
    """Use reservoir sampling so each matching record has equal selection chance."""
    rng = random.Random(seed)
    reservoir: list[tuple[str, str, str, str, str, str, str]] = []
    matched = 0
    for row in rows:
        matched += 1
        if len(reservoir) < max_papers:
            reservoir.append(row)
        else:
            slot = rng.randrange(matched)
            if slot < max_papers:
                reservoir[slot] = row

    # Defensively deduplicate selected records before INSERT OR IGNORE. This keeps
    # the fresh AUTOINCREMENT IDs contiguous even if the source repeats an arXiv ID.
    unique_rows = {row[0]: row for row in reservoir}
    return list(unique_rows.values()), matched


def ingest(
    input_path: Path,
    database_path: Path = Path("papers.db"),
    max_papers: int = 150_000,
    min_year: int = 2018,
    categories: set[str] | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Import a random sample; return (matching record count, inserted count)."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")
    if max_papers < 1:
        raise ValueError("--max-papers must be at least 1")
    if min_year < 1:
        raise ValueError("--min-year must be a positive year")
    if database_path.exists() and not overwrite:
        raise FileExistsError(
            f"Database already exists: {database_path}. Pass --overwrite to replace it."
        )

    if database_path.exists():
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"lines_scanned": 0}
    kept_rows, matched_count = sample_rows(
        iter_matching_rows(
            input_path, categories or {"cs.AI", "cs.LG", "cs.CL"}, min_year, counts
        ),
        max_papers=max_papers,
        seed=seed,
    )
    kept_rows.sort(key=lambda row: (row[5], row[0]))

    try:
        with closing(sqlite3.connect(database_path)) as connection, connection:
            connection.execute(
                """CREATE TABLE papers (
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
            connection.executemany(
                """INSERT OR IGNORE INTO papers
                       (arxiv_id, title, abstract, authors, categories, published, url)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                kept_rows,
            )
            inserted_count = int(
                connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            )
    except Exception:
        # Avoid leaving a database that looks complete after a failed import.
        database_path.unlink(missing_ok=True)
        raise

    print(
        f"\nIngestion complete: {counts['lines_scanned']:,} lines scanned; "
        f"{matched_count:,} records matched; {inserted_count:,} papers kept."
    )
    return matched_count, inserted_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Kaggle arXiv JSON-lines file"
    )
    parser.add_argument("--db", type=Path, default=Path("papers.db"))
    parser.add_argument("--max-papers", type=int, default=150_000)
    parser.add_argument("--min-year", type=int, default=2018)
    parser.add_argument("--categories", nargs="+", default=["cs.AI", "cs.LG", "cs.CL"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        ingest(
            input_path=args.input,
            database_path=args.db,
            max_papers=args.max_papers,
            min_year=args.min_year,
            categories=set(args.categories),
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
