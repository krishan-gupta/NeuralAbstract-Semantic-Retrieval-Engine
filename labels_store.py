"""SQLite storage for human benchmark queries and judgments."""

from __future__ import annotations

import json
import random
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEED_QUERIES = (
    "making large models run on phones",
    "teaching AI to say it doesn't know",
    "making neural networks smaller without losing accuracy",
    "reducing hallucinations in chatbots",
    "learning from very few examples",
    "explaining why a model made a decision",
    "training models without labeled data",
    "AI that improves its own reasoning step by step",
    "LoRA",
    "FlashAttention",
    "Mamba state space model",
    "RLHF",
    "retrieval augmented generation",
    "mixture of experts routing",
    "diffusion models image synthesis",
    "graph neural networks molecular property prediction",
)


def connect_labels(
    path: str | Path = "labels.db", labels_json: str | Path | None = None
) -> sqlite3.Connection:
    """Open the labels database, creating its schema and seed queries as needed."""
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_existed = database_path.exists()
    connection = sqlite3.connect(database_path, timeout=10.0, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS benchmark_queries "
                "(id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS labels (
                    query_id INTEGER NOT NULL,
                    arxiv_id TEXT NOT NULL,
                    relevant INTEGER NOT NULL CHECK(relevant IN (0, 1)),
                    labeled_at TEXT NOT NULL,
                    PRIMARY KEY(query_id, arxiv_id)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS pool (
                    query_id INTEGER NOT NULL,
                    arxiv_id TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY(query_id, arxiv_id)
                )"""
            )
            pool_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(pool)")
            }
            if "position" not in pool_columns:
                connection.execute(
                    "ALTER TABLE pool ADD COLUMN position INTEGER NOT NULL DEFAULT 0"
                )
                connection.execute(
                    """UPDATE pool AS p SET position = (
                           SELECT COUNT(*) - 1 FROM pool AS earlier
                           WHERE earlier.query_id = p.query_id
                             AND earlier.arxiv_id <= p.arxiv_id
                       )"""
                )
            connection.executemany(
                "INSERT OR IGNORE INTO benchmark_queries(text) VALUES (?)",
                ((query,) for query in SEED_QUERIES),
            )
        if not database_existed:
            import_labels_json(
                connection,
                Path(labels_json)
                if labels_json
                else database_path.with_name("labels.json"),
            )
    except Exception:
        connection.close()
        raise
    return connection


def list_queries(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT q.id, q.text,
                  (SELECT COUNT(*) FROM labels l WHERE l.query_id = q.id) AS labeled_count,
                  (SELECT COALESCE(SUM(l.relevant), 0) FROM labels l
                   WHERE l.query_id = q.id) AS relevant_count,
                  (SELECT COUNT(*) FROM pool p WHERE p.query_id = q.id) AS pool_count
           FROM benchmark_queries q ORDER BY q.id"""
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_query(
    connection: sqlite3.Connection, text: str, query_id: int | None = None
) -> int:
    """Add a benchmark query or edit an existing query text."""
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("query text must not be empty")
    with connection:
        if query_id is None:
            connection.execute(
                "INSERT OR IGNORE INTO benchmark_queries(text) VALUES (?)",
                (normalized,),
            )
            row = connection.execute(
                "SELECT id FROM benchmark_queries WHERE text = ?", (normalized,)
            ).fetchone()
            return int(row[0])
        cursor = connection.execute(
            "UPDATE benchmark_queries SET text = ? WHERE id = ?", (normalized, query_id)
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Benchmark query {query_id} does not exist")
        return query_id


def list_labels(connection: sqlite3.Connection, query_id: int) -> dict[str, int]:
    rows = connection.execute(
        "SELECT arxiv_id, relevant FROM labels WHERE query_id = ?", (query_id,)
    ).fetchall()
    return {str(row["arxiv_id"]): int(row["relevant"]) for row in rows}


def list_pool(connection: sqlite3.Connection, query_id: int) -> list[str]:
    """Return a query's stored, stable pool order."""
    rows = connection.execute(
        "SELECT arxiv_id FROM pool WHERE query_id = ? ORDER BY position, arxiv_id",
        (query_id,),
    ).fetchall()
    return [str(row["arxiv_id"]) for row in rows]


def add_pool_candidates(
    connection: sqlite3.Connection, query_id: int, arxiv_ids: list[str]
) -> int:
    """Append newly discovered candidates without changing pool order or labels."""
    if (
        connection.execute(
            "SELECT 1 FROM benchmark_queries WHERE id = ?", (query_id,)
        ).fetchone()
        is None
    ):
        raise KeyError(f"Benchmark query {query_id} does not exist")

    existing = set(list_pool(connection, query_id))
    new_ids = list(dict.fromkeys(str(value) for value in arxiv_ids))
    new_ids = [value for value in new_ids if value not in existing]
    if not new_ids:
        return 0

    next_position = int(
        connection.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM pool WHERE query_id = ?",
            (query_id,),
        ).fetchone()[0]
    )
    random.Random(query_id).shuffle(new_ids)
    first_seen = datetime.now(UTC).isoformat()
    with connection:
        connection.executemany(
            """INSERT OR IGNORE INTO pool(query_id, arxiv_id, first_seen, position)
               VALUES (?, ?, ?, ?)""",
            (
                (query_id, arxiv_id, first_seen, next_position + offset)
                for offset, arxiv_id in enumerate(new_ids)
            ),
        )
    return len(new_ids)


def save_label(
    connection: sqlite3.Connection, query_id: int, arxiv_id: str, relevant: bool
) -> None:
    query = connection.execute(
        "SELECT 1 FROM benchmark_queries WHERE id = ?", (query_id,)
    ).fetchone()
    if query is None:
        raise KeyError(f"Benchmark query {query_id} does not exist")
    with connection:
        connection.execute(
            """INSERT INTO labels(query_id, arxiv_id, relevant, labeled_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(query_id, arxiv_id) DO UPDATE SET
                   relevant = excluded.relevant, labeled_at = excluded.labeled_at""",
            (query_id, arxiv_id, int(relevant), datetime.now(UTC).isoformat()),
        )


def delete_label(connection: sqlite3.Connection, query_id: int, arxiv_id: str) -> bool:
    """Remove a judgment when the human explicitly undoes their last action."""
    with connection:
        cursor = connection.execute(
            "DELETE FROM labels WHERE query_id = ? AND arxiv_id = ?",
            (query_id, arxiv_id),
        )
    return cursor.rowcount > 0


def export_labels(
    connection: sqlite3.Connection, output_path: str | Path = "labels.json"
) -> int:
    """Export human judgments keyed by query text and stable arXiv IDs."""
    queries = connection.execute(
        """SELECT q.text, l.arxiv_id, l.relevant, l.labeled_at
           FROM benchmark_queries q JOIN labels l ON q.id = l.query_id
           ORDER BY q.id, l.arxiv_id"""
    ).fetchall()
    payload: dict[str, list[dict[str, Any]]] = {}
    for row in queries:
        payload.setdefault(str(row["text"]), []).append(
            {
                "arxiv_id": row["arxiv_id"],
                "relevant": int(row["relevant"]),
                "labeled_at": row["labeled_at"],
            }
        )
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sum(len(values) for values in payload.values())


def import_labels_json(connection: sqlite3.Connection, input_path: Path) -> int:
    """Restore labels from the versionable JSON export if one is available."""
    if not input_path.is_file():
        return 0
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a query-to-labels object in {input_path}")
    restored = 0
    with connection:
        for query_text, judgments in payload.items():
            connection.execute(
                "INSERT OR IGNORE INTO benchmark_queries(text) VALUES (?)",
                (query_text,),
            )
            query = connection.execute(
                "SELECT id FROM benchmark_queries WHERE text = ?", (query_text,)
            ).fetchone()
            if not isinstance(judgments, list):
                continue
            for judgment in judgments:
                if not isinstance(judgment, dict) or "arxiv_id" not in judgment:
                    continue
                try:
                    relevant = int(judgment.get("relevant", -1))
                except (TypeError, ValueError):
                    continue
                if relevant not in (0, 1):
                    continue
                connection.execute(
                    """INSERT OR IGNORE INTO labels(query_id, arxiv_id, relevant, labeled_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        int(query[0]),
                        str(judgment["arxiv_id"]),
                        relevant,
                        str(
                            judgment.get("labeled_at") or datetime.now(UTC).isoformat()
                        ),
                    ),
                )
                restored += int(connection.execute("SELECT changes()").fetchone()[0])
    return restored


def close_labels(connection: sqlite3.Connection) -> None:
    """Close a labels connection (helper for lifespan cleanup)."""
    connection.close()


def export_from_file(
    database_path: Path = Path("labels.db"), output_path: Path = Path("labels.json")
) -> int:
    with closing(connect_labels(database_path)) as connection:
        return export_labels(connection, output_path)
