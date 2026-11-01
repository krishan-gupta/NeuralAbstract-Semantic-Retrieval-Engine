import tempfile
from pathlib import Path

from labels_store import (
    add_pool_candidates,
    connect_labels,
    export_labels,
    list_labels,
    list_pool,
    save_label,
)


def test_pool_order_and_labels_survive_database_reopen() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
        database = Path(folder) / "labels.db"
        connection = connect_labels(database)
        query_id = int(
            connection.execute(
                "SELECT id FROM benchmark_queries ORDER BY id LIMIT 1"
            ).fetchone()[0]
        )

        assert add_pool_candidates(connection, query_id, ["a", "b", "c"]) == 3
        original_order = list_pool(connection, query_id)
        save_label(connection, query_id, original_order[0], relevant=True)
        connection.close()

        reopened = connect_labels(database)
        assert list_pool(reopened, query_id) == original_order
        assert list_labels(reopened, query_id) == {original_order[0]: 1}

        assert add_pool_candidates(reopened, query_id, ["a", "b", "c", "d"]) == 1
        assert list_pool(reopened, query_id)[: len(original_order)] == original_order
        assert list_labels(reopened, query_id) == {original_order[0]: 1}
        export_path = Path(folder) / "labels.json"
        assert export_labels(reopened, export_path) == 1
        reopened.close()

        database.unlink()
        restored = connect_labels(database, labels_json=export_path)
        assert list_labels(restored, query_id) == {original_order[0]: 1}
        restored.close()
