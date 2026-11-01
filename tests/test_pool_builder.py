import tempfile
from pathlib import Path

from labels_store import connect_labels, list_labels, list_pool, save_label
from pool_builder import build_query_pool, check_pool_coverage, read_query_pool


class FakeSearchEngine:
    def __init__(self) -> None:
        self.dense_indexes = {"local": object()}

    def search_sparse(self, query: str, top_k: int) -> list[int]:
        return [1, 2][:top_k]

    def search_dense(
        self, query: str, top_k: int, embedder: str = "local"
    ) -> list[int]:
        return ([1, 2] if embedder == "local" else [3, 4])[:top_k]

    def search_hybrid(
        self, query: str, top_k: int, embedder: str = "local"
    ) -> list[int]:
        return ([2, 1] if embedder == "local" else [4, 3])[:top_k]

    @staticmethod
    def get_metadata(ids: list[int]) -> list[dict]:
        return [{"id": value, "arxiv_id": f"paper-{value}"} for value in ids]

    @staticmethod
    def get_metadata_by_arxiv_ids(arxiv_ids: list[str]) -> list[dict]:
        return [
            {"id": int(arxiv_id.removeprefix("paper-")), "arxiv_id": arxiv_id}
            for arxiv_id in arxiv_ids
        ]


def test_new_embedder_extends_pool_and_preserves_existing_judgments() -> None:
    # This fake label is isolated in pytest's temporary database, never labels.db.
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
        database = Path(folder) / "scratch_labels.db"
        connection = connect_labels(database)
        query_id = int(
            connection.execute(
                "SELECT id FROM benchmark_queries ORDER BY id LIMIT 1"
            ).fetchone()[0]
        )
        engine = FakeSearchEngine()

        _, first_added = build_query_pool(engine, connection, query_id)
        first_order = list_pool(connection, query_id)
        assert first_added == 2
        assert len(first_order) == 2
        save_label(connection, query_id, first_order[0], relevant=True)
        connection.close()

        # Reopening verifies the displayed order and scratch judgment persist.
        connection = connect_labels(database)
        assert list_pool(connection, query_id) == first_order
        assert list_labels(connection, query_id) == {first_order[0]: 1}

        engine.dense_indexes["test"] = object()
        _, added = build_query_pool(engine, connection, query_id)
        refreshed = read_query_pool(engine, connection, query_id)

        assert added == 2
        assert list_pool(connection, query_id)[:2] == first_order
        assert set(list_pool(connection, query_id)) == {
            "paper-1",
            "paper-2",
            "paper-3",
            "paper-4",
        }
        assert refreshed["labeled_count"] == 1
        assert list_labels(connection, query_id) == {first_order[0]: 1}
        connection.close()


def test_pool_coverage_reports_top_ten_results_outside_pool() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
        connection = connect_labels(Path(folder) / "scratch_labels.db")
        query_id = int(
            connection.execute(
                "SELECT id FROM benchmark_queries ORDER BY id LIMIT 1"
            ).fetchone()[0]
        )
        connection.execute(
            "INSERT INTO pool(query_id, arxiv_id, first_seen, position) "
            "VALUES (?, ?, 'test', 0)",
            (query_id, "paper-1"),
        )
        save_label(connection, query_id, "paper-1", relevant=True)

        coverage = check_pool_coverage(FakeSearchEngine(), connection)

        assert coverage["status"] == "out_of_date"
        assert coverage["out_of_pool_count"] == 3
        assert coverage["warnings"]
        connection.close()
