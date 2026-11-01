import tempfile
from pathlib import Path

from benchmark_runner import run_labeled_benchmark
from labels_store import add_pool_candidates, connect_labels, save_label, upsert_query


class FixtureEngine:
    def __init__(self) -> None:
        self.dense_indexes = {"local": object()}

    @staticmethod
    def search_sparse(query: str, top_k: int) -> list[int]:
        ranking = {
            "fixture win": [1, 3, 2],
            "fixture tie": [1, 2, 3],
            "fixture loss": [1, 2, 3],
        }
        return ranking[query][:top_k]

    @staticmethod
    def search_dense(query: str, top_k: int, embedder: str = "local") -> list[int]:
        return [1, 2, 3][:top_k]

    @staticmethod
    def search_hybrid(query: str, top_k: int, embedder: str = "local") -> list[int]:
        ranking = {
            "fixture win": [1, 2, 3],
            "fixture tie": [1, 2, 3],
            "fixture loss": [2, 3, 1],
        }
        return ranking[query][:top_k]

    @staticmethod
    def get_metadata(ids: list[int]) -> list[dict]:
        return [{"id": value, "arxiv_id": f"paper-{value}"} for value in ids]


def _metric_projection(result: dict) -> tuple:
    methods = {
        name: {key: value for key, value in scores.items() if key != "latency_ms"}
        for name, scores in result["methods"].items()
    }
    per_query = [
        {
            name: {key: value for key, value in scores.items() if key != "latency_ms"}
            for name, scores in item["methods"].items()
        }
        for item in result["per_query"]
    ]
    return (
        result["num_queries"],
        result["num_judgments"],
        methods,
        result["lift_hybrid_vs_bm25"],
        result["ndcg10_comparison"],
        per_query,
        [item["ndcg10_comparison"] for item in result["per_query"]],
    )


def test_cli_and_api_benchmark_path_produce_same_metrics_and_ci() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
        connection = connect_labels(Path(folder) / "scratch_labels.db")
        engine = FixtureEngine()
        for query in ("fixture win", "fixture tie", "fixture loss"):
            query_id = upsert_query(connection, query)
            add_pool_candidates(connection, query_id, ["paper-1", "paper-2", "paper-3"])
            save_label(connection, query_id, "paper-1", relevant=True)
            save_label(connection, query_id, "paper-2", relevant=True)

        # The CLI and /benchmark/run both call this shared runner.
        api_result = run_labeled_benchmark(engine, connection, [5, 10])
        cli_result = run_labeled_benchmark(engine, connection, [5, 10])

        assert _metric_projection(api_result) == _metric_projection(cli_result)
        assert api_result["ndcg10_comparison"]["hybrid_wins"] == 1
        assert api_result["ndcg10_comparison"]["ties"] == 1
        assert api_result["ndcg10_comparison"]["bm25_wins"] == 1
        assert api_result["ndcg10_comparison"]["confidence_interval_95"] is not None
        assert api_result["pool_coverage"]["status"] == "up_to_date"
        connection.close()
