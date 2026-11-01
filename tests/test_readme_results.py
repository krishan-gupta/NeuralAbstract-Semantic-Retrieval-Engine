import json
import tempfile
from pathlib import Path

from scripts.update_readme_results import PENDING, update_readme


def test_readme_results_are_pending_without_benchmark_artifact() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
        scratch = Path(folder)
        readme = scratch / "README.md"
        readme.write_text("# Demo\n\n## Results\n", encoding="utf-8")

        block = update_readme(readme, scratch / "benchmark_results.json")

        assert block == PENDING
        assert (
            f"<!-- RESULTS:START -->\n{PENDING}\n<!-- RESULTS:END -->"
            in readme.read_text(encoding="utf-8")
        )


def test_readme_result_values_and_hash_come_from_json() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
        scratch = Path(folder)
        readme = scratch / "README.md"
        results = scratch / "benchmark_results.json"
        readme.write_text(
            "## Results\n\n<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n",
            encoding="utf-8",
        )
        result = {
            "status": "ok",
            "num_queries": 12,
            "num_judgments": 120,
            "methods": {
                name: {
                    "recall@5": 0.5,
                    "recall@10": 0.75,
                    "ndcg@5": 0.4,
                    "ndcg@10": 0.6,
                    "latency_ms": 12.0,
                    "index_size_bytes": 1024,
                }
                for name in ("bm25", "dense:local", "hybrid:local")
            },
            "lift_hybrid_vs_bm25": {
                "by_embedder": {"hybrid:local": {"recall@10": 10.0}}
            },
            "ndcg10_comparison": {
                "mean_difference": 0.1,
                "confidence_interval_95": [0.02, 0.18],
                "hybrid_wins": 8,
                "ties": 2,
                "bm25_wins": 2,
            },
        }
        results.write_text(json.dumps(result), encoding="utf-8")

        block = update_readme(readme, results)

        assert "75.00%" in block
        assert "12 queries, 120 judgments, 1 labeler, pooled judgments" in block
        assert "8/2/2" in block
        assert "<!-- BENCHMARK_RESULTS_SHA256:" in block
