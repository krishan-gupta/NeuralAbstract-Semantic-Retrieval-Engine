"""Start the real FastAPI server and smoke-test retrieval on the local corpus."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8765"
QUERIES = (
    "low rank adaptation",
    "medical",
    "LoRA",
    "making large models run on phones",
    "teaching AI to say it doesn't know",
    "reducing hallucinations in chatbots",
    "FlashAttention GPU optimization",
    "graph neural networks",
    "diffusion models image synthesis",
    "attention mechanism",
    "multimodal learning",
    "retrieval augmented generation",
    "learning from few examples",
    "molecular property prediction",
    "mixture of experts routing",
    "reinforcement learning from human feedback",
    "neural network compression",
    "vision transformer patch projection",
    "state space models",
    "self supervised representation learning",
)


def _request(
    opener: urllib.request.OpenerDirector, path: str, body: dict | None = None
) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with opener.open(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_until_ready(
    opener: urllib.request.OpenerDirector,
    process: subprocess.Popen,
    started: float,
    timeout: float = 300,
) -> float:
    while time.perf_counter() - started < timeout:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"Uvicorn exited before readiness with code {exit_code}")
        try:
            _request(opener, "/")
            return time.perf_counter() - started
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    raise TimeoutError(f"API did not become ready within {timeout:.0f} seconds")


def main() -> int:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    server_log = (log_dir / "phase3_api_server.log").open("w", encoding="utf-8")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
        ],
        cwd=ROOT,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    started = time.perf_counter()
    try:
        startup_seconds = _wait_until_ready(opener, process, started)
        print(f"backend_startup_seconds={startup_seconds:.2f}")

        embedders = _request(opener, "/embedders")["embedders"]
        print(f"available_embedders={embedders}")
        if not {"local", "bge"}.issubset(embedders):
            raise AssertionError("The API did not expose both local and BGE indexes")

        with sqlite3.connect(ROOT / "papers.db") as connection:
            doc_id, exact_title = connection.execute(
                "SELECT id, title FROM papers WHERE id = 1234"
            ).fetchone()

        first_results: dict[str, int] = {}
        for method in ("dense", "sparse", "hybrid"):
            response = _request(
                opener,
                "/search",
                {"query": "low rank adaptation", "method": method, "top_k": 10},
            )
            first_results[method] = len(response["results"])
        print(f"low_rank_adaptation_result_counts={first_results}")

        model_results = {
            tag: _request(
                opener,
                "/search",
                {
                    "query": "making large models run on phones",
                    "method": "dense",
                    "top_k": 20,
                    "embedder": tag,
                },
            )["results"]
            for tag in ("local", "bge")
        }
        model_ids = {
            tag: [int(paper["id"]) for paper in papers]
            for tag, papers in model_results.items()
        }
        models_differ = model_ids["local"] != model_ids["bge"]
        print(f"local_vs_bge_results_differ={models_differ}")
        if not models_differ:
            raise AssertionError("Local and BGE dense results unexpectedly match")

        exact = _request(
            opener,
            "/search",
            {"query": exact_title, "method": "sparse", "top_k": 10},
        )
        exact_passed = bool(exact["results"]) and int(exact["results"][0]["id"]) == int(
            doc_id
        )
        print(f"exact_title_first_result={exact_passed}; title={exact_title[:100]}")
        if not exact_passed:
            raise AssertionError(
                "Sparse exact-title retrieval did not return the source paper first"
            )

        nonsense = _request(
            opener,
            "/search",
            {"query": "qzxv wlkj", "method": "sparse", "top_k": 10},
        )
        print(f"nonsense_sparse_result_count={len(nonsense['results'])}")

        comparison = _request(
            opener,
            "/compare",
            {"query": "making large models run on phones", "top_k": 10},
        )
        comparison_passed = all(
            key in comparison for key in ("hybrid", "bm25", "overlap")
        )
        print(
            f"compare_passed={comparison_passed}; overlap="
            f"{comparison.get('overlap', {}).get('overlap_pct')}%"
        )
        if not comparison_passed:
            raise AssertionError("/compare response is missing required fields")

        bge_comparison = _request(
            opener,
            "/compare",
            {
                "query": "making large models run on phones",
                "top_k": 10,
                "embedder": "bge",
            },
        )
        bge_comparison_passed = all(
            key in bge_comparison for key in ("hybrid", "bm25", "overlap")
        )
        print(f"bge_compare_passed={bge_comparison_passed}")
        if not bge_comparison_passed:
            raise AssertionError("/compare did not accept embedder=bge")

        queries = _request(opener, "/benchmark/queries")["queries"]
        query_id = int(queries[0]["id"])
        built_pool = _request(
            opener,
            "/labels/pool/refresh",
            {"query_id": query_id},
        )
        first_pool_order = [paper["arxiv_id"] for paper in built_pool["papers"]]
        persisted_pool = _request(opener, f"/labels/pool?query_id={query_id}")
        stable_order = first_pool_order == [
            paper["arxiv_id"] for paper in persisted_pool["papers"]
        ]
        print(
            f"persistent_pool_passed={bool(first_pool_order) and stable_order}; "
            f"pool_count={len(first_pool_order)}; stable_order={stable_order}"
        )
        if not first_pool_order or not stable_order:
            raise AssertionError("Persistent labeling pool is empty or unstable")

        all_pools = _request(opener, "/labels/pool/refresh-all", {})
        pool_counts = [int(item["pool_count"]) for item in all_pools["pools"]]
        all_pools_passed = (
            all_pools["queries_processed"] == len(queries)
            and len(pool_counts) == len(queries)
            and all(count > 0 for count in pool_counts)
        )
        print(
            f"all_pools_passed={all_pools_passed}; "
            f"queries={all_pools['queries_processed']}; "
            f"pool_sizes={pool_counts}"
        )
        if not all_pools_passed:
            raise AssertionError("Could not build non-empty pools for every query")

        averages: dict[str, float] = {}
        for method in ("sparse", "dense", "hybrid"):
            durations = []
            for query in QUERIES:
                request_started = time.perf_counter()
                _request(
                    opener,
                    "/search",
                    {"query": query, "method": method, "top_k": 10},
                )
                durations.append((time.perf_counter() - request_started) * 1000)
            averages[method] = sum(durations) / len(durations)
        print(
            "average_latency_ms_over_20_queries="
            + json.dumps({key: round(value, 2) for key, value in averages.items()})
        )
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        server_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
