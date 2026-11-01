"""Verify pool and judgment persistence across real API process restarts.

All test judgments use a temporary labels database under logs/ and are deleted
when the script exits. The project's labels.db and human labels.json are untouched.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8766"


def _request(path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _start(database: Path, log_path: Path) -> tuple[subprocess.Popen, TextIO]:
    log_handle = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["LABELS_DB"] = str(database)
    environment["LABELS_EXPORT"] = str(database.with_name("labels_export.json"))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8766",
        ],
        cwd=ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    )
    started = time.perf_counter()
    while time.perf_counter() - started < 180:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"API exited with code {process.returncode}")
        try:
            _request("/")
            print(f"backend_startup_seconds={time.perf_counter() - started:.2f}")
            return process, log_handle
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    process.terminate()
    log_handle.close()
    raise TimeoutError("API did not become ready")


def _stop(process: subprocess.Popen, log_handle: TextIO) -> None:
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log_handle.close()


def main() -> int:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pool-lifecycle-", dir=log_dir) as folder:
        scratch = Path(folder)
        database = scratch / "labels_smoke.db"
        process, log_handle = _start(database, log_dir / "phase5_pool_first_start.log")
        try:
            query = _request("/benchmark/queries")["queries"][0]
            query_id = int(query["id"])
            first = _request("/labels/pool/refresh", {"query_id": query_id})
            if len(first["papers"]) < 2:
                raise AssertionError("Expected at least two pool candidates")
            for paper, relevant in zip(first["papers"][:2], (True, False)):
                _request(
                    "/labels",
                    {
                        "query_id": query_id,
                        "arxiv_id": paper["arxiv_id"],
                        "relevant": relevant,
                    },
                )
            expected_order = [paper["arxiv_id"] for paper in first["papers"]]
            expected_labels = _request(f"/labels?query_id={query_id}")["labels"]
            if len(expected_labels) != 2:
                raise AssertionError("Scratch API did not save two test judgments")
            export = _request("/labels/export", {})
            if export["count"] != 2:
                raise AssertionError(
                    "Scratch label export did not include both judgments"
                )
        finally:
            _stop(process, log_handle)

        process, log_handle = _start(database, log_dir / "phase5_pool_restart.log")
        try:
            restored = _request(f"/labels/pool?query_id={query_id}")
            actual_order = [paper["arxiv_id"] for paper in restored["papers"]]
            actual_labels = _request(f"/labels?query_id={query_id}")["labels"]
            passed = actual_order == expected_order and actual_labels == expected_labels
            print(
                f"pool_lifecycle_passed={passed}; pool_count={len(actual_order)}; "
                f"judgments_restored={len(actual_labels)}; "
                f"export_timestamp={export['exported_at']}; scratch_only=True"
            )
            if not passed:
                raise AssertionError(
                    "Pool order or scratch judgments changed on restart"
                )
        finally:
            _stop(process, log_handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
