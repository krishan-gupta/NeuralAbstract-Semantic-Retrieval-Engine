"""Check ArXivLens Definition-of-Done items from local files and databases."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent


@dataclass
class Check:
    name: str
    status: str
    evidence: str


def _database(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve().as_posix()), safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def _papers_summary(path: Path) -> tuple[int, str, str, int, dict[str, int]]:
    with _database(path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count, MIN(published) AS oldest, "
            "MAX(published) AS newest FROM papers"
        ).fetchone()
        category_counts: dict[str, int] = {}
        for (categories,) in connection.execute("SELECT categories FROM papers"):
            for category in set(re.split(r"[,\s]+", str(categories or ""))):
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1
    count = int(row["count"])
    oldest = str(row["oldest"] or "unknown")
    newest = str(row["newest"] or "unknown")
    years = set()
    if count:
        with _database(path) as connection:
            years = {
                int(value[0])
                for value in connection.execute(
                    "SELECT DISTINCT substr(published, 1, 4) FROM papers"
                )
                if value[0] and str(value[0]).isdigit()
            }
    return count, oldest, newest, len(years), category_counts


def _corpus_checks(path: Path) -> tuple[Check, Check, int | None]:
    if not path.is_file():
        failed = Check("Corpus size", "FAILED", f"{path.name} is missing")
        return failed, Check("Corpus is real", "FAILED", "No paper database"), None
    try:
        count, oldest, newest, year_count, categories = _papers_summary(path)
    except (sqlite3.Error, OSError, ValueError) as exc:
        failed = Check("Corpus size", "FAILED", f"Cannot read papers table: {exc}")
        return failed, Check("Corpus is real", "FAILED", "Cannot inspect corpus"), None
    category_summary = (
        ", ".join(
            f"{name}={value:,}"
            for name, value in sorted(categories.items(), key=lambda item: -item[1])[:8]
        )
        or "none"
    )
    evidence = (
        f"{count:,} papers; published {oldest} to {newest}; "
        f"categories: {category_summary}"
    )
    size = Check("Corpus size", "DONE" if count >= 100_000 else "FAILED", evidence)
    years = Check(
        "Corpus is real",
        "DONE" if count >= 100_000 and year_count >= 3 else "FAILED",
        f"{year_count} distinct publication years; requires >= 3 years and >= 100,000 rows",
    )
    return size, years, count


def _dense_count(path: Path) -> tuple[int | None, str]:
    if not path.is_file():
        return None, f"{path.name} is missing"
    try:
        import faiss

        count = int(faiss.read_index(str(path)).ntotal)
        return count, f"{path.name}: {count:,} vectors"
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return None, f"Cannot read {path.name}: {exc}"


def _index_checks(row_count: int | None) -> list[Check]:
    checks: list[Check] = []
    for label, path in (
        ("Dense index (local)", ROOT / "faiss_local.index"),
        ("Dense index (bge)", ROOT / "faiss_bge.index"),
    ):
        count, evidence = _dense_count(path)
        passed = row_count is not None and count == row_count
        checks.append(Check(label, "DONE" if passed else "FAILED", evidence))

    mapping_path = ROOT / "bm25_index" / "doc_ids.npy"
    if not mapping_path.is_file():
        checks.append(Check("BM25 mapping", "FAILED", f"{mapping_path} is missing"))
    else:
        try:
            import numpy as np

            doc_ids = np.load(mapping_path, mmap_mode="r", allow_pickle=False)
            count = len(doc_ids)
            passed = row_count is not None and count == row_count
            checks.append(
                Check(
                    "BM25 mapping",
                    "DONE" if passed else "FAILED",
                    f"{count:,} doc IDs; database has {row_count if row_count is not None else 'unknown'} rows",
                )
            )
        except (ImportError, OSError, ValueError) as exc:
            checks.append(
                Check("BM25 mapping", "FAILED", f"Cannot read mapping: {exc}")
            )

    stats_path = ROOT / "index_stats.json"
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        missing = [
            tag
            for tag in ("local", "bge")
            if not isinstance(stats.get(tag), dict)
            or not stats[tag].get("build_seconds")
            or not stats[tag].get("faiss_size_bytes")
        ]
        if missing:
            checks.append(
                Check(
                    "Index stats",
                    "FAILED",
                    f"Missing build time/file size for: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                Check(
                    "Index stats",
                    "DONE",
                    "local and bge build time and FAISS file size are recorded",
                )
            )
    except (OSError, json.JSONDecodeError) as exc:
        checks.append(Check("Index stats", "FAILED", f"Cannot read stats: {exc}"))
    return checks


def _labels_check() -> tuple[Check, int, int, datetime | None]:
    path = ROOT / "labels.db"
    if not path.is_file():
        return (
            Check("Labels and pool coverage", "BLOCKED-HUMAN", "labels.db not created"),
            0,
            0,
            None,
        )
    try:
        with _database(path) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not {"benchmark_queries", "labels"}.issubset(tables):
                return (
                    Check(
                        "Labels and pool coverage",
                        "FAILED",
                        "Required label tables are missing",
                    ),
                    0,
                    0,
                    None,
                )
            relevant_rows = connection.execute(
                "SELECT q.id, COUNT(l.arxiv_id) AS judgments "
                "FROM benchmark_queries q JOIN labels l ON l.query_id=q.id "
                "WHERE l.relevant=1 GROUP BY q.id"
            ).fetchall()
            relevant_queries = len(relevant_rows)
            total = int(connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0])
            last_label = connection.execute(
                "SELECT MAX(labeled_at) FROM labels"
            ).fetchone()[0]
            pool_coverage = (
                "no labeled queries; pool schema ready"
                if "pool" in tables
                else "pool table missing"
            )
            if "pool" in tables and relevant_queries:
                missing = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM labels l LEFT JOIN pool p "
                        "ON p.query_id=l.query_id AND p.arxiv_id=l.arxiv_id "
                        "WHERE p.arxiv_id IS NULL"
                    ).fetchone()[0]
                )
                pool_coverage = (
                    "complete" if missing == 0 else f"{missing} labels outside pool"
                )
        if relevant_queries < 12:
            status = "BLOCKED-HUMAN"
        else:
            status = "DONE" if pool_coverage == "complete" else "FAILED"
        last_time = datetime.fromisoformat(str(last_label)) if last_label else None
        evidence = (
            f"{relevant_queries} queries with >=1 relevant; {total:,} total judgments; "
            f"pool coverage: {pool_coverage}"
        )
        return (
            Check("Labels and pool coverage", status, evidence),
            relevant_queries,
            total,
            last_time,
        )
    except (sqlite3.Error, OSError, ValueError) as exc:
        return (
            Check(
                "Labels and pool coverage", "FAILED", f"Cannot inspect labels: {exc}"
            ),
            0,
            0,
            None,
        )


def _label_artifact_checks(
    relevant_queries: int, last_label: datetime | None
) -> list[Check]:
    labels_path = ROOT / "labels.json"
    results_path = ROOT / "benchmark_results.json"
    if relevant_queries < 12 or last_label is None:
        labels_check = Check(
            "Labels export",
            "BLOCKED-HUMAN",
            "Needs human judgments on at least 12 queries before export",
        )
    elif not labels_path.is_file():
        labels_check = Check("Labels export", "FAILED", "labels.json is missing")
    else:
        label_time = (
            last_label.replace(tzinfo=UTC) if last_label.tzinfo is None else last_label
        )
        export_time = datetime.fromtimestamp(labels_path.stat().st_mtime, UTC)
        fresh = export_time > label_time
        labels_check = Check(
            "Labels export",
            "DONE" if fresh else "FAILED",
            f"labels.json {'is' if fresh else 'is not'} newer than the latest database label",
        )

    if relevant_queries < 12:
        results_check = Check(
            "Benchmark results",
            "BLOCKED-HUMAN",
            "Needs labels on at least 12 queries before calculating results",
        )
    elif not results_path.is_file() or not labels_path.is_file():
        results_check = Check(
            "Benchmark results", "FAILED", "Results or labels export is missing"
        )
    else:
        try:
            result = json.loads(results_path.read_text(encoding="utf-8"))
            fresh = results_path.stat().st_mtime > labels_path.stat().st_mtime
            covers = int(result.get("num_queries", 0)) >= 12
            passed = fresh and covers
            results_check = Check(
                "Benchmark results",
                "DONE" if passed else "FAILED",
                f"{result.get('num_queries', 0)} queries; "
                f"newer than labels export={fresh}; coverage >=12={covers}",
            )
        except (OSError, ValueError, TypeError) as exc:
            results_check = Check(
                "Benchmark results", "FAILED", f"Cannot validate results: {exc}"
            )
    return [labels_check, results_check]


def _readme_check(relevant_queries: int) -> Check:
    path = ROOT / "README.md"
    if not path.is_file():
        return Check("README results", "FAILED", "README.md is missing")
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start not in text or end not in text or text.index(start) > text.index(end):
        return Check("README results", "FAILED", "Marked RESULTS block is missing")
    block = text.split(start, 1)[1].split(end, 1)[0]
    if "pending" in block.lower():
        status = "BLOCKED-HUMAN" if relevant_queries < 12 else "FAILED"
        return Check(
            "README results",
            status,
            "Waiting for at least 12 positively labeled queries before generating results",
        )
    results_path = ROOT / "benchmark_results.json"
    if not results_path.is_file():
        return Check("README results", "FAILED", "No benchmark_results.json to match")
    try:
        result = json.loads(results_path.read_text(encoding="utf-8"))
        # The generated block must include every aggregate metric that it claims
        # to publish; the updater can add exact values in a machine-readable marker.
        marker = f"<!-- BENCHMARK_RESULTS_SHA256:{_sha256(results_path)} -->"
        passed = marker in block and int(result.get("num_queries", 0)) >= 12
        return Check(
            "README results",
            "DONE" if passed else "FAILED",
            "Generated results marker matches benchmark_results.json"
            if passed
            else "Results block does not carry a matching generated-results marker",
        )
    except (OSError, ValueError, TypeError) as exc:
        return Check(
            "README results", "FAILED", f"Cannot compare README/results: {exc}"
        )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _screenshot_check() -> Check:
    required = (
        ROOT / "docs" / "screenshots" / "overlap-banner.png",
        ROOT / "docs" / "screenshots" / "compare-mode.png",
        ROOT / "docs" / "screenshots" / "benchmark-tab.png",
    )
    missing = [
        path.relative_to(ROOT).as_posix() for path in required if not path.is_file()
    ]
    if not missing:
        return Check("Screenshots", "DONE", "All three required screenshots exist")
    return Check("Screenshots", "BLOCKED-HUMAN", "Missing: " + ", ".join(missing))


def _test_check() -> Check:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("Tests", "FAILED", f"pytest -q could not complete: {exc}")
    output = (result.stdout + "\n" + result.stderr).strip()
    summary = next(
        (
            line.strip()
            for line in output.splitlines()
            if " passed" in line or " failed" in line
        ),
        output[-500:] or "pytest produced no summary",
    )
    return Check(
        "Tests", "DONE" if result.returncode == 0 else "FAILED", f"pytest -q: {summary}"
    )


def collect_checks() -> list[Check]:
    papers_path = ROOT / "papers.db"
    corpus_size, corpus_real, row_count = _corpus_checks(papers_path)
    labels, relevant_queries, _, last_label = _labels_check()
    checks = [corpus_size, corpus_real]
    checks.extend(_index_checks(row_count))
    checks.append(labels)
    checks.extend(_label_artifact_checks(relevant_queries, last_label))
    checks.extend(
        [
            _readme_check(relevant_queries),
            _screenshot_check(),
            _test_check(),
        ]
    )
    return checks


def main() -> int:
    print("ArXivLens Definition-of-Done status (checked from local files/data)\n")
    checks = collect_checks()
    name_width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{check.name:<{name_width}} | {check.status:<14} | {check.evidence}")
    done = sum(check.status == "DONE" for check in checks)
    failed = sum(check.status == "FAILED" for check in checks)
    blocked = sum(check.status == "BLOCKED-HUMAN" for check in checks)
    print(
        f"\n{done} of {len(checks)} done; {failed} failed; "
        f"{blocked} blocked on human input."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
