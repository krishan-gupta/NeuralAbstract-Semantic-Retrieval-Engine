"""Start the local API and Vite UI, wait for readiness, then open both URLs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def http_is_ready(url: str) -> bool:
    """Check a local service without routing the request through a system proxy."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=2) as response:
            return 200 <= response.status < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_services(
    processes: list[tuple[subprocess.Popen, str]], timeout: int
) -> bool:
    """Wait for both services together and fail immediately if either exits."""
    started = time.monotonic()
    checks = {
        "Backend API": "http://127.0.0.1:8000/",
        "Frontend UI": "http://127.0.0.1:5174/",
    }
    ready: set[str] = set()
    while time.monotonic() - started < timeout:
        for process, name in processes:
            exit_code = process.poll()
            if exit_code is not None:
                print(f"{name} exited during startup (exit code {exit_code}).")
                return False
            if name in checks and name not in ready and http_is_ready(checks[name]):
                ready.add(name)
                print(f"{name} is ready: {checks[name]}")
        if len(ready) == len(checks):
            return True
        time.sleep(0.5)
    missing = ", ".join(sorted(set(checks) - ready))
    print(f"Timed out waiting for: {missing} (after {timeout} seconds).")
    return False


def stop_process(process: subprocess.Popen, name: str) -> None:
    """Stop a child and wait for it to exit, escalating only if necessary."""
    if process.poll() is not None:
        return
    print(f"Stopping {name}...")
    if os.name == "nt":
        # npm.cmd starts Node as a child of cmd.exe; terminate the process tree
        # so a double-clicked launcher does not leave Vite running in background.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    venv_python = root_dir / ".venv" / "Scripts" / "python.exe"
    python_bin = str(venv_python) if venv_python.is_file() else sys.executable
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    timeout = int(os.getenv("ARXIVLENS_STARTUP_TIMEOUT", "180"))
    processes: list[tuple[subprocess.Popen, str]] = []

    print("=" * 60)
    print("  Starting ArXivLens")
    print("=" * 60)
    try:
        print("\nLaunching FastAPI on http://127.0.0.1:8000 ...")
        backend = subprocess.Popen(
            [
                python_bin,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=root_dir,
        )
        processes.append((backend, "Backend API"))

        print("Launching Vite UI on http://localhost:5174 ...")
        frontend_args = [
            npm_cmd,
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            "5174",
            "--strictPort",
        ]
        if os.name == "nt":
            # Windows cannot launch .cmd files directly with CreateProcess.
            frontend = subprocess.Popen(
                [
                    os.environ.get("COMSPEC", "cmd.exe"),
                    "/d",
                    "/s",
                    "/c",
                    subprocess.list2cmdline(frontend_args),
                ],
                cwd=root_dir,
            )
        else:
            frontend = subprocess.Popen(frontend_args, cwd=root_dir)
        processes.append((frontend, "Frontend UI"))

        print(
            f"\nWaiting up to {timeout} seconds for model loading and service startup..."
        )
        if not wait_for_services(processes, timeout):
            print(
                "\nStartup failed. Review the service errors above, fix the issue, and run start.bat again."
            )
            return 1

        frontend_url = "http://localhost:5174"
        docs_url = "http://127.0.0.1:8000/docs"
        print("\nBoth services are ready:")
        print(f"  Web UI:   {frontend_url}")
        print(f"  API Docs: {docs_url}")
        webbrowser.open(frontend_url)
        webbrowser.open(docs_url)
        print(
            "\nLeave this terminal open while using ArXivLens. Press Ctrl+C to stop both services."
        )

        while True:
            time.sleep(0.5)
            for process, name in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"{name} stopped (exit code {exit_code}).")
                    return 1
    except KeyboardInterrupt:
        print("\nShutting down ArXivLens...")
        return 0
    except FileNotFoundError as exc:
        print(f"Could not start a required program: {exc}", file=sys.stderr)
        return 1
    finally:
        for process, name in reversed(processes):
            stop_process(process, name)


if __name__ == "__main__":
    raise SystemExit(main())
