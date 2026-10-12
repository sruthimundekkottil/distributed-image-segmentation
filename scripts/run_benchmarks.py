"""Run the Locust load test against several setups and write results/results.md.

Scenarios:
  * the synchronous baseline (POST /predict_sync, no queue, no workers)
  * the queue with 1, 2 and 4 workers (POST /jobs + polling)

Usage (Docker must be running; locust must be installed, see requirements-dev.txt):
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --users 16 --duration 60 --workers 1 2 4

The model is whatever MODEL_NAME docker compose uses (default mobilenet_v2).
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil  # installed with locust

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LOCUSTFILE = ROOT / "loadtest" / "locustfile.py"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def get_health(host: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=2) as response:
            return json.load(response)
    except OSError:
        return None


def start_stack(workers: int, host: str) -> dict:
    """(Re)start docker compose with the given number of workers and wait until ready."""
    run(["docker", "compose", "up", "-d", "--scale", f"worker={workers}"])
    deadline = time.time() + 180
    while time.time() < deadline:
        health = get_health(host)
        # Wait until exactly this many workers have registered with Redis.
        # (Workers only register after loading and warming up their model.)
        if health and health["status"] == "ok" and health["workers"] == workers:
            time.sleep(3)  # let everything settle
            return health
        time.sleep(1)
    raise RuntimeError(f"Stack with {workers} workers did not become healthy: {get_health(host)}")


def run_locust(user_class: str, users: int, duration: int, host: str, csv_prefix: Path) -> None:
    command = [
        sys.executable, "-m", "locust",
        "-f", str(LOCUSTFILE),
        "--headless",
        "--users", str(users),
        "--spawn-rate", str(users),  # start all users at once
        "--run-time", f"{duration}s",
        "--host", host,
        "--csv", str(csv_prefix),
        "--only-summary",
        user_class,
    ]  # fmt: skip
    print("$", " ".join(command), flush=True)
    # Locust exits with code 1 if any request failed (e.g. a 429); that is still a valid run.
    subprocess.run(command, cwd=ROOT, check=False)


def read_stats(csv_prefix: Path) -> dict[tuple[str, str], dict]:
    """Load Locust's <prefix>_stats.csv into {(type, name): row}."""
    with open(f"{csv_prefix}_stats.csv", newline="") as f:
        return {(row["Type"], row["Name"]): row for row in csv.DictReader(f)}


def cpu_name() -> str:
    """Human-readable CPU model (platform.processor() is vague on Windows)."""
    if platform.system() == "Windows":
        command = ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return result.stdout.strip() or platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def docker_cpus() -> str:
    result = subprocess.run(
        ["docker", "info", "--format", "{{.NCPU}}"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=16, help="concurrent Locust users")
    parser.add_argument("--duration", type=int, default=60, help="seconds per scenario")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--no-build", action="store_true", help="skip docker compose build")
    parser.add_argument("--skip-sync", action="store_true", help="skip the sync baseline")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "results.md")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    if not args.no_build:
        run(["docker", "compose", "build"])

    # (label, workers, locust user class, stats row type, stats row name)
    scenarios = []
    if not args.skip_sync:
        scenarios.append(
            ("Sync baseline (/predict_sync)", 0, "SyncUser", "POST", "POST /predict_sync")
        )
    scenarios += [
        (f"Queue + {n} worker{'s' if n > 1 else ''}", n, "AsyncJobUser", "JOB", "end_to_end")
        for n in args.workers
    ]

    rows = []
    model_name = "unknown"
    for label, workers, user_class, row_type, row_name in scenarios:
        print(f"\n=== {label} ===", flush=True)
        model_name = start_stack(workers, args.host)["model"]
        csv_prefix = RESULTS_DIR / f"locust_{user_class}_{workers}w"
        run_locust(user_class, args.users, args.duration, args.host, csv_prefix)

        stats = read_stats(csv_prefix)
        main_row = stats[(row_type, row_name)]
        rejected = stats.get(("POST", "POST /jobs"), {}).get("Failure Count", "0")
        rows.append(
            {
                "label": label,
                "workers": workers if workers else "- (in API process)",
                "completed": int(main_row["Request Count"]),
                "failed": int(main_row["Failure Count"]),
                "rejected": int(rejected),
                "throughput": float(main_row["Requests/s"]),
                "p50": main_row["50%"],
                "p95": main_row["95%"],
            }
        )
        print(rows[-1], flush=True)

    lines = [
        "# Benchmark results",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- **CPU cores: {psutil.cpu_count(logical=False)} physical / {os.cpu_count()} logical "
        f"on the host, {docker_cpus()} available to Docker**",
        f"- CPU: {cpu_name()} ({platform.system()} {platform.release()})",
        f"- Model: `{model_name}` on ONNX Runtime (CPU), "
        f"`ORT_NUM_THREADS={os.environ.get('ORT_NUM_THREADS', '1')}` per process, "
        f"`API_WORKERS={os.environ.get('API_WORKERS', '1')}` uvicorn process(es)",
        f"- Load: {args.users} concurrent Locust users, no think time, "
        f"{args.duration} s per scenario, same JPEG (loadtest/sample.jpg) every time",
        "- Locust, the API, Redis and the workers all ran on the same machine.",
        "",
        "| Scenario | Workers | Images completed | Throughput (images/s) "
        "| p50 latency (ms) | p95 latency (ms) | Failed | Rejected (429) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['workers']} | {r['completed']} | {r['throughput']:.1f} "
            f"| {r['p50']} | {r['p95']} | {r['failed']} | {r['rejected']} |"
        )
    lines += [
        "",
        "Latency for the queue scenarios is the full client-side round trip: upload, wait in "
        "the queue, inference, and polling `GET /jobs/{id}` every 20 ms until the result is ready.",
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
