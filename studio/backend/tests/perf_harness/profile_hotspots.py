# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""cProfile hotspot pass over the bench_api.py workload.

Boots the FastAPI app exactly the way ``bench_api.py`` does -- same throwaway
``UNSLOTH_STUDIO_HOME``, same in-process ``TestClient`` on
``http://127.0.0.1:5001``, same minted owner token -- then wraps N iterations of
the same ``WORKLOAD`` (imported from ``bench_api``, never re-typed) in
``cProfile`` and reports:

  * the top 30 functions by cumulative time, and
  * the top 15 functions by total (self) time,

with ``file:line`` for every entry. Per-endpoint wall medians are recorded too
so the profiled numbers can be tied back to the un-profiled harness baseline.

Stdlib only (``cProfile``/``pstats``); no third-party profiler dependencies.

Usage:
    python tests/perf_harness/profile_hotspots.py --iterations 10 --out hotspots.json
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import statistics
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

# Imported, not re-typed: the profiled workload must be the very list bench_api
# measures, so the two reports are directly comparable.
from bench_api import WORKLOAD  # noqa: E402  (path set above + script dir)


def _percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _stat_rows(stats: pstats.Stats, limit: int, *, sort_by: str) -> list[dict]:
    """pstats rows as dicts. sort_by is 'cumulative' (cumtime) or 'total' (tottime)."""
    rows = []
    for (filename, lineno, funcname), (
        ccalls,
        ncalls,
        tt,
        ct,
        _callers,
    ) in stats.stats.items():
        rows.append(
            {
                "function": funcname,
                "file": filename,
                "line": lineno,
                "primcalls": ccalls,
                "calls": ncalls,
                "tottime_ms": round(tt * 1000.0, 3),
                "cumtime_ms": round(ct * 1000.0, 3),
            }
        )
    key = "cumtime_ms" if sort_by == "cumulative" else "tottime_ms"
    rows.sort(key = lambda row: row[key], reverse = True)
    return rows[:limit]


def _print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    header = f"{'tottime_ms':>10} {'cumtime_ms':>10} {'calls':>10}  function (file:line)"
    print(header)
    print("-" * len(header))
    for row in rows:
        location = f"{row['file']}:{row['line']}"
        print(
            f"{row['tottime_ms']:>10.3f} {row['cumtime_ms']:>10.3f} "
            f"{str(row['calls']):>10}  {row['function']} ({location})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description = "cProfile the bench_api workload.")
    parser.add_argument("--iterations", type = int, default = 10)
    parser.add_argument("--warmup", type = int, default = 1)
    parser.add_argument("--out", type = str, default = "", help = "Write the report as JSON.")
    args = parser.parse_args()

    home = tempfile.mkdtemp(prefix = "unsloth-profile-home-")
    os.environ["UNSLOTH_STUDIO_HOME"] = home
    os.environ.setdefault("UNSLOTH_TELEMETRY", "0")
    os.environ.setdefault("UNSLOTH_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("ENVIRONMENT_TYPE", "production")

    from fastapi.testclient import TestClient  # noqa: PLC0415

    imported_at = time.perf_counter()
    import main as backend_main  # noqa: PLC0415
    import_seconds = time.perf_counter() - imported_at

    import auth.authentication as authentication  # noqa: PLC0415
    import auth.storage as auth_storage  # noqa: PLC0415

    with TestClient(backend_main.app, base_url = "http://127.0.0.1:5001") as client:
        secret = auth_storage.update_password("unsloth", "bench-password-123")
        token = authentication.create_access_token("unsloth", secret = secret)
        headers = {"Authorization": f"Bearer {token}"}

        # Same warm-up contract as bench_api: lazy imports, SQLite open and the
        # hardware snapshot stay off the clock. Kept short because cProfile adds
        # its own overhead and warm-up is not what this tool reports.
        for _ in range(args.warmup):
            for method, path in WORKLOAD:
                client.request(method, path, headers = headers)

        per_endpoint: dict[str, list[float]] = {path: [] for _method, path in WORKLOAD}
        statuses: dict[str, int] = {}
        profiler = cProfile.Profile()
        profiler.enable()
        loop_started = time.perf_counter()
        for _ in range(args.iterations):
            for method, path in WORKLOAD:
                started = time.perf_counter()
                response = client.request(method, path, headers = headers)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                per_endpoint[path].append(elapsed_ms)
                statuses[path] = response.status_code
        profiled_wall_s = time.perf_counter() - loop_started
        profiler.disable()

    stats = pstats.Stats(profiler)
    top_cumulative = _stat_rows(stats, 30, sort_by = "cumulative")
    top_total = _stat_rows(stats, 15, sort_by = "total")

    # Human-readable output first (stdout is the primary interface of this tool).
    print(f"import_seconds={import_seconds:.2f}  iterations={args.iterations}  "
          f"requests={args.iterations * len(WORKLOAD)}  "
          f"profiled_wall_s={profiled_wall_s:.2f}")
    print("\n=== per-endpoint wall ms under cProfile (median / p95) ===")
    for _method, path in WORKLOAD:
        samples = per_endpoint[path]
        print(f"{statistics.median(samples):>10.2f} {_percentile(samples, 0.95):>10.2f}  {path}")
    _print_rows("top 30 by cumulative time", top_cumulative)
    _print_rows("top 15 by total (self) time", top_total)

    if args.out:
        report = {
            "tool": "profile_hotspots",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "requests": args.iterations * len(WORKLOAD),
            "import_seconds": round(import_seconds, 2),
            "profiled_wall_s": round(profiled_wall_s, 2),
            "statuses": statuses,
            "per_endpoint_median_ms": {
                path: round(statistics.median(samples), 2)
                for path, samples in per_endpoint.items()
            },
            "per_endpoint_p95_ms": {
                path: round(_percentile(samples, 0.95), 2) for path, samples in per_endpoint.items()
            },
            "top_cumulative": top_cumulative,
            "top_total": top_total,
        }
        out = Path(args.out)
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_text(json.dumps(report, indent = 2) + "\n", encoding = "utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
