# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Critical-path API benchmark harness.

Measures in-process HTTP latency for the read-only requests the Studio UI
issues on its critical path (app boot -> first screen -> polling). One process,
one warm app, a fixed order of a fixed set of endpoints, so a before/after run
of the same script on the same machine differs only in the code under test.

The app is served through ``TestClient`` against the app object itself: no
sockets, no browser, no model weights. ``base_url`` carries a literal loopback
address so ``scope["server"]`` looks like a real uvicorn loopback bind rather
than TestClient's ``testserver`` hostname; the two take different branches in
``utils.host_policy`` and only the former is what a user runs.

State is isolated in a throwaway ``UNSLOTH_STUDIO_HOME``, so a run cannot read or
write the developer's real Studio database.

Usage:
    python tests/perf_harness/bench_api.py --iterations 40 --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

# A fixed workload: the reads the shell performs while it boots and while it sits
# open polling. Ordered as the UI issues them, so a warm-cache effect that only
# the second request sees cannot be hidden inside the median.
WORKLOAD = [
    ("GET", "/api/health"),
    ("GET", "/api/auth/status"),
    ("GET", "/api/liveness"),
    ("GET", "/api/settings/hub"),
    ("GET", "/api/settings/caches"),
    ("GET", "/api/settings/multi-model"),
    ("GET", "/api/settings/remote-access"),
    ("GET", "/api/llama/architectures"),
    ("GET", "/api/llama/backend"),
    ("GET", "/api/models/local"),
    ("GET", "/api/inference/status"),
    ("GET", "/api/sandbox/capability"),
    ("GET", "/api/profile/stats"),
]


def _percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type = int, default = 40)
    parser.add_argument("--warmup", type = int, default = 5)
    parser.add_argument("--out", type = str, default = "")
    parser.add_argument("--label", type = str, default = "")
    args = parser.parse_args()

    home = tempfile.mkdtemp(prefix = "unsloth-bench-home-")
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
        # The throwaway home mints an owner whose bootstrap password still has to
        # be changed, and every authenticated route refuses until it is. Clearing
        # that is setup, not workload: it is what the user does once on first run.
        secret = auth_storage.update_password("unsloth", "bench-password-123")
        token = authentication.create_access_token("unsloth", secret = secret)
        headers = {"Authorization": f"Bearer {token}"}

        # Warm-up: pay one-time lazy imports, SQLite open, and the hardware
        # snapshot off the clock. Neither is what this benchmark measures.
        for _ in range(args.warmup):
            for method, path in WORKLOAD:
                client.request(method, path, headers = headers)

        per_endpoint: dict[str, list[float]] = {path: [] for _method, path in WORKLOAD}
        statuses: dict[str, int] = {}
        all_samples: list[float] = []
        for _ in range(args.iterations):
            for method, path in WORKLOAD:
                started = time.perf_counter()
                response = client.request(method, path, headers = headers)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                per_endpoint[path].append(elapsed_ms)
                all_samples.append(elapsed_ms)
                statuses[path] = response.status_code

        report = {
            "label": args.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
            "python": platform.python_version(),
            "import_seconds": round(import_seconds, 2),
            "iterations": args.iterations,
            "requests": len(all_samples),
            "median_ms": round(statistics.median(all_samples), 2),
            "p95_ms": round(_percentile(all_samples, 0.95), 2),
            "mean_ms": round(statistics.fmean(all_samples), 2),
            "max_ms": round(max(all_samples), 2),
            "statuses": statuses,
            "per_endpoint_median_ms": {
                path: round(statistics.median(samples), 2)
                for path, samples in per_endpoint.items()
            },
            "per_endpoint_p95_ms": {
                path: round(_percentile(samples, 0.95), 2) for path, samples in per_endpoint.items()
            },
        }

    print(json.dumps(report, indent = 2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_text(json.dumps(report, indent = 2) + "\n", encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
