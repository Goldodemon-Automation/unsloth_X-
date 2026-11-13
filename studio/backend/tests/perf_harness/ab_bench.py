# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Before/after for the optimisations, measured in ONE process, back to back.

The tree-wide before/after from `benchmarks/baseline-api.json` cannot be reproduced exactly once
the source has moved, and the machine load drifts between runs (the independent benchmarker
measured up to +149% drift). This harness removes both problems: it measures the same running app
twice, alternating between the optimised code path and the pre-optimisation behaviour, by turning
each optimiser off through its own constant. Same process, same warm caches, same load, interleaved
A/B/B/A rounds, so load drift hits both arms.

Turned off for the "before" arm:
  utils.host_policy._RESOLUTION_TTL_SECONDS = 0   (resolver consulted per call, as before)
  routes.models._COMPAT_LOCAL_INVENTORY_MEMO_TTL_SECONDS = 0
  utils.cache_inventory's signature-check window, when it exposes one

Usage: python tests/perf_harness/ab_bench.py --iterations 12
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

os.environ["UNSLOTH_STUDIO_HOME"] = tempfile.mkdtemp(prefix = "unsloth-ab-home-")

from fastapi.testclient import TestClient  # noqa: E402

import auth.authentication as authentication  # noqa: E402
import auth.storage as auth_storage  # noqa: E402
import main as backend_main  # noqa: E402
import routes.models as models_route  # noqa: E402
import utils.host_policy as host_policy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_api import WORKLOAD, _percentile  # noqa: E402


def _cache_inventory_knobs():
    """The signature-check window, if the module exposes one, and its shipped value."""
    try:
        import utils.cache_inventory as cache_inventory  # noqa: PLC0415

        return cache_inventory, getattr(cache_inventory, "_SIGNATURE_RECHECK_TTL_SECONDS", None)
    except Exception:  # noqa: BLE001
        return None, None


def _set_optimisers(enabled: bool) -> None:
    host_policy._RESOLUTION_TTL_SECONDS = 60.0 if enabled else 0.0
    host_policy.clear_resolution_cache()
    models_route._COMPAT_LOCAL_INVENTORY_MEMO_TTL_SECONDS = 3.0 if enabled else 0.0
    models_route._compat_local_inventory_memo_clear()
    module, shipped = _cache_inventory_knobs()
    if module is not None and shipped is not None:
        module._SIGNATURE_RECHECK_TTL_SECONDS = shipped if enabled else 0.0


def _measure(client, headers, iterations: int, scope_host: str | None) -> dict:
    per_endpoint: dict[str, list[float]] = {path: [] for _method, path in WORKLOAD}
    samples: list[float] = []
    for _ in range(iterations):
        for method, path in WORKLOAD:
            started = time.perf_counter()
            client.request(method, path, headers = headers)
            elapsed = (time.perf_counter() - started) * 1000.0
            per_endpoint[path].append(elapsed)
            samples.append(elapsed)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "mean_ms": statistics.fmean(samples),
        "per_endpoint_median_ms": {
            path: statistics.median(values) for path, values in per_endpoint.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type = int, default = 12)
    parser.add_argument("--rounds", type = int, default = 2)
    parser.add_argument("--out", type = str, default = "")
    args = parser.parse_args()

    with TestClient(backend_main.app, base_url = "http://127.0.0.1:5001") as client:
        secret = auth_storage.update_password("unsloth", "ab-password-123")
        token = authentication.create_access_token("unsloth", secret = secret)
        headers = {"Authorization": f"Bearer {token}"}

        for _ in range(4):  # warm the app once, outside both arms
            for method, path in WORKLOAD:
                client.request(method, path, headers = headers)

        arms: dict[str, list[dict]] = {"before": [], "after": []}
        for round_index in range(args.rounds):
            # Alternate the order each round so a monotonic drift cannot favour one arm.
            order = ("before", "after") if round_index % 2 == 0 else ("after", "before")
            for arm in order:
                _set_optimisers(arm == "after")
                arms[arm].append(_measure(client, headers, args.iterations, None))

    def combine(runs: list[dict]) -> dict:
        merged: list[float] = []
        endpoint_values: dict[str, list[float]] = {}
        for run in runs:
            merged.extend([run["median_ms"]] * 1)
            for path, value in run["per_endpoint_median_ms"].items():
                endpoint_values.setdefault(path, []).append(value)
        return {
            "median_ms": round(statistics.median([r["median_ms"] for r in runs]), 2),
            "p95_ms": round(statistics.median([r["p95_ms"] for r in runs]), 2),
            "mean_ms": round(statistics.median([r["mean_ms"] for r in runs]), 2),
            "per_endpoint_median_ms": {
                path: round(statistics.median(values), 2)
                for path, values in endpoint_values.items()
            },
        }

    before = combine(arms["before"])
    after = combine(arms["after"])
    report = {
        "iterations_per_arm": args.iterations,
        "rounds": args.rounds,
        "before": before,
        "after": after,
        "p95_improvement_pct": round((1 - after["p95_ms"] / before["p95_ms"]) * 100, 1),
        "median_improvement_pct": round(
            (1 - after["median_ms"] / before["median_ms"]) * 100, 1
        ),
        "mean_improvement_pct": round((1 - after["mean_ms"] / before["mean_ms"]) * 100, 1),
        "poll_cycle_before_ms": round(sum(before["per_endpoint_median_ms"].values()), 2),
        "poll_cycle_after_ms": round(sum(after["per_endpoint_median_ms"].values()), 2),
    }
    report["poll_cycle_improvement_pct"] = round(
        (1 - report["poll_cycle_after_ms"] / report["poll_cycle_before_ms"]) * 100, 1
    )
    print(json.dumps(report, indent = 2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_text(json.dumps(report, indent = 2) + "\n", encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
