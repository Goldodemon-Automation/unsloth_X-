# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-middleware cost on a cold-but-warm-routed GET.

Boots the real app once, then, for each registered middleware, rebuilds the
middleware stack without that one and re-measures. The difference against the
full stack is that middleware's share of the request. Only layers that keep the
probe answering 200 are counted; a layer whose removal changes the reply is
reported as such rather than compared.

Run: python tests/perf_harness/measure_middleware_cost.py
"""

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

os.environ["UNSLOTH_STUDIO_HOME"] = tempfile.mkdtemp(prefix = "unsloth-mw-home-")

from fastapi.testclient import TestClient  # noqa: E402

import auth.authentication as authentication  # noqa: E402
import auth.storage as auth_storage  # noqa: E402
import main as backend_main  # noqa: E402

PROBE = "/api/liveness"
SAMPLES = 25


def _measure(client, headers) -> float:
    for _ in range(5):
        client.get(PROBE, headers = headers)
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter()
        client.get(PROBE, headers = headers)
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    app = backend_main.app
    with TestClient(app, base_url = "http://127.0.0.1:5001") as client:
        secret = auth_storage.update_password("unsloth", "mw-password-123")
        token = authentication.create_access_token("unsloth", secret = secret)
        headers = {"Authorization": f"Bearer {token}"}

        baseline = _measure(client, headers)
        print(f"full stack            {baseline:7.2f}ms")

        original = list(app.user_middleware)
        for index, entry in enumerate(original):
            name = getattr(entry.cls, "__name__", str(entry.cls))
            app.user_middleware = original[:index] + original[index + 1 :]
            app.middleware_stack = app.build_middleware_stack()
            try:
                response = client.get(PROBE, headers = headers)
                if response.status_code != 200:
                    print(f"without {name:26s} status={response.status_code} (not comparable)")
                    continue
                without = _measure(client, headers)
                print(f"without {name:26s} {without:7.2f}ms  saved={baseline - without:6.2f}ms")
            finally:
                app.user_middleware = original
                app.middleware_stack = app.build_middleware_stack()


if __name__ == "__main__":
    main()
