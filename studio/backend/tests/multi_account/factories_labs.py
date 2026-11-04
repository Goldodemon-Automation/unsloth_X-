# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Labs records: benchmarks and saved pipeline plans.

Both deletes are owner-only, so the isolation contract has a different shape from
the storage-backed routes: a managed account is refused with 403 before its own
record is ever looked up, while the owner reaches the delete and finds nothing at
its own path for a record seeded in someone else's workspace.
"""

import json

from .factory_base import Factory, seeder

BENCHMARK_ID = "0123456789abcdef"
PIPELINE_ID = "matrixpipeline01"


def _enable_labs(account) -> None:
    """Labs is off by default, and its switch lives in the owner's settings.

    The matrix re-seeds every workspace per test, so this cannot leak into
    another case; without it every Labs route answers 403 before the account
    lookup this table is here to check.
    """
    from core import labs as labs_module
    from utils.account_context import OWNER, run_as

    # read_state() always reads the owner's settings, so the write must land there
    # too, whichever account this seed is creating resources for.
    run_as(OWNER, labs_module.set_state, enabled = True)


@seeder("labs-benchmark")
def seed_labs_benchmark(account) -> dict[str, str]:
    """One stored benchmark record in the account's own outputs folder."""
    from core.labs import benchmarks
    from utils.account_context import run_as

    _enable_labs(account)
    path = run_as(account, benchmarks.records_root) / f"{BENCHMARK_ID}.json"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps(
            {
                "id": BENCHMARK_ID,
                "job_id": "matrix",
                "artifact": "matrix",
                "baseline": "matrix",
                "report": {},
                "suites": [],
                "created_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding = "utf-8",
    )
    return {"record_id": BENCHMARK_ID}


@seeder("labs-pipeline")
def seed_labs_pipeline(account) -> dict[str, str]:
    """One saved pipeline plan in the account's own outputs folder."""
    from core.labs import pipelines
    from utils.account_context import run_as

    _enable_labs(account)
    path = run_as(account, pipelines.root) / f"{PIPELINE_ID}.json"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps({"id": PIPELINE_ID, "name": "matrix", "spec": {}}), encoding = "utf-8"
    )
    return {"pipeline_id": PIPELINE_ID}


FACTORIES = {
    "routes.labs:DELETE:/benchmarks/{record_id}": Factory(
        "labs-benchmark",
        right = (403,),
        wrong = (403,),
        reason = (
            "owner-only delete: a managed account is refused with 403 before its own record is "
            "looked up, and the owner's copy of a managed record does not exist there, hence 404"
        ),
    ),
    "routes.labs:DELETE:/pipelines/{pipeline_id}": Factory(
        "labs-pipeline",
        right = (403,),
        wrong = (403,),
        reason = (
            "owner-only delete: a managed account is refused with 403 before its own plan is "
            "looked up, and the owner's copy of a managed plan does not exist there, hence 404"
        ),
    ),
}

SKIPPED: dict[str, str] = {}
