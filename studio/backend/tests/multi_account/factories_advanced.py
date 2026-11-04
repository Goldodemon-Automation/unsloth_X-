# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Advanced Studio job records.

A job lives in the route module's in-process registry, not on disk, so the seeder
inserts a finished record directly. It carries the account that started it, which
is what makes a job id useless to another account: the read answers 404 rather
than handing over another account's output path and results.
"""

from .factory_base import Factory, seeder

JOB_ID = "matrixadvancedjob"


@seeder("advanced-job")
def seed_advanced_job(account) -> dict[str, str]:
    """One finished job record owned by ``account``.

    ``status`` is not "running" and ``output_dir`` is None so the Labs
    housekeeper, which walks the live registry for running jobs, never sees it.
    """
    from routes import advanced

    record = {
        "job_id": JOB_ID,
        "kind": "merge",
        "account_id": account.account_id,
        "status": "completed",
        "created_at": 1000.0,
        "finished_at": 1001.0,
        "output_dir": None,
        "library_visible": False,
        "resources": {"status": "granted", "need": None, "notes": []},
        "run_manifest": None,
        "result": None,
        "error": None,
        "progress": None,
    }
    with advanced._JOBS_LOCK:
        advanced._JOBS[JOB_ID] = record
    return {"job_id": JOB_ID}


FACTORIES = {"routes.advanced:GET:/jobs/{job_id}": Factory("advanced-job")}

SKIPPED: dict[str, str] = {}
