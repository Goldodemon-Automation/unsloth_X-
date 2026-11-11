# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""HTTP surface for the telemetry collector.

Two audiences, two surfaces:

* ``POST /api/v1/report`` -- the desktop app's opt-in reporter. Anonymous apart
  from an install id; optionally gated by a shared ingest token.
* ``/api/admin/*`` -- the operator console. A token when
  ``X8_TELEMETRY_ADMIN_TOKEN`` is set, loopback-only when it is not.

Run from the repository root::

    uvicorn telemetry.server:app --port 8899

and open http://127.0.0.1:8899/.
"""

from __future__ import annotations

import hmac
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from telemetry import store

ADMIN_TOKEN_ENV = "X8_TELEMETRY_ADMIN_TOKEN"
INGEST_TOKEN_ENV = "X8_TELEMETRY_INGEST_TOKEN"
ADMIN_COOKIE = "x8_telemetry_admin"
MAX_REPORT_BYTES = 2 * 1024 * 1024
DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


def _admin_token() -> str:
    return os.environ.get(ADMIN_TOKEN_ENV, "").strip()


def _ingest_token() -> str:
    return os.environ.get(INGEST_TOKEN_ENV, "").strip()


def _loopback(request: Request) -> bool:
    """A direct loopback request, not a reverse proxy standing in front of one.

    A proxy connects from 127.0.0.1 too, so the peer address alone would make an
    un-tokened collector's admin API public the moment it is put behind one. The
    Host header has to name loopback as well; an empty HTTP/1.0 Host is not a
    public name and stays allowed.
    """
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return False
    host_header = request.headers.get("host", "").strip()
    if host_header.startswith("["):
        host_name = host_header[1:].split("]", 1)[0]
    elif ":" in host_header:
        host_name = host_header.split(":", 1)[0]
    else:
        host_name = host_header
    return host_name.lower() in {"127.0.0.1", "localhost", "::1", "testserver", ""}


def _presented_token(request: Request) -> str:
    cookie = request.cookies.get(ADMIN_COOKIE, "")
    if cookie:
        return cookie
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _is_admin(request: Request) -> bool:
    token = _admin_token()
    if not token:
        return _loopback(request)
    presented = _presented_token(request)
    return bool(presented) and hmac.compare_digest(presented, token)


def require_admin(request: Request) -> None:
    if not _is_admin(request):
        raise HTTPException(status_code = 401, detail = "Operator sign-in required.")


def _require_ingest_auth(request: Request) -> None:
    token = _ingest_token()
    if not token:
        return
    header = request.headers.get("authorization", "")
    presented = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not presented or not hmac.compare_digest(presented, token):
        raise HTTPException(status_code = 401, detail = "Invalid ingest token.")


class Device(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    os: Optional[str] = Field(None, max_length = 48)
    os_version: Optional[str] = Field(None, max_length = 64)
    arch: Optional[str] = Field(None, max_length = 32)
    cpu: Optional[str] = Field(None, max_length = 160)
    cpu_cores: int = Field(0, ge = 0, le = 4096)
    cpu_threads: int = Field(0, ge = 0, le = 4096)
    ram_gb: float = Field(0, ge = 0, le = 4096)
    gpu: Optional[str] = Field(None, max_length = 160)
    gpu_vendor: Optional[str] = Field(None, max_length = 48)
    vram_gb: float = Field(0, ge = 0, le = 4096)
    gpu_count: int = Field(0, ge = 0, le = 64)
    backend: Optional[str] = Field(None, max_length = 24)
    apple_silicon: bool = False


class Machine(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    locale: Optional[str] = Field(None, max_length = 32)
    python: Optional[str] = Field(None, max_length = 24)
    torch: Optional[str] = Field(None, max_length = 24)
    transformers: Optional[str] = Field(None, max_length = 24)
    accounts: int = Field(0, ge = 0, le = 1000)


class UsageEntry(BaseModel):
    day: str
    source: Literal["chat", "api"]
    requests: int = Field(0, ge = 0, le = 1 << 40)
    prompt_tokens: int = Field(0, ge = 0, le = 1 << 40)
    completion_tokens: int = Field(0, ge = 0, le = 1 << 40)

    @field_validator("day")
    @classmethod
    def valid_day(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("day must be an ISO date") from error
        return parsed.isoformat()


class ModelEntry(BaseModel):
    day: str
    model: str = Field(min_length = 1, max_length = 200)
    kind: Literal["chat", "api"] = "chat"
    uses: int = Field(0, ge = 0, le = 1 << 40)
    prompt_tokens: int = Field(0, ge = 0, le = 1 << 40)
    completion_tokens: int = Field(0, ge = 0, le = 1 << 40)

    @field_validator("day")
    @classmethod
    def valid_day(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("day must be an ISO date") from error
        return parsed.isoformat()


class TrainingEntry(BaseModel):
    run_key: str = Field(min_length = 1, max_length = 64)
    day: Optional[str] = None
    base_model: Optional[str] = Field(None, max_length = 200)
    dataset: Optional[str] = Field(None, max_length = 200)
    status: Optional[str] = Field(None, max_length = 32)
    method: Optional[str] = Field(None, max_length = 32)
    steps: int = Field(0, ge = 0, le = 1 << 40)
    epochs: float = Field(0, ge = 0, le = 100000)


class MergeEntry(BaseModel):
    merge_key: str = Field(min_length = 1, max_length = 64)
    day: Optional[str] = None
    models: list[str] = Field(default_factory = list, max_length = 8)
    method: Optional[str] = Field(None, max_length = 32)


class Report(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    schema_version: int = Field(store.REPORT_SCHEMA, ge = 1, le = 16)
    install_id: str = Field(pattern = r"^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$")
    sent_at: int = Field(default_factory = lambda: int(time.time()), ge = 0)
    version: Optional[str] = Field(None, max_length = 64)
    window_days: int = Field(180, ge = 1, le = store.MAX_WINDOW_DAYS)
    device: Device = Field(default_factory = Device)
    machine: Machine = Field(default_factory = Machine)
    usage: list[UsageEntry] = Field(default_factory = list, max_length = 800)
    models: list[ModelEntry] = Field(default_factory = list, max_length = 6000)
    trainings: list[TrainingEntry] = Field(default_factory = list, max_length = 1000)
    merges: list[MergeEntry] = Field(default_factory = list, max_length = 1000)
    features: dict[str, int] = Field(default_factory = dict)


class LoginPayload(BaseModel):
    token: str = Field(min_length = 1, max_length = 512)


app = FastAPI(
    title = "Unsloth X8 telemetry",
    description = "Opt-in usage telemetry collector and operator console.",
    docs_url = None,
    redoc_url = None,
)


@app.middleware("http")
async def _bounded_report(request: Request, call_next):
    if request.url.path == "/api/v1/report":
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > MAX_REPORT_BYTES:
                    return JSONResponse(
                        {"detail": "Report too large."},
                        status_code = 413,
                    )
            except ValueError:
                return JSONResponse({"detail": "Bad Content-Length."}, status_code = 400)
    return await call_next(request)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    try:
        with store.transaction() as conn:
            count = conn.execute("SELECT COUNT(*) FROM installs").fetchone()[0]
        return {"ok": True, "installs": int(count)}
    except Exception as error:  # noqa: BLE001 -- health must answer, not raise
        return {"ok": False, "error": type(error).__name__}


@app.post("/api/v1/report")
def ingest_report(payload: Report, request: Request) -> dict[str, Any]:
    _require_ingest_auth(request)
    with store.transaction() as conn:
        counts = store.ingest(conn, payload.model_dump())
    return {"ok": True, **counts}


@app.get("/api/admin/session")
def admin_session(request: Request) -> dict[str, Any]:
    return {
        "authenticated": _is_admin(request),
        "tokenRequired": bool(_admin_token()),
    }


@app.post("/api/admin/login")
def admin_login(payload: LoginPayload, request: Request, response: Response) -> dict[str, Any]:
    token = _admin_token()
    if not token:
        return {"ok": True}
    if not hmac.compare_digest(payload.token, token):
        raise HTTPException(status_code = 401, detail = "Wrong token.")
    response.set_cookie(
        ADMIN_COOKIE,
        token,
        max_age = 30 * 24 * 60 * 60,
        httponly = True,
        samesite = "strict",
        secure = request.url.scheme == "https",
        path = "/",
    )
    return {"ok": True}


@app.post("/api/admin/logout")
def admin_logout(response: Response) -> dict[str, Any]:
    response.delete_cookie(ADMIN_COOKIE, path = "/")
    return {"ok": True}


@app.get("/api/admin/overview", dependencies = [Depends(require_admin)])
def admin_overview(days: int = Query(30, ge = 1, le = store.MAX_WINDOW_DAYS)) -> dict[str, Any]:
    with store.transaction() as conn:
        return store.overview(conn, days)


@app.get("/api/admin/installs", dependencies = [Depends(require_admin)])
def admin_installs(limit: int = Query(store.MAX_LIST_ROWS, ge = 1, le = store.MAX_LIST_ROWS)) -> dict[str, Any]:
    with store.transaction() as conn:
        return {"installs": store.list_installs(conn, limit = limit)}


@app.get("/api/admin/export.csv", dependencies = [Depends(require_admin)])
def admin_export() -> PlainTextResponse:
    with store.transaction() as conn:
        body = store.installs_csv(conn)
    return PlainTextResponse(
        body,
        media_type = "text/csv; charset=utf-8",
        headers = {"Content-Disposition": 'attachment; filename="x8-installs.csv"'},
    )


app.mount("/", StaticFiles(directory = DASHBOARD_DIR, html = True), name = "dashboard")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "telemetry.server:app",
        host = os.environ.get("X8_TELEMETRY_HOST", "127.0.0.1"),
        port = int(os.environ.get("X8_TELEMETRY_PORT", "8899")),
        reload = False,
    )


if __name__ == "__main__":
    main()
