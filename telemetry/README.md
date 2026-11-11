# Unsloth X8 telemetry

An opt-in product-analytics collector and operator console for Unsloth X8. It
answers the questions the app itself cannot: which models people actually run,
what they fine-tune on, which checkpoints they merge, how many tokens flow in
and out over 2 / 4 / 9 / 12 / 30 / 180 days, and what hardware they run it on.

Nothing is collected until an owner turns it on in **Settings → Telemetry**. The
Settings tab also renders the exact JSON payload before anything is sent.

## What is stored

Per install, and per day:

- prompt and completion token counts, split by chat and OpenAI-compatible API
- model id, number of responses, and tokens attributed to it
- training runs: base model, dataset, method, status, steps, epochs
- merges: parent model ids, method, date
- hardware classes: OS, arch, CPU name, core/thread counts, RAM, GPU name, GPU
  vendor, VRAM, backend, GPU count
- app version, Python / torch / transformers versions, locale, feature counters

Never stored: prompts, replies, chat titles, file or folder paths, hostnames,
usernames, email addresses, Hugging Face tokens, API keys, or IP addresses. The
install id is a random UUID generated on the machine; it is not derived from any
hardware or account identifier.

A report replaces the day rows it covers, so retries are idempotent.

## Run it

From the repository root:

```bash
python -m pip install fastapi uvicorn
python -m uvicorn telemetry.server:app --host 127.0.0.1 --port 8899
```

Open <http://127.0.0.1:8899/>.

Configuration:

| Variable | Purpose |
| --- | --- |
| `X8_TELEMETRY_DB` | SQLite file path (default: `telemetry/telemetry.sqlite3`) |
| `X8_TELEMETRY_ADMIN_TOKEN` | Required for `/api/admin/*` once set. Unset = loopback-only. |
| `X8_TELEMETRY_INGEST_TOKEN` | Optional shared bearer token for `POST /api/v1/report` |
| `X8_TELEMETRY_HOST` / `X8_TELEMETRY_PORT` | Used by `python -m telemetry.server` |

For a deployment, put it behind TLS (Caddy, nginx, a tunnel) and set both tokens.
The admin cookie is `HttpOnly`, `SameSite=Strict`, and `Secure` on HTTPS. Set
`X8_TELEMETRY_INGEST_TOKEN` when the endpoint is public so a stranger cannot
inflate the counters; enter the same value under Settings → Telemetry.

## Wire the app to it

1. Open Unsloth X8 → **Settings → Telemetry**.
2. Set the endpoint to `https://your-collector.example.com/api/v1/report`.
3. Paste the ingest token if the collector has one.
4. Turn the switch on (and use **Send now** for the first report).

The reporter batches six-hourly while the app is running. **Preview payload**
shows exactly what will leave the machine; **Send now** is the same send, on
demand.

## API

Public:

- `POST /api/v1/report` — one install snapshot (validated, size-capped, bounded)
- `GET /healthz`

Operator (cookie or `Authorization: Bearer`):

- `GET /api/admin/overview?days=30` — KPIs, token windows, 180-day series, model
  leaderboard, merges, training, devices, feature adoption
- `GET /api/admin/installs` — one row per installation
- `GET /api/admin/export.csv` — installs as CSV
- `POST /api/admin/login` / `POST /api/admin/logout`
- `GET /api/admin/session`

## Layout

```text
telemetry/
  server.py            FastAPI app: ingest, admin API, static mount
  store.py             SQLite schema, idempotent ingest, aggregate queries
  dashboard/           index.html + styles.css + app.js (no build step)
  tests/               store, API, auth and window tests
  telemetry.sqlite3    created on first ingest (gitignored)
```

## Backup and retention

The whole dataset is one SQLite file plus its `-wal` / `-shm` siblings. Copy all
three while the service is stopped, or use `sqlite3 telemetry.sqlite3 ".backup"`.
Daily rows older than the client's 180-day window remain queryable through the
API's `days` parameter up to 400 days; prune with a simple `DELETE` when the
history is no longer useful.

## Tests

```bash
python -m pytest telemetry/tests -q
```

The suite covers idempotent re-ingest, the 2/4/9/12/30/180-day windows, model /
merge / training ranking, validation, the admin token and cookie flow, the
optional ingest token, CSV export and the dashboard being served.
