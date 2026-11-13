# Security Audit — Unsloth Studio hardening pass

Scope: `C:/Projectas/UnslothX-` (fork of Unsloth Studio). Backend `studio/backend` (FastAPI, `routes/`, `auth/`, `utils/`, `main.py`), frontend `studio/frontend/src` where relevant.
Method: read-only inspection (`rg`/`grep`/`sed`/`read`). Every claim below carries a `file:line`. Nothing was modified except this file; no servers started; no state-changing git commands.

Route inventory: **323 mutating handlers** (`POST/PUT/PATCH/DELETE`) across `routes/**/*.py` + `main.py`, enumerated mechanically and each one checked for an auth dependency (router-level `APIRouter(dependencies=...)`, per-route `Depends(get_current_subject)`/owner dependency, or an in-handler credential check).

## Checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | No unauthenticated destructive endpoints | **pass** | All 323 mutating routes are authenticated. Router-level: `routes/accounts.py:29` (auth + owner), `routes/engines.py:15`, `routes/labs.py:43`, `routes/mcp_servers.py:70`, `routes/providers.py:76`, `routes/advanced.py:43`, `routes/data_recipe/__init__.py:26` (wraps seed/validate/jobs/mcp/library), `routes/settings.py:247-249` (account/owner/shared sub-routers, included at `5864-5866`). Per-route on the rest, incl. `main.py:2383-2385` (`/api/shutdown`, auth+owner). The only two mutating routes without a dependency authenticate in-handler: `main.py:2397` via `x-desktop-secret` (`main.py:2180-2189`), `inference.py:37637/37676` via `_authenticate_header_or_query` (`inference.py:4413-4443`). `/api/auth/login`+`/refresh` are unauthenticated by design (`routes/auth.py:441,576`). |
| 2 | User input validated at every boundary | **partial** | Strong on the dangerous boundaries: path containment `training.py:5269-5287` + `5206-5230`, `inference.py:37479-37500`, browse allowlist `models.py:2017-2031`, provider-URL SSRF/cloud-metadata block `core/inference/providers.py:1101-1155`, webhook-URL class check `utils/webhook_settings.py:51-75`, SQL is fully parameterized (only fixed fragments interpolated, e.g. `storage/studio_db.py:2330-2394`, `2541-2700`). Residual gaps: SEC-7, SEC-8. |
| 3 | No hardcoded secrets | **pass** | No literal key/password/token in non-test backend code (`rg` for `api_key|secret|password|token = "<16+ chars>"` and `hf_…|sk-…|ghp_…` returned only sentinels at `core/inference/sandbox_probe.py:29-32`). Desktop secret is minted (`auth/storage.py:1684`) and PBKDF2-hashed (`1704-1723`); signing secret comes from SQLite. |
| 4 | Authorization on protected resources | **partial** | Correct centrally: per-account storage roots with containment check (`utils/paths/storage_roots.py:281-287`, per-account `studio.db`), owner-only account lifecycle (`routes/accounts.py:29`), foreign-job 404 (`core/training/account_jobs.py:212-220` called from `training.py:3206,3248`), `policy.require_account_scope` (`auth/policy.py:114-121`). Gap: SEC-2 (canvas links lose the account), SEC-6 (missing owner check on one sibling route). |
| 5 | Error messages do not leak internals | **partial** | No catch-all handler; unhandled errors yield the bare Starlette 500. `log_and_http_error` (`utils/utils.py:831-852`) and `safe_error_detail` (`utils/utils.py:787-812`) keep detail user-facing, and `/v1` envelopes sanitize validation input echo (`utils/api_errors.py:230-291`). Residual leaks: SEC-7. |
| 6 | No `eval`/`exec`/unsafe deserialization | **pass** | No builtin `eval()` at all in the backend. `exec()` only on the installed library's own source (`core/inference/diffusion_capture_safe.py:350`, `core/inference/diffusion_autoopt/patches.py:175`) — both compile-then-exec with signature/name resolution checks → **defended**. `torch.load` always `weights_only=True` (`core/advanced/merge.py:802-804`, `core/inference/diffusion.py:6411`, `diffusion_prequant.py:330`, `prequant_legacy_int8.py:228`, `training/diffusion_checkpoint.py:2089`). All `yaml` loads are `safe_load` (`core/inference/skills.py:304`, `utils/models/model_config.py:4115,4659-4683`, `utils/inference/inference_config.py:96`, `hub/utils/comfy_models.py:91`) → **defended**; `yaml.load` (non-safe) is never called. `ast.literal_eval` only on bounded literals (`inference/tool_loop_controller.py:615`, `run.py:1724`, `utils/transformers_version.py:2982`, `diffusion_nvfp4_install.py:411`) → **defended**. No `marshal`. No `shell=True` anywhere in the backend (a shell is only reached deliberately through the sandboxed bash tool, `core/inference/tools.py:23733+`). |
| M1 | `SecurityHeadersMiddleware` bypass | **partial** | CSP/`nosniff`/`Referrer-Policy`/`X-Frame-Options`/`no-store` all applied with `setdefault` so a route may legitimately override; script-src never gets `unsafe-inline` except the nonce (`main.py:1237-1310`). Its path tests ignore `root_path` → SEC-5. |
| M2 | `TrustedHostsMiddleware` bypass | **fail** | Off unless the env var is set and nothing sets it → SEC-1. Matching itself is sound (`utils/trusted_hosts.py:118-134`). |
| M3 | `MaxBodyMiddleware` bypass | **pass** | Raw ASGI with counted streaming, `Content-Length` pre-check and post-read recount (`main.py:1724-1750`); covers `/v1`, `/api/`, `/mcp`, `/p`, `/tokenize`, `/detokenize` (`main.py:1461-1486`); normalizes `root_path` (`1687-1691`); 411 for unbounded passthroughs not explicitly allow-listed (`1716-1721`, list `1536`). |
| M4 | `XApiKeyMiddleware` bypass | **pass** | Rewrites only a single `x-api-key` starting `sk-unsloth-` and only when `Authorization` is absent or an empty bearer; ≥2 keys or ≥2 authorizations leave headers alone (`auth/x_api_key.py:24-40`). Downstream repeated-`Authorization` ambiguity is rejected (`auth/authentication.py:160-168`). |
| M5 | `KeylessToolPolicyMiddleware` bypass | **pass** | Fails closed on an unreadable store, on any in-flight settings write, and on a stale generation (`utils/keyless_api_access.py:579-587`); repeated `Authorization` rejected (`620-622`); tool grant clamped contextually (`596-597`). |
| M6 | `RemoteAccessStopResponseMiddleware` bypass | **partial** | Exact literal path compare with no `root_path` handling and no trailing-slash tolerance (`utils/remote_access_settings.py:41`) → SEC-5. |
| M7 | `RemoteAccessCORSMiddleware` bypass | **pass** | Adds exactly the published tunnel origin on top of the startup allowlist (`main.py:1801-1814`); `allow_credentials=False` with `["*"]` is consistent because no route authenticates by cookie (`1838-1845`); `max_age=60` bounds preflight staleness (`1851`). |

## Findings

### SEC-1 — Host header is never validated by default (DNS rebinding) — medium
`main.py:1159-1161` registers `TrustedHostsMiddleware` "so the check runs before routing", but the middleware returns immediately unless `UNSLOTH_TRUSTED_HOSTS` is set (`utils/trusted_hosts.py:138-141`), and neither `run.py` nor any other module ever sets it (`rg UNSLOTH_TRUSTED_HOSTS` hits only `main.py:1157`).
Combined with `allow_origins=["*"]` (`utils/host_policy.py:277-280`, applied `main.py:1833`) and no cookie auth, a page on `evil.example` whose DNS is re-pointed at `127.0.0.1:<port>` becomes same-origin with the backend and can read every unauthenticated route (`/api/health`, `/api/liveness`, `/api/auth/status`) and drive `/api/auth/login` / `/api/auth/identity`.
Why it matters: unauthenticated host-identifying data and login/bootstrap state are readable by any page the user visits; the module docstring argues keyless admission already blocks the serious case, which is true, but the hardening pass asked for a host check, and the current default is "no check".
Remediation: default the pattern list to loopback + `localhost` + the Tauri origins when the variable is unset (keeping the env var as an override), rather than skipping the middleware.
Covering test: `studio/backend/tests/test_trusted_hosts.py` (exercises the middleware, but only with the variable set).

### SEC-2 — Canvas `?link=` capability carries no account binding, and leaves the account context at `OWNER` — medium
`_canvas_link_subject` (`routes/inference.py:4404-4410`) verifies the HMAC and returns only `claims["sub"]`; the callers discard that return value (`37651`, `37687`, `37637-37660`, `37809`). No `bind_account`/`run_as` runs, so `current_account()` keeps its `OWNER` default (`utils/account_context.py:37`) and every downstream path (`_sandbox_dir_for` → `sandbox_root` → `account_path("sandbox")`, `37753-37762`, `core/inference/tools.py:11286-11300`, `utils/paths/storage_roots.py:257-287`) resolves inside the **owner** workspace.
The media-link path does it correctly for comparison: the account id rides in the signed target (`hub/services/models/account_access.py:45-48`) and the handler re-binds it with `run_as` (`inference.py:52319-52331`; same for images `53237-53248`; library `routes/library.py:369,385,493`).
Why it matters: once links are minted, a managed account's `<img>`/`<iframe>` canvas links resolve the wrong root, and any session id that collides serves another account's tree. It is currently latent because there is no minter at all (SEC-9).
Remediation: mint with `media_link_target(session_id)`-style payload and bind the resolved `AccountContext` with `run_as`/`bind_account` before touching the filesystem, mirroring `get_gallery_audio_file_signed`.
Covering test: `tests/test_canvas_link_auth.py` — but only owner-subject cases (`:90-141`), so the gap is untested.

### SEC-3 — Keyless `full` scope is a per-host, not per-user, boundary — medium
With `keyless_api_access_scope=full` any caller that reaches the port over loopback, addressing it by literal/`localhost`, with no `Origin`, no cross-site `Sec-Fetch-Site` and a direct `Host`, authenticates as the installation owner for **every** route (`utils/keyless_api_access.py:228-241`, `443-464`, `600-629`) — including `POST /api/shutdown`, model deletion (`routes/models.py:5745`) and `/api/settings/remote-access/start` (`settings.py:4260`). There is no OS-user check: a second local user, or any local process, is owner-equivalent while the setting is on.
Note also that a browser **cannot** use it — `keyless_transport_allowed` refuses any request carrying `Origin` (`utils/keyless_api_access.py:483-486`) — so the scope is effectively "local non-browser clients only". That is a defensible design, but it is a broader grant than the docstring's `curl`/SDK framing suggests, and the UI's `exposure` field (`access_exposure`, `:262-306`) does not mention same-host other users.
Why it matters: on a shared workstation the setting silently grants full owner mutations to every local account; the window between a successful write and the cache refresh is covered (`579-587`), but the trust boundary itself is host-wide.
Remediation: either document "any local user" in the setting's warning text, or add an optional loopback-peer-user check where the platform allows it.
Covering test: `tests/test_keyless_api_access.py`, `tests/test_keyless_api_access_adversarial.py`, `tests/test_keyless_tunnel_exposure.py`.

### SEC-4 — Unauthenticated 100k-iteration PBKDF2 behind `POST /api/desktop/shutdown`, unthrottled — low
`main.py:2397` has no auth dependency; `_desktop_shell_subject` (`main.py:2180-2189`) spends `validate_desktop_secret` → `_pbkdf2_desktop_secret` → 100_000 PBKDF2 rounds (`auth/storage.py:1165`, `1189-1201`, `1703-1713`) on a caller-supplied value, in the threadpool, with no rate limit. Only the cheap shape regex runs first (`1710`; `desktop_secret_is_well_formed`, `1175-1186`), so any `desktop-` + 64-char guess reaches the KDF. `/api/auth/desktop-login` does throttle the same KDF (`routes/auth.py:521-529`); this route does not.
Why it matters: an unauthenticated attacker can queue unbounded KDF work against a small shared executor; every request also re-reads SQLite in a transaction.
Remediation: reuse the `/desktop-login` bucket (or a dedicated one keyed on `client_key`) before calling `validate_desktop_secret`.
Covering test: `tests/test_auth_rate_limit.py` / `tests/test_login_rate_limit.py` cover login throttling only.

### SEC-5 — Two middlewares match `scope["path"]` without stripping `root_path` — low
`utils/remote_access_settings.py:41` compares `scope.get("path") == "/api/settings/remote-access/stop"` and `main.py:1344/1347/1359` uses `path` for the docs CSP, the `_FRAME_SHELL_PATHS` exemption and the `/api/` `no-store` rule. Under `--root-path /x` uvicorn keeps the prefix inside `path` — stated in this very file's own `MaxBodyMiddleware` (`main.py:1689-1691`), and handled by `normalize_request_path` for keyless admission (`utils/keyless_api_access.py:313-322`).
Consequences when deployed behind a path prefix: the stop request does not take the connector lease (re-opening the stale-preflight window `max_age=60` exists to close, `main.py:1847-1851`); framed shells lose the `X-Frame-Options` exemption and get `DENY`; `/api` responses lose `no-store`; `/docs` loses its nonce-less font/worker CSP. A trailing-slash `POST …/stop/` misses the compare for the same reason.
Remediation: strip `root_path` (or use `starlette.routing.get_route_path`) in both middlewares, and `rstrip("/")` the stop path.
Covering test: `tests/test_middleware.py` (no `root_path` cases for these paths).

### SEC-6 — Sandbox `reveal` (directory form) skips the installation-owner check its siblings have — low
`POST /api/inference/sandbox/{session_id}/open` requires the owner (`inference.py:37651-37655`), and `reveal` requires it in the `file` branch (`37695-37696`), but the no-`file` branch that launches the host file manager (`37712-37750`) never calls `account_access.require_installation_owner()`. `routes/library.py:316-318` also guards its reveal with the owner check.
Why it matters: a managed account — which should not necessarily drive the host's desktop — can make the backend spawn a file manager on the host. Impact is limited (no path is attacker-chosen: the directory comes from `_sandbox_dir_for`), so this is consistency/least-privilege rather than an escape.
Remediation: add `account_access.require_installation_owner()` before the no-`file` branch.
Covering test: `tests/test_sandbox_files_and_storage_roots.py`, `tests/test_canvas_link_auth.py:133`.

### SEC-7 — Absolute host paths interpolated into HTTP error details — low
`routes/rag.py:1017` (503) and `:1020` (422) format `f"{filename} could not be read: {exc}"`; `exc` is an arbitrary parser exception and the parsed path is the stored upload (`_save_upload`, absolute, under the account workspace), so the install layout and the account id reach the client. Same shape at `routes/export.py:670` (`f"Q4NX conversion failed: {e}"`, arbitrary exception) and `routes/training.py:5913`. Elsewhere the codebase is careful — `log_and_http_error` and `safe_error_detail` exist precisely for this (`utils/utils.py:831-852`, `787-812`) and the `/v1` envelope truncates and drops the raw input echo (`utils/api_errors.py:230-248`).
Why it matters: host-path disclosure to an authenticated caller (relevant in tunnel/LAN mode where the caller may not own the host); not a credential leak.
Remediation: pass these through `safe_error_detail`/`log_and_http_error`, or `os.path.basename` the path before interpolating.
Covering test: none specific (the `_login_failure_detail` path-form rationale at `routes/auth.py:424-437` is the closest precedent).

### SEC-8 — Library upload display name is not stripped of `.`/`..` or control characters — low
`_base_name` (`routes/library.py:680-681`) splits on `[\\/]` and truncates to 255 but keeps `..`, `.`, leading dots and control bytes, unlike the deliberate `_safe_dataset_image_path` (`training.py:5269-5272`) and `_sanitize_filename` (`rag.py:175`) checks elsewhere.
Why it matters: contained today because the bytes are written under a uuid file name (`core/library.py:221-240`) and only the label reaches the DB, so this is defence-in-depth, not a live traversal.
Remediation: reject/replace `.`/`..`, control characters and leading dots in `_base_name`, matching the dataset path rule.
Covering test: none specific.

### SEC-9 — Canvas routes hard-reject `?token=` but nothing mints `?link=` — low
`_authenticate_header_or_query` raises 400 for any `?token=` (`inference.py:4424-4431`), telling the caller to send `?link=`. `_sign_canvas_link` (`4397-4402`) has **no production caller** — `rg` finds only `_canvas_link_subject` and `tests/test_canvas_link_auth.py:90-141`; the frontend works around it with `authFetch` and a comment saying so (`studio/frontend/src/components/assistant-ui/sandbox-files-view.tsx:38-41`).
Why it matters: the documented header-less path (`<img>`, `<iframe>`, native save) is unreachable in production, so the only exercised auth path on these routes is the bearer. Every finding that depends on link behaviour (SEC-2) is therefore untested in the field.
Remediation: expose the mint (e.g. return `link` alongside a sandbox listing) and add the managed-account case to `tests/test_canvas_link_auth.py`.
Covering test: `tests/test_canvas_link_auth.py` (mints by calling the private helper directly).

Findings by severity: **critical 0, high 0, medium 3 (SEC-1, SEC-2, SEC-3), low 6 (SEC-4…SEC-9). Total 9.**

## Verified safe

Checked and found already defended (so the report is not one-sided):

- **Auth dependency shape.** A single bearer parser (`auth/authentication.py:139-186`) that fails closed when disabled, rejects repeated `Authorization` headers, distinguishes UI session vs `sk-unsloth` key vs keyless, and re-binds the `AccountContext` from the DB row — not from the token's `sub` (`:502-505`, `:535`).
- **Account isolation by construction.** Per-account workspace + per-account `studio.db`, with a containment assertion on every resolution (`utils/paths/storage_roots.py:281-287`) and 404-not-403 for foreign resources (`auth/policy.py:114-121`).
- **Owner gating on the destructive surface.** Shutdown (`main.py:2385`), account lifecycle (`routes/accounts.py:29`), engine install/remove (`engines.py:36-71`), labs `PUT`/housekeeping/benchmarks (`labs.py:203-788`), cache purge (`settings.py:1489`), model deletion (`models.py:5745`), compute target (`compute.py:290`), NPU (`npu.py:178-237`), remote/LAN/keyless/MCP switches (`settings.py:4251-4620` with the UI-session-only rule at `4561-4569`, `4607-4614`).
- **Link/token crypto.** HMAC-SHA256 with domain separation, length bounds, strict base64url, signature-before-parse, `compare_digest`, exclusive `exp`, scope check, reserved-claim protection (`utils/scoped_link.py:86-192`); preview links sign the canonical ref only, versioned, account-prefixed for managed runs (`utils/preview_token.py:19-31`).
- **SSRF gas.** Provider base URLs reject non-http(s), control chars, missing host, cloud-metadata hosts *and* DNS aliases of them, with an opt-in private-address block (`core/inference/providers.py:1119-1155`); webhooks reject link-local/multicast/reserved/unspecified and embedded credentials (`utils/webhook_settings.py:51-75`); YouTube input is reduced to a regex-extracted video id (`routes/youtube.py:54-56`).
- **Path traversal.** `_resolve_dataset_folder` rejects traversal and symlinked dataset dirs and re-proves containment (`training.py:5206-5230`); `_safe_dataset_image_path` rejects separators/`..`/NUL and non-image suffixes (`5269-5287`); `_contained_sandbox_path` allow-lists each segment, rejects absolute/drive parts and decides containment on `realpath` (`inference.py:37479-37500`), then opens with `O_NOFOLLOW` (`37830-37842`); `browse-folders` is allow-listed and resolves symlinks (`models.py:2017-2060`).
- **SQL.** Fully parameterized; interpolated fragments are fixed column/placeholder lists (`storage/studio_db.py:2330-2394`, `2541-2700`, `storage/providers_db.py:302`, `routes/rag.py:497`).
- **Body limits.** Streaming counter plus `Content-Length` pre-check, root-path normalized, 411 for the passthroughs that could hold memory (`main.py:1461-1536`, `1687-1750`).
- **CSP / framing.** No `unsafe-inline` in `script-src` for the app shell, per-response nonce for the only inline script, `nosniff`, `Referrer-Policy: no-referrer`, `frame-ancestors 'none'` off-Colab, and sandbox shells that are allowed to be framed carry their own explicit `frame-ancestors` + `sandbox` (`main.py:1237-1360`, `inference.py:4158-4196`, `routes/browser.py:170-180`).
- **MCP surface.** API key mandatory, dummy keyless bearers refused, `Origin` allow-list, one bounded body, owner-only host-file access (`studio_mcp/gate.py:146-227`); the decisions mount shares the auth dependency (`routes/systemone.py:514-527`).
- **Login hardening.** Per-name *and* per-IP buckets, a shared unknown-user bucket to hide existence, `equalize_login_work` for the unknown path, non-leaky 401 text, throttled refresh and desktop login (`routes/auth.py:390-530`), conditional password/secret writes against concurrent rotation (`:660-680`, `:690-760`), API-key expiry capped by the parent key's expiry (`:816-824`).
- **Error surfacing.** No catch-all handler, so an unexpected exception returns a bodyless 500; `/v1` envelopes sanitize validation errors, and the login hint deliberately uses a PATH form to avoid leaking the install layout (`routes/auth.py:424-437`).
- **Health/liveness disclosure.** Unauthenticated `/api/health` returns only non-fingerprinting flags; `version`/`device_type` need a bearer (`main.py:2192-2201`, `2259-2277`).

## Not confirmed / limits of this audit

- Multi-account behaviour of the settings, training and inference routes was verified by reading the central mechanisms (`account_path`, `job_is_foreign`, `require_model_access`, per-account DBs) and sampling call sites, not by executing the multi-account suites; the 1,778 test files were not run in full.
- `UNSLOTH_TRUSTED_HOSTS` and `--root-path` deployments (SEC-1, SEC-5) were reasoned from code and the file's own comments, not reproduced against a running prefix-stripping proxy.
- The `studio/backend/vendor`, `hub`, `core/**` trees were only inspected where a checklist item pointed at them; a full review of worker/engine subprocess argv construction was out of scope for this pass.

## Remediation (wave 2)

Second wave, scoped to `utils/trusted_hosts.py`, `utils/remote_access_settings.py`, `routes/inference.py`,
`routes/library.py`, `routes/rag.py`, `routes/export.py`, `routes/training.py` and the tests beside them.
`main.py`, `routes/models.py`, `utils/host_policy.py`, `utils/cache_inventory.py` and the frontend were owned by
other hands and left alone.

| ID | Status | Files touched | Test proving it | Suite result |
|---|---|---|---|---|
| SEC-1 | fixed | `utils/trusted_hosts.py` (docstring, `_DEFAULT_PATTERNS`, `_DEFAULT_PARSED`, `_is_lan_literal`, `__call__`), `tests/test_trusted_hosts.py`, `tests/test_series_regressions_v2.py` (renamed its "inert" test) | `test_a_foreign_host_is_refused_by_default`, `test_a_public_literal_still_has_to_be_listed` (both fail on the old file) plus `test_the_default_list_admits_every_host_the_app_answers_to` and `test_the_variable_overrides_the_defaults_rather_than_extending_them` as the no-mode-breaks guard | `test_trusted_hosts.py` 10 passed; `test_series_regressions_v2.py` 23 passed |
| SEC-2 | fixed | `routes/inference.py` (`_sign_canvas_link` signs `media_link_target`, `_canvas_link_subject` becomes `_canvas_link_account`, `_authenticate_header_or_query` binds the account), `tests/test_canvas_link_auth.py` | `test_a_managed_accounts_link_resolves_inside_that_accounts_sandbox` and `test_a_link_naming_an_account_that_no_longer_exists_is_refused` (fail on the old file) | `test_canvas_link_auth.py` 8 passed; `test_keyless_api_access.py` 40 passed |
| SEC-5a | fixed (stop-path half only; the `SecurityHeadersMiddleware` half is `main.py`'s) | `utils/remote_access_settings.py` (`_is_stop_request` strips `root_path` and the trailing slash), `tests/test_remote_access_settings.py` | `test_stop_response_lease_survives_a_root_path_and_a_trailing_slash` (4 of its 8 cases fail on the old file), `test_the_stop_lease_is_post_only` | `test_remote_access_settings.py` 29 passed |
| SEC-6 | fixed | `routes/inference.py` (`reveal_sandbox_dir`, no-`file` branch), `tests/test_sandbox_files_and_storage_roots.py` | `test_revealing_a_sandbox_directory_is_refused_to_a_managed_account` (fails on the old file) | targeted 1 passed; whole file 261 passed, 40 failed, 5 skipped — all 40 are the local Windows symlink-privilege (`WinError 1314`) failures, none in the reveal path |
| SEC-7 | fixed | `routes/rag.py` (`_parse_failure_detail`), `routes/export.py`, `routes/training.py`, `tests/test_rag_extract_text.py`, `tests/test_export_q4nx_accounts.py`, `tests/test_diffusion_dataset_api.py` | `test_a_parser_failure_reports_no_host_path`, `test_a_missing_parser_dependency_reports_no_host_path`, `test_a_failed_conversion_reports_no_host_path`, `test_a_failed_import_reports_no_host_path` (all fail on the old files) | `test_rag_extract_text.py` 10 passed; `test_export_q4nx_accounts.py` 5 passed, 1 failed (pre-existing symlink); `test_diffusion_dataset_api.py` 57 passed, 4 failed (pre-existing: 3 symlink, 1 case-insensitive-probe) |
| SEC-8 | fixed | `routes/library.py` (`_base_name` delegates to `core.library.safe_file_name`), `tests/test_library_routes.py` | `test_the_display_name_is_a_label_not_a_path` (9 of its 15 cases fail on the old file), `test_the_label_keeps_the_extension_and_stays_bounded`, `test_an_upload_named_with_a_path_is_stored_and_listed_under_its_label` | `test_library_routes.py` 160 passed, 8 failed, 3 skipped — the 8 are the same pre-existing symlink failures |
| SEC-3 | skipped | — | — | out of this wave's scope (design/documentation call, no code defect pinned) |
| SEC-4 | skipped | — | — | out of this wave's scope (needs the `/desktop-login` bucket reused; `main.py` untouched) |
| SEC-9 | skipped | — | — | out of this wave's scope (no production minter exposed); SEC-2 changed the mint's payload, so the minter work now starts from an account-bound link |

Notes for the next hand:

- Default-on means the shipped default list must keep every host a mode sends. It carries loopback, `localhost`,
  `*.localhost` (the Tauri origins), private/link-local literals, `*.local`, `*.trycloudflare.com`,
  `*.googleusercontent.com`, and the two in-process hosts `testserver` and `unsloth-mcp.invalid` — the last one
  matters because `studio_mcp/forward.py` replays every MCP tool call into the app under it. `main.py:1157`'s
  comment ("Off unless `UNSLOTH_TRUSTED_HOSTS` is set") and `docs/security.md`'s "off by default" wording are now
  stale and sit in files another hand owned.
- `routes/inference.py:4443` was calling `subject_for_header_or_query_token(request, None)` while
  `auth/authentication.py` had already dropped the token parameter; that call is now the one-argument form the
  docstring and `test_series_regressions_v2.py` describe.
- SEC-7 redacts rather than drops, so the reason survives: `redact_paths_in_text` replaces a path with `<path>`
  (and swallows the punctuation after it), so `routes/export.py` first reduces its own two known paths to their
  last component and lets the redactor handle anything else.
