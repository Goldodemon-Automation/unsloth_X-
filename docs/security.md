# Security model

This is the one place that states what Unsloth Studio exposes, what stands on that
surface, and what does not. It is a record of this checkout, not a policy: every row
cites a `file:line` that was read, and a row that stops describing the code is a bug in
this file. Nothing here is aspirational — a control is either in the table below or in
[Open items](#open-items), and there is no third category.

## The threat model

- The normal shape is a local desktop app: one uvicorn process on loopback, hosted by
  the Tauri webview, whose CSP is `studio/src-tauri/tauri.conf.json:19`. The session
  JWT and refresh token sit in the renderer's `localStorage`
  (`studio/frontend/src/features/auth/session.ts:73`), so any script that runs on the
  app's origin holds them.
- The same server is optionally a LAN listener (`-H 0.0.0.0`, private-LAN inference) or
  a Cloudflare tunnel. Both bring other people into the trust boundary, and both make
  the socket peer useless as an identity: through the managed tunnel the peer is the
  local `cloudflared` process, so every visitor is `127.0.0.1`
  (`studio/backend/utils/client_ip.py:12`).
- The secrets are the Hugging Face token and the provider API keys (encrypted rows in
  `studio.db`, `studio/backend/storage/credential_secrets.py:4`), the account session,
  `sk-unsloth` API keys, and the local desktop secret. The attacker is a page the user
  has open (cross-origin fetch, DNS rebinding, injected script on the app's origin), a
  peer on the LAN, or a visitor on a published tunnel URL.

## Controls that exist

| Control | Where | What it stops |
| --- | --- | --- |
| CORS never pairs a wildcard origin with credentials | `studio/backend/main.py:1818` | Starlette reflects the request `Origin` when `["*"]` and credentials are both set, so this pairing would let any open page read credentialed responses from the loopback backend, `/api/auth` included. Nothing authenticates by cookie, so no client needs the flag. |
| The startup origin allowlist is chosen per launch mode | `studio/backend/main.py:1798`, `studio/backend/utils/host_policy.py:234` | A browser from an unlisted origin reading API responses. `["*"]` is only the uncredentialed desktop default; no env var puts an operator regex in front of `Origin` (`studio/backend/tests/test_cors_origins.py:115`). |
| A running tunnel adds exactly one published origin, and preflights expire in 60 s | `studio/backend/main.py:1785`, `studio/backend/main.py:1835` | Remote access turning into an allow-any-origin switch, and a preflight cached for 10 minutes still carrying a state-changing request after the tunnel is stopped (measured in WebKit). |
| Per-response CSP, inline script only behind a fresh nonce | `studio/backend/main.py:1228`, `studio/backend/main.py:1320` | Injected script on the origin that holds the session tokens. `script-src 'self'` plus a nonce spliced from an internal header that is stripped before the response leaves (`studio/backend/main.py:1316`). |
| Baseline headers on every response: `X-Frame-Options: DENY`, `nosniff`, `no-referrer`, camera/mic/geolocation off | `studio/backend/main.py:1325`, `studio/backend/main.py:1326`, `studio/backend/main.py:1327`, `studio/backend/main.py:1330` | Clickjacking (except the four framed shells listed at `studio/backend/main.py:1172`), MIME sniffing, referrer leakage, and a page reaching the microphone. |
| Swagger/ReDoc bundles are vendored and digest-checked, not loaded from a CDN | `studio/backend/main.py:1185`, `studio/backend/main.py:1352`, `studio/backend/tests/test_docs_ui_assets.py` | Third-party script running on the token origin: `localStorage` is origin-scoped, not path-scoped, so a CDN bundle at `/docs` can read the session JWT and call the API as that user. |
| The public preview page ships a nonce, not `unsafe-inline` | `studio/backend/routes/preview.py:300`, `studio/backend/routes/preview.py:319`, `studio/backend/assets/preview_page.html:316` | Inline script on the shared Studio origin, which holds the session tokens. `base-uri 'none'` and `object-src 'none'` close the same hole from the other side. |
| The desktop webview CSP sets `base-uri 'none'` and `object-src 'none'` | `studio/src-tauri/tauri.conf.json:19` | One injected `<base>` tag re-pointing relative module loads inside a webview that exposes the whole Tauri IPC surface. `base-uri` does not inherit from `default-src`. |
| Login is throttled per account and per IP, with an unknown-user slot and a separate desktop slot | `studio/backend/routes/auth.py:166`, `studio/backend/routes/auth.py:344`, `studio/backend/routes/auth.py:391`, `studio/backend/routes/auth.py:231`, `studio/backend/routes/auth.py:235` | Password spray, username rotation, and an unauthenticated caller locking out every real account in five attempts a minute. The bucket dict is bounded at 4096 with a sharded overflow that never evicts a live counter (`studio/backend/routes/auth.py:171`). |
| One shared lockout budget for every credential the backend checks | `studio/backend/auth/rate_limit.py:143`, used at `studio/backend/routes/auth.py:629` (`/refresh`) and `studio/backend/routes/auth.py:748` (`/change-password`) | Guessing that rotates between a refresh token and a password. Keyed on the client address, never on the submitted secret (`studio/backend/auth/rate_limit.py:150`), and `allow` is a pure read so a retrying caller cannot lengthen its own ban (`studio/backend/auth/rate_limit.py:74`). |
| PBKDF2-HMAC-SHA256 at 100k rounds, `hmac.compare_digest`, and a spent-on-miss hash | `studio/backend/auth/hashing.py:12`, `studio/backend/auth/hashing.py:25`, `studio/backend/auth/hashing.py:42` | A stored-hash timing oracle, which would answer "does this username exist" before any credential check. |
| Credentials at rest are AES-GCM ciphertext with the kind and scope as authenticated data | `studio/backend/storage/credential_secrets.py:99`, `studio/backend/storage/credential_secrets.py:41`, `studio/backend/storage/credential_secrets.py:108` | A copied or swapped ciphertext row presenting one account's provider key as another's, and tampering that fails closed (`studio/backend/tests/test_credential_secrets.py:104`). The key lives in `auth.db`, separate from the rows it protects. |
| Auth and settings databases are owner-only | `studio/backend/auth/storage.py:277`, `studio/backend/auth/storage.py:59`, `studio/backend/storage/credential_secrets.py:76` | Another account on the host, or a stray backup of the data directory, reading password hashes, refresh-token hashes, the JWT secret and the encryption key. |
| One credential scheme on the wire: `x-api-key` is rewritten to `Authorization` before anything reads it | `studio/backend/auth/x_api_key.py:20`, mounted outermost at `studio/backend/main.py:1849` | A second scheme that some readers honour and others ignore — a Bearer always wins, and a non-`sk-unsloth` value is left alone. |
| Callers are classifiable: UI session vs API key vs keyless | `studio/backend/auth/authentication.py:190`, `studio/backend/auth/authentication.py:341` | A route that treats every bearer as the local owner. Guards keyed on this: an API key cannot define a stdio MCP command (`studio/backend/auth/authentication.py:396`) and cannot reach the backend's ambient HF token (`studio/backend/auth/authentication.py:409`). |
| Keyless API access is off by default and bounded by transport facts | `studio/backend/utils/keyless_api_access.py:23`, `studio/backend/utils/keyless_api_access.py:480` | An owner who turns on "no key needed" for chat also opening training, or the setting surviving a public tunnel or Colab (`studio/backend/utils/keyless_api_access.py:367`, `studio/backend/utils/keyless_api_access.py:352`). |
| Keyless admission reads `Sec-Fetch-Site` and the `Host` literal | `studio/backend/utils/keyless_api_access.py:405`, `studio/backend/utils/keyless_api_access.py:423` | A cross-site `no-cors` GET reaching loopback (`Origin` is absent on those), and DNS rebinding, which keeps the page's own origin and so reads as a local client to every socket check. Repeated `Host` or `Sec-Fetch-Site` is refused rather than resolved (`studio/backend/utils/keyless_api_access.py:396`). |
| Server-side tools stay off for a keyless caller whatever the scope | `studio/backend/utils/keyless_api_access.py:567` | The `full`/`inference` grant silently becoming code execution in the python/terminal sandbox, which `/v1/chat/completions` runs on this machine. |
| Public preview links are HMAC capability tokens scoped to one ref and one account | `studio/backend/utils/preview_token.py:39`, `studio/backend/routes/preview.py:48`, `studio/backend/routes/preview.py:58` | Ref guessing and enumeration: a bad ref, a bad signature and sharing switched off all answer 404, and the token names no host, so the link survives a localhost/LAN/tunnel address change. |
| Preview chat is rate-limited per client IP and fails closed | `studio/backend/routes/preview.py:88`, `studio/backend/utils/preview_rate_limit.py:43` | One source driving GPU generation. 20 requests per 60 s; when the bucket table is full of live clients a new key is denied rather than given room by evicting a throttled bucket, which would hand out a reset (`studio/backend/utils/preview_rate_limit.py:55`). Output is capped at 1024 tokens per call (`studio/backend/routes/preview.py:42`) and the payload is stripped of tools, providers and thread scope (`studio/backend/routes/preview.py:104`). |
| The client address is resolved by one documented rule | `studio/backend/utils/client_ip.py:55` | A direct LAN caller forging a header to dodge a per-IP bucket: `X-Forwarded-For` is read only when `UNSLOTH_STUDIO_TRUST_FORWARDED` is set, and then the rightmost hop (`studio/backend/utils/client_ip.py:62`); `CF-Connecting-IP` is read only from a loopback peer (`studio/backend/utils/client_ip.py:67`), which is the tunnel. |
| Owner-only surfaces require a direct, un-proxied loopback connection | `studio/backend/utils/client_ip.py:112` | DNS rebinding and proxy or tunnel relays reaching the settings routes that install host-level tooling — the check rejects any request carrying a proxy header (`studio/backend/utils/client_ip.py:87`) or a non-loopback `Host` (`studio/backend/utils/client_ip.py:96`). Used at `studio/backend/routes/settings.py:5700` and `studio/backend/main.py:3216`. |
| `trust_remote_code` is never auto-enabled without a scan, a pin and explicit consent | `studio/backend/utils/security/__init__.py:65`, `studio/backend/utils/security/consent.py:179`, `studio/backend/utils/security/trusted_org.py:45` | Hub `auto_map` Python running on the user's machine. The scanner is the same one CI runs (`scripts/scan_packages.py`), and its ruleset version is recorded with each approval so an older verdict cannot auto-approve under new rules (`studio/backend/utils/security/remote_code_scan.py:32`). |
| The remote-code approval cache skips the dialog, never the scan, and never honors CRITICAL | `studio/backend/utils/security/remote_code_approvals.py:164` | A hand-edited approval store turning into a permanent allow for malicious code. Any store or commit-SHA error degrades to "ask again". |
| The pickle gate reads Hub's own scan verdict and fails closed on an unknown level | `studio/backend/utils/security/file_security.py:552`, `studio/backend/utils/security/file_security.py:23`, `studio/backend/utils/security/file_security.py:617` | Code execution during `from_pretrained` deserialization with `trust_remote_code=False`. Only clean/not-yet-scanned levels are non-blocking, so Hub schema drift cannot silently allow a bad verdict. Metadata only: the flagged files are never downloaded or unpickled. |
| A Hub dataset card cannot name host files | `studio/backend/utils/security/hub_data_files.py:83`, `studio/backend/utils/security/hub_data_files.py:131` | A README listing `/home/owner/private.csv` turning `load_dataset("org/repo")` into a host-file read, and an `http(s)` card path making the server dial loopback, LAN or cloud metadata. |
| Request bodies are capped before authentication and before parsing | `studio/backend/main.py:1599`, `studio/backend/main.py:1745` | An unauthenticated oversized body being held and parsed. Raw ASGI, so a chunked upload cannot bypass the cap; the chunked exemption is an explicit list of paths (`studio/backend/main.py:1510`). |
| An exposed first-run install with the bootstrap password unchanged tears itself down | `studio/backend/auth/bootstrap_timeout.py:146`, `studio/backend/auth/bootstrap_timeout.py:23` | A fresh, unconfigured instance staying publicly reachable after `--secure` / `-H 0.0.0.0`. Default 3600 s, `0` disables, never armed for `--api-only` or Colab. |
| A public launch prompts for the password before the server is reachable | `studio/backend/auth/terminal_prompt.py:240`, `studio/backend/auth/terminal_prompt.py:265` | The window between a bind on a public interface and the first login, where the seeded bootstrap password is a valid credential. Masked input, output on stderr. |
| A pre-auth error message names no local fact | `studio/backend/routes/auth.py:468` | Handing the OS account name and install layout to any open page: the failure hint is a bare PATH form and does not echo the submitted username, because the browser-served default resolves CORS to `["*"]`. |

## Open items

Three items were carried here, and two are cut: code in this checkout closes them, and a row that stops
describing the code is a bug in this file.

- The Host allow-list is an opt-in control now — `studio/backend/utils/trusted_hosts.py`, off unless
  `UNSLOTH_TRUSTED_HOSTS` is set, described in its own section below.
- The two client-IP trust models are one resolver — `studio/backend/utils/client_ip.py:152`, which
  `studio/backend/routes/auth.py:248` delegates to, so the login buckets and the shared credential lockout
  can no longer key on different clients. `studio/backend/tests/test_client_ip_agreement.py` pins it.

What is left is the residue of the query-token finding rather than the finding itself:

| Item | Why it is open | A fix touches |
| --- | --- | --- |
| The auth layer can still resolve a session bearer read from a query string, though no route asks it to. | Nothing on the surface hands one over any more: `studio/backend/routes/library.py:407` and `studio/backend/routes/inference.py:4426` refuse a `?token=` with a 400 that names `?link=`, `studio/backend/routes/data_recipe/jobs.py:142` never reads the session bearer from the query, and all three call the resolver with `None` (`studio/backend/routes/library.py:425`, `studio/backend/routes/inference.py:4444`, `studio/backend/routes/data_recipe/jobs.py:176`). The `?token=` that still appears in a served URL is an HMAC media link, not a session — `studio/backend/routes/rag.py:1176`, `studio/backend/routes/video.py:1072`, the gallery file routes. What is not removed is the capability: `studio/backend/auth/authentication.py:374` still takes a token argument and still resolves it to the same subject as an `Authorization` bearer, so the closure lives in the call sites and a new route that passes a query token re-opens the finding. | `studio/backend/auth/authentication.py:374` and `:350`, by dropping the parameter so the header is the only read — or by a source-level pin in the style of `studio/backend/tests/test_series_regressions.py` that fails on a route passing a non-`None` second argument. No client builds a `?token=` URL in this checkout — `studio/frontend/src/features/library/api.ts` appends `link=` only when the server minted one, and `studio/frontend/src/components/assistant-ui/sandbox-files-view.tsx` reads through `authFetch` — but neither minting helper has a route behind it (`_sign_download_link` in `studio/backend/routes/library.py`, `_sign_canvas_link` in `studio/backend/routes/inference.py`, called from tests only), so a headerless native save has no credential to carry and the desktop save does not work. |

## Re-running the audit

The greps below are the ones that produced this file. Each is a question about the
checkout, not a pass/fail: run them, read the hits, then update the tables above.

```bash
cd studio/backend

# Any shell-string execution. Expected: no hits in the backend.
grep -rn "shell=True" --include=*.py .

# Pickle-loading weight readers. Every hit must carry weights_only=True on the same
# call; the rest of the count is prose mentions, so read them.
grep -rn "torch\.load(" --include=*.py . ../../unsloth ../../scripts | grep -v __pycache__

# Where inline script is still permitted. The backend's own CSP allows 'unsafe-inline'
# for style only (studio/backend/main.py:1230); every script-src hit is a framed shell
# that needs
# its own justification.
grep -rn "unsafe-inline" --include=*.py --include=*.json --include=*.html ..

# The credentials flag. Exactly one production hit, and it reads False.
grep -rn "allow_credentials" --include=*.py . | grep -v tests/

# Session bearer in a URL. Each hit is the first open item, or a signed link.
grep -rn "?token=" --include=*.py --include=*.ts --include=*.tsx ..
```

Two AST linters cover what a grep cannot: `scripts/lint_risky_loaders.py` fails CI on a
new unsafe-deserialize, dynamic-import, `trust_remote-code` or unpinned-code-fetch call
site, and `scripts/lint_exec_literals.py` does the same for `exec`/`eval`/`compile`. Both
run in `.github/workflows/lint-ci.yml`, with the package scanner behind
`.github/workflows/security-audit.yml`.

```bash
python scripts/lint_risky_loaders.py
python scripts/lint_exec_literals.py
```

The suites that pin the controls above:

| Control | Test file |
| --- | --- |
| The three fixes a prior review asked for (CORS credentials, preview CSP nonce, desktop `base-uri`) | `studio/backend/tests/test_security_hardening.py:25`, `:35`, `:51` |
| CORS origin allowlist per mode | `studio/backend/tests/test_cors_origins.py` |
| Login buckets: per account, per IP, overflow, unknown-user slot | `studio/backend/tests/test_login_rate_limit.py` |
| The shared credential lockout (`token_attempts`) | `studio/backend/tests/test_auth_rate_limit.py` |
| CSP of the framed shells and the artifact preview | `studio/backend/tests/test_artifact_preview_frame_csp.py` |
| Vendored docs bundles: presence and digest | `studio/backend/tests/test_docs_ui_assets.py` |
| Keyless admission, including the adversarial header cases | `studio/backend/tests/test_keyless_api_access.py`, `studio/backend/tests/test_keyless_api_access_adversarial.py`, `studio/backend/tests/test_keyless_tunnel_exposure.py` |
| Preview capability links and per-IP limiting | `studio/backend/tests/test_preview_token.py`, `studio/backend/tests/test_preview_routes.py`, `studio/backend/tests/test_preview_followups.py` |
| `trust_remote_code` consent, nested `auto_map`, the approval cache | `studio/backend/tests/test_consent_gate.py`, `studio/backend/tests/test_nested_auto_map_consent.py`, `studio/backend/tests/test_trc_approval_cache.py` |
| The pickle gate and the Hub card guard | `studio/backend/tests/test_file_security.py`, `studio/backend/tests/test_hub_data_files_guard.py` |
| Direct-local verdict used by owner-only surfaces | `studio/backend/tests/test_sandbox_settings_route.py:376` |
| Body caps, including chunked uploads | `studio/backend/tests/test_middleware.py` |
| Desktop secret and login-mode gating | `studio/backend/tests/test_desktop_auth.py` |
| Bootstrap timeout and the terminal password prompt | `studio/backend/tests/test_bootstrap_timeout.py`, `studio/backend/tests/test_password_prompt.py` |
| Credential encryption: tampering, key loss, restart | `studio/backend/tests/test_credential_secrets.py:104` |

```bash
cd studio/backend
python -m pytest tests/test_security_hardening.py tests/test_cors_origins.py \
    tests/test_auth_rate_limit.py tests/test_login_rate_limit.py \
    tests/test_preview_routes.py tests/test_keyless_api_access_adversarial.py \
    tests/test_file_security.py tests/test_consent_gate.py -q
```

## Host allow-list (`UNSLOTH_TRUSTED_HOSTS`)

The audit's "no Host allow-list" item now has a control, off by default: set
`UNSLOTH_TRUSTED_HOSTS=studio.example,*.internal` and any request whose `Host` matches none of the
patterns is refused with 400. Loopback literals and `localhost` always pass, so enabling it cannot lock
a user out of their own machine, and leaving it unset is the previous behaviour exactly — LAN, tunnel and
Colab modes carry Host values a single list cannot know. WebSocket scopes pass through unchanged.
