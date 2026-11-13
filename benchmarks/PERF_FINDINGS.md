# API performance findings — read-only UI workload

All numbers below come from runs I performed on this machine on 2026-11-12 (local time,
Windows 11 AMD64, Python 3.13.17 — the same machine/Python as `baseline-api.json`).
Nothing in the application was modified by me; every measurement is reproducible from the
commands cited next to it.

## 0. How each measurement was produced

| What | Command / method | Finished at |
|---|---|---|
| Repro run 1 | `python studio/backend/tests/perf_harness/bench_api.py --iterations 15` (from `C:/Projectas/UnslothX-`) | 22:25:10 |
| Repro run 2 | same command | 22:30:55 |
| Repro run 3 | same command | 22:34:43 |
| Profile #1 | `python studio/backend/tests/perf_harness/profile_hotspots.py --iterations 8` | 22:27:59 |
| Profile #2 | same command | ~22:47 |
| DNS / SQLite probes | inline `python - <<EOF` scripts booting the app via the same `TestClient` recipe as `bench_api.py`, wrapping `socket.getaddrinfo`, `sqlite3.connect` and `ResearchRunsSupervisor.note_server_address` with counting timers | ~22:40–22:55 |

**Code state — the tree changed under me mid-session.** Another agent on this pass edited
application source while I measured. File mtimes:

* `studio/backend/utils/host_policy.py` — edited **22:29:40**: added a 60 s-TTL
  `getaddrinfo` cache (`_cached_resolved_host_ip_addresses`) on exactly the request path
  investigated in §4.
* `studio/backend/routes/models.py` — edited **22:31:53**: added a 3 s-TTL memo
  (`_compat_local_inventory_memo`) around the `/api/models/local` inventory scan.

`utils/cache_inventory.py`, `utils/stat_signature.py`, `hub/utils/inventory_scan.py` were
untouched (mtimes 2026-11-09/10), so the top hotspot findings are unaffected. Consequences
for comparability are called out per table below. Per my write constraints, the JSON run
artifacts were written to a scratch dir and removed again; every number is reproduced in
this document.

## 1. Harness reproducibility — **REFUTE** the "within ~15%" expectation

`bench_api.py --iterations 15`, overall statistics (mine vs `baseline-api.json`):

| Metric | baseline | run 1 | run 2 | run 3 | run1 vs base | within 15%? |
|---|---|---|---|---|---|---|
| median_ms | 17.18 | 22.07 | 29.39 | 42.82 | +28.5% | **No** |
| p95_ms | 554.06 | 929.51 | 1144.38 | 2147.91 | +67.8% | **No** |
| mean_ms | 91.58 | 135.43 | 184.99 | 310.51 | +47.9% | **No** |
| max_ms | 929.80 | 1237.31 | 2053.04 | 7149.83 | +33.1% | **No** |

Per-endpoint medians (ms):

| endpoint | baseline | run 1 | run 2 | run 3 | run1 Δ | run3 Δ |
|---|---|---|---|---|---|---|
| /api/health | 12.71 | 14.03 | 16.74 | 24.19 | +10.4% | +90.3% |
| /api/auth/status | 14.14 | 16.71 | 21.75 | 31.31 | +18.2% | +121.4% |
| /api/liveness | 7.10 | 8.88 | 9.75 | 16.16 | +25.1% | +127.6% |
| /api/settings/hub | 12.96 | 19.66 | 21.56 | 36.70 | +51.7% | +183.2% |
| /api/settings/caches | 745.83 | 1027.83 | 1393.27 | 2241.55 | +37.8% | +200.5% |
| /api/settings/multi-model | 16.39 | 21.48 | 25.38 | 40.84 | +31.1% | +149.2% |
| /api/settings/remote-access | 18.25 | 23.46 | 27.86 | 39.49 | +28.5% | +116.4% |
| /api/llama/architectures | 22.08 | 28.13 | 35.43 | 68.71 | +27.4% | +211.2% |
| /api/llama/backend | 47.03 | 58.22 | 66.89 | 121.64 | +23.8% | +158.6% |
| /api/models/local | 260.68 | 401.09 | 496.82 | 718.54 | +53.9% | +175.6% |
| /api/inference/status | 55.90 | 93.98 | 114.72 | 177.75 | +68.1% | +218.0% |
| /api/sandbox/capability | 10.83 | 14.49 | 20.51 | 36.55 | +33.8% | +237.5% |
| /api/profile/stats | 16.36 | 19.82 | 23.67 | 38.09 | +21.1% | +132.8% |

Per-endpoint p95 (ms), run 1 vs baseline: /api/settings/caches 1166.08 vs 921.57 (+26.5%),
/api/models/local 462.85 vs 336.54 (+37.5%), /api/inference/status 109.61 vs 94.45 (+16.1%),
/api/liveness 10.90 vs 8.25 (+32.1%), /api/llama/architectures 48.72 vs 48.98 (−0.5%).

**Verdict: the harness is not stable within 15% on this machine.** Only `/api/health`
(+10.4%) lands inside 15% for run 1; the two disk-heavy endpoints drift +38…+68% even
between runs separated by two minutes. Contributing factors, each observed:

1. **Machine load.** A load check between runs (`Get-CimInstance Win32_Processor`) showed
   60% CPU with no Python process of mine running; other agents work on this host. The
   drift is monotonic over the session (run1 < run2 < run3 everywhere), matching rising
   external load.
2. **Absolute times depend on the real home-directory caches.** `utils/cache_inventory.py`
   roots resolve through `Path.home()` (lines 80–403: HF, npm, torch, triton, … caches), so
   the throwaway `UNSLOTH_STUDIO_HOME` isolates the DB but **not** the disk scans; scan cost
   scales with the developer's real cache trees and OS/antivirus cache state.
3. **The tree changed mid-session** (host_policy.py 22:29:40, routes/models.py 22:31:53), so
   runs 2–3 are not code-identical to the baseline (runs 1 and the baseline are).

What *is* stable: the ranking and the shape of the profile — the same two endpoints and the
same functions dominate every run. **Recommendation for the optimization pass: compare
before/after back-to-back in one session and treat the sum-of-medians (§5) as the metric,
not absolute wall times.**

## 2. Per-request fixed cost

Measured floor = median of the cheapest endpoint `/api/liveness` (near-zero handler):
**7.10 ms** (baseline) → 8.88 / 9.75 / 16.16 ms (my runs, load-inflated). Breakdown of the
fixed machinery, from profile #2 (104 requests, cProfile ≈2–3× inflation, so relative
shares are the signal) and the instrumentation probe:

| Component | Evidence | Per request |
|---|---|---|
| Starlette/FastAPI route matching, linear scan of the route table | `fastapi/routing.py:1728` `_match`: 3,000 calls / 104 req = 28.8/req, cum 1,037 ms | ~10 ms under cProfile (≈2–3 ms real) — the largest identified fixed-cost block |
| `ResearchPortMiddleware` → `note_server_address` | probe wrapper: 1 call per request (39/39), 107 µs/req total; the chain itself 3.6 µs (literal IP) | 0.1 ms |
| SQLite connection opens | `sqlite3.connect` 415 calls / 104 req ≈ 4.0/req (probe: 0 for liveness, 8 for models/local; 41 opens per 13-endpoint poll cycle); `execute` self-time 1.29 s / 104 req | ~12 ms under cProfile |
| TestClient/asyncio plumbing | `asyncio` `_poll` 4,612 calls ≈ 44/req | inside the 7–16 ms floor |

## 3. Ranked hotspots

Profile #2 (`profile_hotspots.py --iterations 8`, current code, 22:47; profile #1 at 22:28
gave the same shape). "Measured ms" = per-request cost from that run unless noted;
baseline column = un-profiled endpoint median from `baseline-api.json`.

| Rank | endpoint / function | file:line | measured ms | root cause (redundant work) | cheapest safe optimization | risk |
|---|---|---|---|---|---|---|
| 1 | `/api/settings/caches` → `cache_inventory` → `_described` → `_roots_unchanged` → `is_unchanged` → `directory_signature`/`_collect`/`_scan_sorted` | `utils/cache_inventory.py:1070,1053,1104`; `utils/stat_signature.py:181,147,86,72` | **745.8 ms** baseline median (60.1% of a poll cycle); 1,677 ms/call under cProfile (13,412 ms cum / 8 calls) | **Disk scan**: even a TTL-fresh memo is validated by a full recursive stat-signature walk of every cache root on every read. Measured per request: 4,234 `nt.lstat` (6.16 s/8 req), 2,545 `nt.scandir` (5.77 s), 2,573 `sorted` (2.39 s), 15 `directory_signature` walks (120 walks / 8 requests). The "bounded" check (`_roots_unchanged`) still walks ~19,640 dir entries per request | Replace the per-read root re-signature with one stat per root (root dir mtime + entry count) or give the signature the same 60 s TTL as the size memo | cache sizes/flags can display stale up to TTL after out-of-band cache changes (display-only data) |
| 2 | `/api/models/local` → `collect_local_models` → `_scan_hf_cache` → `is_snapshot_partial`/`_compose_partial` + per-file `Path.resolve()` | `routes/models.py:946,606`; `hub/utils/inventory_scan.py:1393,510` | **260.7 ms** baseline median (21.0% of cycle); 622 ms/call under cProfile (3,733 ms / 6 calls) | **Disk scan + redundant canonicalization**: per-snapshot GGUF header re-reads and 66 `nt._getfinalpathname` realpath calls per request (6,902 calls, 2.17 s / 104 req) resolving paths that have not changed. The in-flight 3 s-TTL memo (`routes/models.py:1210-1280`, added 22:31:53) caught only 2 of 8 calls in my run — request spacing exceeds its TTL | Key the memo on the snapshot dirs' stat signature instead of wall-clock TTL; use `absolute()` where the realpath is not semantically needed | model list stale up to TTL after on-disk change (same exposure the memo already accepts) |
| 3 | all endpoints — SQLite connection churn | `storage/sqlite_gate.py:42`, `auth/storage.py:275` | ~4.0 opens/req, 41 opens/poll cycle (probe); `_sqlite3.connect` self 0.43 s + `execute` self 1.29 s / 104 req under cProfile | **Repeated DB open**: every handler opens its own short-lived connection through the gate (`/api/models/local` 8 opens, `/api/settings/caches` 6, `/api/inference/status` 5, `/api/llama/backend` 4; `/api/liveness` 0) | Reuse one connection per request/account behind the existing gate, or a small per-thread pool | thread-safety — must keep the gate's serialization; a long-lived connection can miss external DB writes |
| 4 | all endpoints — route matching | `fastapi/routing.py:1728` (`_match`) | ~10 ms/req under cProfile (28.8 calls/req; ≈2–3 ms real) | **Redundant per-request scan**: every request linearly re-matches against the entire route table until the handler's route is reached | Mount the large routers so the hot paths match early (Starlette first-match order); keep `/api/*` behind one mount | route shadowing / precedence changes need a route-table regression check |
| 5 | `/api/llama/backend` → `llama_backend_status` | `routes/llama.py:385` | **47.0 ms** baseline median | threadpool hop into `get_backend_status` (installer probe) + 4 SQLite opens per call; only partially covered by its 24 h resolver cache | Raise the resolver-cache hit rate (key it on the hardware snapshot) | backend status stale until cache expiry (already accepted for 24 h) |
| 6 | all endpoints — per-request host resolution (see §4) | `main.py:1181` → `core/research_runs.py:1444` → `utils/host_policy.py:242,204,125,114,98,54` | uncached: **443.9 µs/req median, 1.37 ms p95, 18.2 ms max** (my probe); 0 µs extra in harness config (literal IP) | **Per-request DNS** on the event loop for non-literal scope hosts (mechanism confirmed in §4). A 60 s-TTL cache landed mid-session (host_policy.py 22:29:40); cached chain now 7.1 µs median with 0 `getaddrinfo` over 200 calls | already mitigated; residual cheap fix: resolve once per listener startup (the bind host cannot change per request) instead of on the request path | none material — `clear_resolution_cache()` already covers in-process host changes |
| note | threadpool hops (`asyncio.to_thread`) | `routes/settings.py:1485`, `routes/models.py:1326`, `routes/llama.py:393` | 104 `concurrent.futures.thread.run` calls, 15.5 s cum / 104 req (includes work, not overhead) | The hop itself is **correct** — it keeps the disk scans off the event loop | none — reduce the work (ranks 1–2), not the hop | removing the hop would block the loop: regression |

## 4. Specific hypothesis — per-request `getaddrinfo` via `note_server_address`

**Claim:** `scope_request_host` → `wildcard_loopback_host` → `wildcard_ip_versions` →
`_resolved_ip_addresses` performs a blocking `socket.getaddrinfo` on the event loop from
`core/research_runs.py` `note_server_address`, once per HTTP request, for any scope server
host that is not a literal IP.

**Verdict: CONFIRMED (with two scope conditions), and a fix landed mid-session.**

Evidence, all from my runs:

1. **Once per request — confirmed.** `ResearchPortMiddleware` (`main.py:1173-1183`) calls
   `supervisor.note_server_address(scope.get("server"))` synchronously inside the async ASGI
   middleware. A counting wrapper measured **39 `note_server_address` calls for 39 requests**
   (3 iterations × the 13-endpoint workload). Starlette's TestClient sets
   `scope["server"] = [host, port]` from the request URL host (verified in
   `starlette/testclient.py::_TestClientTransport.handle_request`), so the scope host is
   exactly what the claim describes.
2. **Blocking `getaddrinfo` for non-literal hosts — confirmed.** Calling the chain with
   `"localhost"` while clearing the (new) resolution cache before every call — i.e. the
   behavior of the code before 22:29:40 — gave **100 `getaddrinfo` calls for 100 chain
   calls**, cost per request-path call: median **443.9 µs**, mean 745.2 µs, p95 1.37 ms,
   **max 18.2 ms** (resolver warm). A single failure-path lookup
   (`_resolved_host_ip_addresses("nonexistent-host-xyz.invalid")`) took **13.0 ms**; with a
   cold/timeout resolver this call has no upper bound while it blocks the event loop (the
   in-tree comment on the fix cites 2.29 s in such a case — their measurement, not mine).
3. **Literal-IP short-circuit — confirmed.** `scope_request_host(("127.0.0.1", 5001))`:
   0 `getaddrinfo` calls in 200 chain calls, median **3.6 µs**. Across the full workload
   with the harness's `base_url http://127.0.0.1:5001`: **0 `getaddrinfo` calls in 39
   requests**. So the hypothesis is *refuted as a cost in the harness configuration as
   written* (it deliberately uses a literal loopback address) and confirmed only for
   non-literal binds (`--host localhost`, container hostnames, `testserver`).
4. **Second condition.** `note_server_address` skips the chain entirely once
   `app.state.server_request_host` is published; `run_server` publishes it
   (`run.py:2989`). So the per-request DNS only fires when the app is served *without*
   run_server's publish path (plain uvicorn/TestClient/embedders) **and** the scope host is
   not a literal IP.
5. **Current code (after 22:29:40).** `_cached_resolved_host_ip_addresses` (60 s TTL) makes
   the chain median **7.1 µs** with 0 `getaddrinfo` calls over 200 invocations (1 call per
   TTL window). The hypothesis cost is gone in practice; the residual architectural issue
   (resolution on the request path at all) remains, see rank 6 in §3.

## 5. Recommendation — one primary metric and target

**Primary metric: one full UI poll cycle = the sum of the 13 per-endpoint medians of the
fixed workload.** (The all-sample `median_ms` of 17.18 ms is a poor target metric: 8 of 13
endpoints are cheap, so it barely moves when the disk endpoints improve.)

* **Baseline: 1,240.26 ms** (= Σ `per_endpoint_median_ms` of `baseline-api.json`:
  745.83 + 260.68 + 55.90 + 47.03 + 22.08 + 18.25 + 16.39 + 16.36 + 14.14 + 12.96 + 12.71 +
  10.83 + 7.10).
* **26% target: 1,240.26 × 0.74 = 917.79 ms**, i.e. remove **322.47 ms** per cycle.
* Arithmetic against measured shares: `/api/settings/caches` alone is 745.83 ms = 60.1% of
  the cycle. **Halving that one endpoint (745.83 → 372.92 ms) removes 372.91 ms = 30.1%**,
  overshooting the 26% target by itself — and rank 1's root cause (a ~19,640-entry stat
  walk repeated on every read purely to validate an unchanged memo) is exactly the work a
  cheap root-stat check would remove. Alternatively, the conservative bundle
  `caches −25% (−186.46) + models/local −50% (−130.34) + llama/backend −50% (−23.52) =
  −340.32 ms = 27.4%` also clears the bar.
* Evaluation protocol given §1's instability: run the harness before/after back-to-back in
  one session, compare the sum of medians, and require the cheap-endpoint floor
  (`/api/liveness` median ≈ 7 ms) not to regress.

## Wave 2 — cache inventory signature check

Hotspot #1 from §3 (`/api/settings/caches` → `_described` → `_roots_unchanged`), fixed in
`studio/backend/utils/cache_inventory.py` with regression tests in
`studio/backend/tests/test_cache_inventory_caching.py`. `utils/stat_signature.py` was not
needed and is untouched. Every number below comes from runs I did on this machine
(Windows 11 AMD64, Python 3.13.17) with the throwaway script of §"How measured" below; the
machine-load drift of §1 applies, so the before/after comparison is the same script run
against the two code states back to back, and the metric is the warm repeat read.

### What changed

The 60 s inventory memo stays. The epoch/generation invalidation (`_size_epochs`,
`invalidate_cache_size`, `invalidate`) and the `refresh=True` bypass are unchanged. On top
of the memo there is now a **5 s signature-check window** (`_SIGNATURE_CHECK_TTL_SECONDS`):
inside it `_described()` serves the remembered entry without re-walking
`directory_signature()` per root; past it the trees are proven exactly as before
(`_roots_unchanged`, bounded stat walk per root) and re-measured on mismatch. A passing
proof re-arms the window, and only a real proof may re-arm (a window hit must never extend
its own window), so a steady poll proves at most once per window instead of once per read.
Served entries are now deep copies (`copy.deepcopy` per entry), so a response model editing
a row cannot poison the memo. The memo tuple gained one slot (the proof timestamp); it is
one entry per cache key, so the added state is bounded by `CACHE_DEFINITIONS`. Nothing else
in the module changed.

Why the window is not wrong and what it costs the user (full comment above the constant):
the figures are display-only metadata (bytes and entry counts), nothing acts on them, and
every acting path sits outside the window — a purge drops the memo through
`invalidate_cache_size` (epoch-guarded) regardless of proof freshness, and the Recheck
button passes `refresh=True`, which walks unconditionally. The observable staleness is one
thing: after a change made **outside** this app, the sizes card can show old figures for up
to 5 s. Two design points were required for the window to fire at all, both measured:

* **Anchor at when the figures were produced (the store instant), not when the walk
  began.** My first attempt anchored at `begun` and made repeated reads no cheaper: a cold
  sizing call takes 27–130 s on this cache (see the table), so the window was consumed by
  the walk before the entry was even stored. Anchoring at the store also bounds staleness
  tighter (the figures describe the tree as of the walk's end).
* **5 s > one proof round.** A root proven at the start of a `cache_inventory()` round is
  only revisited after the round (~1.5–2 s measured warm), so a window shorter than one
  round could never fire. 5 s is a small number of seconds with headroom over the round;
  if a round ever outlives it, the throttle stops firing and every read proves as before —
  the failure mode is the old cost, never a stale answer.

### Before / after — `cache_inventory()` timed in one process

How measured: a throwaway script (`bench_cache_inventory.py`, written under
`C:/Users/nutti/AppData/Local/Temp`, **not committed**) that times five sequential
`cache_inventory()` calls in one process and counts `directory_signature` walks per call by
monkeypatching both bindings (`utils.cache_inventory.directory_signature` and
`utils.stat_signature.directory_signature`, the latter being what `is_unchanged` resolves).
It resolves the real cache roots (10.6 GB HF cache, uv/pip/npm/… , 16 definitions, 9
resolvable roots on this machine).

| call | before run 1 (ms) | before run 2 (ms) | after (ms) | sig walks before | sig walks after |
|---|---|---|---|---|---|
| 1 (cold) | 96,460.6 | 40,285.9 | 27,109.9 | 9 (takes) | 9 (takes) |
| 2 | 44,534.7 | 1,781.3 | 2,751.3 | 9 | 3 (re-proofs of caches whose window expired mid-call) |
| 3 | 1,990.4 | 1,938.3 | 1.9 | 9 | 0 |
| 4 | — | 1,682.7 | 1.8 | 9 | 0 |
| 5 | — | 1,400.6 | 1.8 | 9 | 0 |

* **Before: warm repeats never got cheaper.** Calls 3+ kept paying ~1.4–2.0 s and exactly
  9 signature walks per call, confirming the §3 diagnosis: the 60 s memo sat warm and
  unused because `_roots_unchanged()` re-proved every root on every read. (Before run 1's
  44 s call 2 is OS-cache warming of the disk walks, not memo behaviour.)
* **After: a repeated read inside the window is 1.8–1.9 ms with 0 signature walks** —
  `deepcopy` of 16 small entries plus the sums.
* **Warm repeat read: 1,810 ms before → 1.85 ms after = −99.9 % (≈980×).** Before is the
  median of the four warm repeats across both before-runs (1,990.4 / 1,938.3 / 1,682.7 /
  1,400.6 ms); after is the three window-served repeats (1.9 / 1.8 / 1.8 ms).
* Cold calls (call 1) are unchanged work — the 60 s-TTL sizing walk, out of scope here.
* Reads spaced further apart than the window still prove the trees as before ("verify as
  today, re-measure on mismatch"); the win is for back-to-back and bursty reads, which is
  what made this endpoint 60 % of a poll cycle. Call 2 after shows the amortized shape:
  one proof round per window, then free reads until the window lapses.

### Tests that pin the behaviour

All in `studio/backend/tests/test_cache_inventory_caching.py`, following the module's
existing style (the `one_cache` fixture monkeypatches `CACHE_DEFINITIONS`, the memo, and
counts sizing walks; the new `sig_walks` fixture counts signature walks at both bindings).
Seven new tests:

1. `test_two_reads_inside_the_signature_window_prove_the_tree_once` — two sequential
   non-refresh reads inside the window cost ONE signature walk (the cold take) and one
   sizing walk.
2. `test_past_the_signature_window_the_tree_is_proven_again` — past the window a signature
   walk happens again, and a passing proof does not re-measure.
3. `test_a_forced_read_walks_even_inside_the_signature_window` — `refresh=True` always
   walks (the Recheck button contract).
4. `test_a_changed_tree_is_rememeasured_in_the_windows_successor` — a tree changed inside
   the window is re-measured once the window's successor proves it (mismatch → fresh
   figures 10 → 15 bytes), and the successor window is re-armed by that walk.
5. `test_a_purge_still_invalidates_inside_the_signature_window` — `purge_caches` empties
   the tree and the post-purge inventory re-measures to 0 even inside a fresh window.
6. `test_a_read_returns_a_copy_not_the_memo` — the returned entry is not the cached object;
   mutating `size_bytes`/`entry_count`/`paths`/`blocked_reason` on a response does not
   poison later reads and forces no walk.
7. `test_the_response_shape_is_the_same_with_and_without_the_window` — the response keys
   (`caches`, `total_bytes`, `reclaimable_bytes`, `free_bytes`, `total_disk_bytes`) and
   entry keys (`key`, `group`, `opt_in`, `paths`, `size_bytes`, `entry_count`, `present`,
   `purgeable`, `blocked_reason`) are unchanged cold vs window-served, and every field
   survives the windowed serve.

One existing test was adapted, not weakened:
`test_a_file_added_between_reads_walks_again` — the change now lands inside the window
where the memo is served on purpose, so the test closes the window (sets
`_SIGNATURE_CHECK_TTL_SECONDS` to 0, which is what expiry does) before asserting the
re-measure; its assertions are otherwise unchanged.

### Pytest output summary

```
$ python -m pytest studio/backend/tests -q -k "cache_inventory or stat_signature" \
      --continue-on-collection-errors
14 failed, 253 passed, 125 skipped, 71817 deselected, 86 warnings, 71 errors in 237.03s
```

* Selected by the filter: 267 tests in 6 modules (`test_cache_inventory.py` 96,
  `test_cache_inventory_path_boundary.py` 146, `test_cache_inventory_caching.py` 10,
  `test_stat_signature.py` 10, plus 5 in `test_text_io_encoding.py` /
  `test_hf_cache_dangling_refs.py`). **253 passed / 14 failed**; the 125 skips and 71
  collection errors are pre-existing environment issues in UNRELATED modules (this host
  has no `torch`, no symlink privilege, ...) - the same run without
  `--continue-on-collection-errors` aborts at collection before executing anything.
* Focused run of the four cache modules, after vs pristine: **248 passed / 14 failed**
  vs **241 passed / 14 failed** (pristine restored via `git stash`, run back to back).
  The +7 are exactly the new tests.
* The 14 failures are identical, byte for byte, on pristine and on this change (diff of
  the FAILED lists = empty). They are environmental on this Windows host - e.g.
  `test_sizing_does_not_follow_a_symlink_out_of_the_cache` dies in the fixture with
  `OSError: [WinError 1314] A required privilege is not held by the client` creating the
  symlink - and are untouched by this change.
