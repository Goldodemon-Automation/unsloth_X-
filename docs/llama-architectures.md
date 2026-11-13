# llama.cpp architectures (Settings → General)

The Studio can install and switch the llama.cpp runtime that GGUF inference runs on. Six rows,
one app setting, one install directory per fork. This is a runtime switch, not a model option:
whatever row is active is the `llama-server` every GGUF load starts.

The manifest and installer live in `studio/backend/utils/llama_cpp_architectures.py`, the routes
in `studio/backend/routes/llama.py`, the picker in
`studio/frontend/src/features/settings/components/llama-backend-section.tsx`, and the setting is
read at launch by `studio/backend/core/inference/llama_cpp.py`
(`LlamaCppBackend._find_llama_server_binary`).

## The rows

| Id | Installs from | What it is here |
| --- | --- | --- |
| `normal` | — (bundled) | The llama.cpp runtime Unsloth installs and updates. The default; this panel never downloads it. |
| `dory` | `brianchmielNV/llama.cpp` | Its change is an unmerged llama.cpp PR (`llama_cpp_architectures.py:101`), so the repo publishes no releases: installable only by compiling the pinned commit. |
| `bee` | `Anbeeld/beellama.cpp` | BeeLlama.cpp. The installer accepts its CPU bundles for Windows/Linux x64, Ubuntu arm64 and macOS. |
| `neurall` | `neurall/llama.cpp` | Neuralls llama.cpp. Only its combined CUDA bundles for Windows and Ubuntu x64 match, never the `cudart-*` runtime archive beside them. |
| `ik` | `Thireus/ik_llama.cpp` | ikawrakow's fork; that repo attaches no binaries, so Thireus' prebuilt builds are the install source, CPU and AVX2 first. |
| `x` | — | X- LLama.zig has no published build. The row exists and is disabled (`kind = "not_ready"`, `reason = "build_not_ready"`) rather than hidden. |

The ids come from `ARCHITECTURES` (`utils/llama_cpp_architectures.py:106`) and are mirrored by
the frontend's `LLAMA_ARCHITECTURES`
(`studio/frontend/src/features/settings/api/llama-architectures.ts:7`); a profile whose id the
frontend does not know is dropped from the parsed payload rather than guessed at.

Nothing in this checkout says what workload each fork is *for*. The module's own `description`
fields are one sentence each ("The BeeLlama.cpp fork."), and this doc does not invent more than
that. Choosing a fork because you need a specific model capability is a decision no file here
supports.

## Surface and API

The picker is one row, "Llama.cpp architecture", inside the **GGUF inference engine** section of
Settings → General, rendered only for the owner
(`studio/frontend/src/features/settings/tabs/general-tab.tsx:505`). The section reads the status
payload on open and polls it every 900 ms while an install job runs, giving up after 20
unanswered checks so a dead endpoint cannot hold the row on "Installing"
(`llama-backend-section.tsx:92`, `:97`, `:110`). Selecting a row activates an installed fork,
refuses an unavailable one with that row's own reason, and — for an installable one — asks for
confirmation before downloading a third-party binary (`llama-backend-section.tsx:363`, `:631`);
the confirmed install posts with `activate: true` (`:328`).

| Route | Who | Returns |
| --- | --- | --- |
| `GET /api/llama/architectures` | any signed-in subject | `active`, `runtime_locked` + `runtime_lock_reason`, `release_lookup_failed`, `profiles[]`, `job` (`routes/llama.py:414`) |
| `POST /api/llama/architectures/install` | owner (`routes/llama.py:442`) | `started`, `reason`, `message`, `job` — the work runs on a worker thread and the caller polls |
| `POST /api/llama/architectures/activate` | owner (`routes/llama.py:464`) | the new status; a `ValueError` from the module becomes `409` with its message (`routes/llama.py:474`) |

Both POST bodies are `{architecture, activate}` with `activate` defaulting to true
(`routes/llama.py:363`). An empty or over-long id is a `422`. An unknown id is a `404` on
**install** (the message lists the ids the manifest knows) and a `409` on **activate** — both
refusals, so a client that checks only the status code cannot read a rejected request as a
started install. A *known* id that cannot be built on this host is not an error: install answers
`200` with `started: false` and a `reason`, because that is a capability answer rather than a
bad request (`studio/backend/tests/test_llama_architecture_routes.py`).

The status read goes through `get_architecture_status`
(`studio/backend/utils/llama_cpp_update.py:1774`): a 10 s payload cache keyed on the active
setting, the job state, the runtime-path environment, the toolchain, the manifest and each
install marker's identity, with a 3 s wall-clock deadline on the release fetch
(`llama_cpp_update.py:1528`, `:1539`). The job half is read fresh on every call, so progress
keeps moving while the rest is cached, and this path never takes the install lock.
`force_refresh=true` bypasses it.

## How an install works

1. **Request.** `start_install` (`llama_cpp_architectures.py:1498`) refuses before anything
   starts: unknown id (`unknown_architecture`), `normal` (`bundled` — it needs no install),
   X- (`build_not_ready`), a runtime pinned by `LLAMA_SERVER_PATH` or `UNSLOTH_LLAMA_CPP_PATH`
   (`environment_override`), or another install/switch holding the single operation lock
   (`already_running`). Otherwise it takes that lock, resets the job to `running`, and starts a
   daemon thread; the route returns immediately.
2. **Release lookup.** `GET https://api.github.com/repos/<repo>/releases/latest`
   (`_fetch_release`, `:370`). A `404` means "this repo has no releases" and is not an error;
   any other failure is memoised for 45 s and the repo flagged (`:405`, `:55-56`). A successful
   answer is cached for 600 s (`:55`). `GH_TOKEN`/`GITHUB_TOKEN`, when set, travels as a
   `Bearer` (`:326`). The read itself is bounded: 8 s and a 4 MiB body cap (`:81-82`, `_read_json`
   at `:338`).
3. **Host and asset matching.** `_host` (`:198`) reduces the machine to
   `windows|linux|darwin` × `x64|arm64`. `_score_asset` (`:235`) scores each release asset's
   filename per fork — CPU-first for IK (a GPU asset would need a driver and a runtime the fork
   does not bundle), the combined CUDA bundle for Neurall, CPU bundles for Bee — and
   `_choose_asset` (`:311`) takes the highest score. No match is a normal outcome, not an error:
   the row then carries `reason: "no_build_for_platform"`.
4. **Download.** `_download_asset` (`:804`): `https` only; GitHub's declared size must be within
   the 2 GiB cap before a byte is requested; after redirects the final URL must still be GitHub
   over TLS (host `github.com` or `*.githubusercontent.com`, `:718` and `:743`); the transfer
   runs against a 1-hour whole-transfer deadline and the same byte cap; a byte count that
   differs from GitHub's declared size fails as an incomplete download; when the asset carries
   a digest it is compared against the bytes actually received (`:771`).
5. **Extraction.** `_extract_archive` (`:1088`) writes member by member rather than calling
   `extractall`. Every member name must resolve inside the destination — no absolute path, no
   `..`, no drive letter or colon, no NUL (`_member_parts` `:887`, `_safe_target` `:915`) — tar
   link targets are resolved and must stay inside too (`:929`), zip symlinks and device/FIFO
   members are refused, duplicate member names are refused, the member count is capped at
   200 000 and the expanded bytes at 4 GiB, and setuid/setgid/sticky bits are dropped.
6. **Binary discovery and marker.** `_find_binary` (`:1117`) takes the shallowest
   `llama-server`/`llama-server.exe` in the tree; an archive without one fails with "does not
   contain llama-server". `_promote_install` (`:1130`) then renames the prepared tree into
   `<install root>/<id>` (the previous tree is kept as `.old-<id>-<uuid>` and restored if
   anything fails) and writes `UNSLOTH_LLAMA_ARCH_INFO.json` beside it recording `architecture`,
   `label`, `repo`, `binary_relpath`, `installed_at_utc`, `release_tag`, `asset`, `digest`,
   `source_build` and `source_commit`. The marker is what later reads treat as "installed".
   The staging directory is removed whatever happens (`:1177`), so a failure before promotion
   leaves the previous install untouched.
7. **Activation.** With `activate` true (the UI always sends true) the job writes the app
   setting `llama_cpp_architecture` (`:51`, `_set_active` at `:552`) after the marker exists.
   An activation failure does not fail the install: the job ends `success` with
   `activated: false` and a "Could not switch to it: …" suffix (`_run_install` `:1405`), and the
   UI shows that as an error toast while the install stands.
8. **Progress.** One module-level job dict carries `state`, `message`, `progress` (0…1),
   `from_tag`, `to_tag`, `activated` and timestamps; the picker polls until `state` leaves
   `running`. While a job runs, `get_status` skips the release lookups entirely
   (`llama_cpp_architectures.py:677`), so a progress poll pays neither threads nor network.

Install root: `$UNSLOTH_STUDIO_HOME` or `$STUDIO_HOME` if set, else `unsloth_home()`, else
`~/.unsloth`, plus `llama.cpp-architectures/<id>` (`:441`).

### The source-build fallback

Only **dory** has one. The pinned commit is `1c3cdd23787750814aa21a070dc4d6dc451af871`
(`:118`) — the Dory PR is unmerged, so the branch head is the only source that contains it
(`:101`), and the pin is what makes the build reproducible.

It runs when the repo has no release or no asset matches this host. Requirements
(`_source_build_tools`, `:213`): `cmake` on every platform, plus a C/C++ compiler
(`cc`, `gcc`, `clang`, `c++`, `g++` or `clang++`) on non-Windows; Windows relies on CMake
locating Visual Studio itself. Without them the row reports `build_tools_missing` and is not
installable.

The source tarball is fetched from `https://github.com/brianchmielNV/llama.cpp/archive/<commit>.tar.gz`
with **no digest and no declared size** (`:1365`–`:1367`): TLS and the GitHub origin check are all it
gets. Configure is static, no tests or examples, `LLAMA_BUILD_SERVER=ON`, `GGML_NATIVE=ON`,
`LLAMA_CURL=OFF`, Release, plus `/NODEFAULTLIB:LIBCMT` on Windows; the build target is
`llama-server` with `UNSLOTH_LLAMA_BUILD_JOBS` parallelism (clamped to 1–32, default
`min(8, cores)`) (`_run_source_build`, `:1286`). Configure is capped at 900 s and the build at
3600 s; both are argument lists with `shell=False` (`:1247`), and a failure appends the last 40
lines of the log to the job error. The marker records `release_tag: "source-1c3cdd23"` and
`source_build: true`, which is what later "update available" comparisons use when no asset
matches the machine.

## What a row's status means

Per-profile flags: `available`, `installable`, `selectable`, `installed`, `active`,
`update_available`, plus `reason`. The reasons the code produces:

| `reason` | Meaning |
| --- | --- |
| *(absent)* | An asset matched this host, or the row is bundled |
| `build_not_ready` | X- LLama.zig: no published build, row disabled on purpose |
| `source_build` | Dory, no matching asset, toolchain present: install means compile |
| `build_tools_missing` | Dory without CMake (or without a compiler off Windows) |
| `no_release` | The repo answered 404: no releases published |
| `no_build_for_platform` | Releases exist, none matches this host |
| `release_lookup_failed` | The lookup failed (offline, rate-limited, timed out); only set on rows **without** a source fallback, so Dory still shows `source_build` |
| `missing_binary` | Marker present, binary gone: the row is broken, not uninstalled, and re-installing is the fix |

A top-level `release_lookup_failed` flag on the status payload says the same thing about the
whole panel (`routes/llama.py:350`).

## Failure and degraded states

- **No network.** Every lookup fails into the 45 s memo; rows lose `latest_tag` and
  `download_size_bytes`, and uninstalled forks become non-selectable because there is nothing
  to install yet. Installed forks stay installed, selectable and activatable: an offline
  machine is degraded, not emptied, and a lookup failure is never reported as "this fork has
  no release". The status layer additionally abandons a wedged lookup after 3 s instead of
  waiting it out. Dory still offers the source build offline — and then fails at the tarball
  fetch with "Could not download the build: …".
- **Gated or private repo.** Only `404` is read as "no releases"; a `401`/`403` raises and
  becomes a failed lookup, i.e. the same `release_lookup_failed` row as an outage. The code
  sends `GH_TOKEN`/`GITHUB_TOKEN` to `api.github.com`; whether a given token can read a given
  fork's repo is between the user and GitHub and is not tested here.
- **Unsupported host.** No asset scores for this host → `no_build_for_platform`, row not
  installable and (unless already installed) not selectable. The UI disables the option and
  says "No build for this computer has been published yet."
- **Partial or truncated download.** Byte count differing from GitHub's declared size fails
  before extraction ("The download was incomplete: expected X bytes, received Y."); a drip-feed
  server hits the 1-hour deadline; an oversized asset hits the 2 GiB cap. In every case the
  staging directory is deleted and the previous install is untouched.

## Security posture

What is verified, and what it stops. Every row cites code that was read; a row that stops
describing the code is a bug in this file.

| Control | Where | What it stops — and the limit |
| --- | --- | --- |
| The digest GitHub publishes is compared against the received bytes | `llama_cpp_architectures.py:804`, `:771` | Corrupted or swapped bytes. **Limit:** only when GitHub publishes one; many older assets publish none, and then no content hash is checked at all. |
| A digest in an algorithm that cannot be reproduced, or a malformed one, refuses the install | `:771` | An `md5:` digest silently passing as "verified" while nothing was compared. |
| The final URL after redirects must be `https` on `github.com` or `*.githubusercontent.com` | `:718`, `:743` | A redirect to an attacker-chosen host, or a downgrade to plain HTTP — for digest-less assets this is the only origin guarantee. |
| 2 GiB transfer cap, 4 GiB extraction cap, 200 000 members | `:69`, `:72`, `:83` | A hostile or misconfigured asset filling the disk, and a compression bomb expanding until it does. |
| 60 s socket timeout, 1-hour whole-transfer deadline, 8 s API read deadline, 4 MiB API body cap | `:73`, `:78`, `:81-82`, `:338` | A stalled or dripping response pinning the install thread — and the install lock — open until the process restarts. |
| Every archive member and link target is resolved against the destination before it is written | `:887`, `:915`, `:929`, `:982`, `:1025` | Path traversal, backslash and drive-letter names, symlink/hardlink escape, device and FIFO entries, duplicate-name overwrite; setuid bits are dropped (`_member_mode`, `:973`). |
| Subprocesses are argument lists with `shell=False`, each with a timeout | `:1247`, `:1286` | The repo name or commit reaching a command interpreter. |
| A marker is written only after a complete tree is in place; staging is always removed; a failed promotion restores the previous install | `:1130`, `:1177` | A half-extracted tree being recorded as installed. |
| Install and activate are owner-only; status needs a signed-in subject | `routes/llama.py:442`, `:464`, `:414` | Another account on the same server replacing the runtime. |
| One operation lock for install and switch; activate refuses while one runs | `:1498`, `:1558` | A switch landing between the two directory renames an install performs. |
| A runtime path pinned by environment disables install and activate | `:538` | Fighting an operator's `LLAMA_SERVER_PATH`/`UNSLOTH_LLAMA_CPP_PATH` pin from the UI. |

What is **not** verified, and what you accept by installing:

- **No digest, no content check.** For an asset GitHub reports no digest for, the bytes are
  bounded in size and origin but never hashed against anything. TLS to GitHub's CDN is the
  whole guarantee.
- **The Dory source tarball is never hashed**: the fetch is built with `size: None` and
  `digest: None` (`llama_cpp_architectures.py:1365`–`:1367`).
- **No signature, no provenance, no review.** The bytes come from a third-party repository's
  release (or its pinned commit) and Studio then executes them as `llama-server` on this
  machine. Installing a fork is accepting that repository's code at the tag you installed;
  Unsloth does not vouch for it. The pinned commit makes the Dory build reproducible, not
  reviewed.
- **TLS verification is left to Python's defaults.** No custom SSL context is created here, and
  none is disabled.
- **A token, if `GH_TOKEN`/`GITHUB_TOKEN` is set, is sent to `api.github.com`** (`:326`) as a
  `Bearer`. The download request itself carries only a `User-Agent`.

## Runtime resolution order

The order is decided in one place:
`LlamaCppBackend._find_llama_server_binary` (`studio/backend/core/inference/llama_cpp.py:10143`,
search order documented at `:10147`). It runs for every GGUF load, embeddings and voice:

1. `LLAMA_SERVER_PATH` — a direct path to a binary (`:10233`).
2. `UNSLOTH_LLAMA_CPP_PATH` — a custom install directory, unless
   `UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH=1` says the desktop set it itself (`:10242`).
3. **The active architecture installed from Settings** (`:10263`).
4. The custom llama.cpp folder stored in Settings (`:10299`).
5. The installer layout (`$STUDIO_HOME/llama.cpp` or `~/.unsloth/llama.cpp`), then the legacy
   in-tree build, `llama-server` on `PATH`, and legacy `bin/` (`:10321`).

So: **environment pins beat an activated architecture; an activated architecture beats the
stored custom folder; the custom folder beats the bundled runtime.** Switching architectures
never rewrites the custom-folder setting — activation writes only `llama_cpp_architecture`, so
switching back to `normal` restores whatever custom folder you had. The UI says the same thing
above the custom-path row while a fork is active ("An installed architecture takes priority
over the custom llama.cpp folder below. Reload the model to use it.",
`studio/frontend/src/i18n/locales/en.ts:3238`).

An active architecture is an app-owned install, not a user-maintained pin: if its marker or
`llama-server` goes missing, discovery logs a warning naming the architecture and the Settings
remedy, then continues to the custom folder or the bundled runtime
(`llama_cpp.py:10278`–`:10310`). Returning `None` there used to turn every load into the generic
"run `unsloth studio setup`" failure, which could not repair a fork install or clear the pin; the
fallback is what lets a load still use a working runtime. The picker reports the same broken
install as `missing_binary` and offers a reinstall. `active_architecture_directory`
(`llama_cpp_architectures.py:519`) returns `None` for `normal` and for an unusable install, but
the resolver treats that as a reason to warn and continue, not as a hard stop.

While a fork is active and an environment pin exists, `runtime_locked` is true, the picker is
disabled, and install/activate refuse (`:538`) — the environment would win at step 1 anyway.

## When something does not work

### "Could not check the latest build. Check your connection and try again."

**Cause:** GitHub was unreachable, rate-limited, or slow when the row resolved. The failure is
memoised for 45 s (`:56`) and the status layer gives up on a wedged lookup after 3 s
(`llama_cpp_update.py:1539`), so polls do not pile up behind it.
**Fix:** check connectivity; `force_refresh=true` on `GET /api/llama/architectures` retries
immediately. Installed forks stay usable throughout; nothing is lost while the lookup fails.

### "Dory builds from source, but CMake is not available on this machine."

**Cause:** `_source_build_tools` (`:213`) found no `cmake`, or — off Windows — no C/C++
compiler beside it. The check runs on every status read, so it also explains why the row never
becomes installable.
**Fix:** install CMake (and a compiler on Linux/macOS), reopen Settings. Configure and build
timeouts are 900 s and 3600 s, so a slow machine can still legitimately time out with the log
tail in the job error.

### "No build for this computer has been published yet. Nothing can be installed."

**Cause:** releases exist but `_score_asset` (`:235`) matched none for this host — the fork
publishes for other platforms or only GPU builds you cannot use. This is a real answer about
the repository, not a network failure (`release_lookup_failed` would say that instead).
**Fix:** nothing to install on this machine. Use `normal`, or a custom llama.cpp folder. If the
fork later publishes a matching asset the row changes on the next lookup.

### The GGUF model will not start after I picked a fork

**Cause:** the active fork's install directory lost its marker or its `llama-server` (deleted,
moved, or removed by a cleaner). Discovery logs a warning naming the fork and the Settings
remedy, then continues to the custom folder and bundled runtime (`llama_cpp.py:10278`–`:10310`).
If neither of those is usable either, the load fails with "no executable llama.cpp runtime
(llama-server) is available" (`llama_cpp.py:841`); otherwise it runs the next runtime while
Settings still shows the broken fork selected, which is why the warning names the remedy.
**Fix:** reinstall that fork from Settings, or switch back to `normal`. The picker reports the
broken install as `missing_binary`, not as a fork that was never installed.

### "Another llama.cpp install or switch is already running."

**Cause:** the single operation lock is held — a job is genuinely running, including one
started from another tab, or activate was pressed mid-install (`:1498`, `:1558`).
**Fix:** wait for `job.state` to leave `running` in `GET /api/llama/architectures`. There is no
second writer by design, and no way to force one from the UI.

### "The llama.cpp path is managed by the LLAMA_SERVER_PATH environment variable."

**Cause:** `LLAMA_SERVER_PATH`, or `UNSLOTH_LLAMA_CPP_PATH` that the desktop did not set
itself, is pinned in the environment (`:538`). The picker reports `runtime_locked` and
disables itself, because step 1/2 of the resolution order would override any fork anyway.
**Fix:** unset the variable for the Studio process to take the fork picker back.

### The picker switched, but the model still runs the old fork

**Cause:** activation records a setting; it does not restart a server that is already running
with the previous binary.
**Fix:** reload the model — the hint under the row says exactly that
(`en.ts:3238`).

## Tests

```bash
cd studio/backend
python -m pytest tests/test_llama_cpp_architectures.py \
  tests/test_llama_architecture_routes.py tests/test_llama_status_cache.py -q
```

`test_llama_cpp_architectures.py` pins asset selection per host, the source fallback, digest
and size refusals, extraction refusals, the marker, and the single-writer lock;
`test_llama_architecture_routes.py` pins auth, owner gating, validation and the response shape;
`test_llama_status_cache.py` pins that repeated status reads do not re-resolve, that
`force_refresh` does, and that a failed lookup still reports installed facts.

## What this doc does not cover

- **Which fork to choose.** No file in this checkout says what any fork is for or which models
  it unlocks; the row labels and one-line descriptions are all that exist.
- **Live GitHub behaviour.** Nothing here was checked against the real release APIs today: asset
  names, tags and digests are taken from the module's rules and its hermetic tests. Whether the
  four repos currently publish assets that match those rules is unverified.
- **A genuinely private/gated repository.** Not exercised; only the code path was read.
- **Frontend strings and layout as a contract.** The Settings section and its locale overlay are
  owned by other work and were read at one point in time; the behaviour described is the
  backend's.
- **The compute-backend row** ("Compute backend": auto/cpu/cuda/rocm/vulkan) and the bundled
  llama.cpp/whisper/audio update job are a separate surface
  (`utils/llama_cpp_update.py`) and are not described here.
- **How an install interacts with `unsloth studio update`** or with a repaired bundled runtime —
  the update path was not read end to end for this feature.
- **What a row says during a running install.** Release lookups are skipped while a job runs,
  so `latest_tag` and `download_size_bytes` are absent from the payload for the duration. This
  is deliberate (one row matters mid-install) but the shape changes.
- **Whether a fork is the right choice for a workload.** Nothing in this checkout records what
  each fork is for, so this page describes repositories, assets and install behaviour only.
