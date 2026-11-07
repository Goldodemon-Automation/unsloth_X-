# Labs (Settings → Labs)

One installation-wide switch that opens the unfinished features, and one
registry that decides what "unfinished" means. Labs is off by default: every
route behind it answers 403 until the switch is on, so a feature is closed even
when its page is reached by URL.

## The switch

`core/labs/__init__.py` owns the state:

| Thing | Where |
| --- | --- |
| Stored value | `app_settings` key `labs_state`, `{"enabled": bool, "features": {id: bool}}` |
| Read | `labs.read_state()` — a failed read means **off**, never on |
| Write | `labs.set_state(...)`, reached through `PUT /api/labs` (owner only) |
| Gate | `labs.require(feature_id)` raises `LabsDisabled`, which routes turn into a 403 |

`GET /api/labs` is readable by any signed-in account because the UI needs the
switch to decide what to draw. The write is behind
`policy.require_owner`, like the other installation-wide settings.

Two rules keep the state honest:

* Unknown feature ids are refused, not stored. A stale value cannot enable a
  feature that no longer exists.
* `feature_enabled()` is `master AND feature`. Turning one experiment off
  closes only that experiment's routes; turning Labs off closes all of them.

## The registry

`FEATURES` in `core/labs/__init__.py` is the single list. The Settings tab
renders what the server sends, so adding an entry is the whole UI change; do
not copy the list into the frontend.

| Id | Kind | What it opens |
| --- | --- | --- |
| `artifacts` | studio | The Artifacts studio: produced checkpoints, provenance, legal next steps |
| `pipelines` | studio | The Pipelines planner: plan a merge → scale → train → automated → quantize → benchmark chain |
| `benchmarks` | studio | The Benchmark studio: score an artifact, keep the report, compare later |
| `cmt` | studio | CMT: per-dataset splits, capacity-matched expansion, new-blocks-only training |
| `housekeeping` | studio | The Cleanup studio: output and HF-cache sizes, guarded delete |
| `auto-scale` | feature | The wizard card that grows a checkpoint by the size its own datasets justify |
| `manifest-viewer` | feature | The raw `run_manifest.json` beside an artifact |
| `scale-from-artifact` | feature | Hand one artifact's path to the Scaler |
| `training-continuation` | feature | Make an artifact the Training model in one click |
| `automated-training` | studio | The Automated training loop: sample datasets, run CMT/CPT/FFT trainings one at a time, benchmark each, continue the best, and record cancellations and timeouts in the manifest |

Each row in the Settings tab toggles one of these. A feature that is off says so
on its own page (`LabsOffCard`) rather than disappearing.

## Benchmarks keep evidence, not the checkpoint

A benchmark record lives under `outputs/labs-benchmarks/records/<id>.json` and
never edits the artifact it describes. The eval job still writes
`auto_eval_report.json` and `.md` under its own run directory; the record is the
small, listable summary the page reads after the job is gone.

The studio reuses the existing `eval_harness`/`eval_runner` pair. It does not
add a second benchmark implementation, so the suites and the `lm-eval` install
check stay in one place. A delta under 1.5 points is labelled within noise, the
same convention the post-training Regression Guard uses.

## Automated training

`core/labs/automated_training.py` is the loop that searches instead of asking
for one setting. Each round it:

1. Samples 2–8 datasets from the pool, balanced by category: code, syntax,
   reasoning or general.
2. Spreads 2–72 training runs over those datasets and the selected methods
   (CMT, CPT and FFT, where FFT means full fine-tuning).
3. Trains and benchmarks every run.
4. Picks the round winner, continue-trains it 2–72 more times, re-benchmarks
   each continuation, and keeps the best checkpoint of the round.
5. Repeats for 1–12 rounds.

Every number is user-selectable, and presets exist for a quick first run, so
those ranges are defaults rather than a hard-coded plan. The loop stops at the
round count the user gave; it does not decide on its own that the search is
done.

Planning is config arithmetic only. `plan()` validates the spec, samples with a
seeded RNG (`seed`, default 3407) and lays out every round, dataset and run
without loading a weight, touching a GPU or the network, so the same spec always
produces the same rounds. `projection.total_train_runs` is
`(runs_per_round + continue_runs) * rounds`.

Execution takes the trainer and benchmark as callables; the real adapters are
imported lazily in `routes/labs.py`. `POST /api/labs/automated/run` starts one
job, which takes one resource lease on the request thread and runs the whole
loop on a single background thread. Runs execute one at a time, never in
parallel, so that one lease covers the entire search rather than a single run.

`presets()` returns four starting specs — `quick-check`, `balanced`, `deep` and
`reasoning-heavy` — each a base layer that every value the user wrote overrides.

The three methods are not interchangeable, and the loop says so:

* **CMT** runs directly: it owns its datasets, per-dataset splits and eval, so
  it needs no extra trainer settings.
* **CPT** and **FFT** need `training_kwargs` for the dataset, method and
  parameters the trainer expects, because they post a normal training start
  rather than the CMT route. The adapter then waits for that job to finish
  (`wait_for_completion`, default true) and bounds the wait with
  `timeout_seconds`, which defaults to 6 hours when the spec leaves it at 0.
  Setting `wait_for_completion: false` restores the fire-and-forget behaviour,
  where the adapter returns as soon as the backend accepts the job.

A category that cannot be sampled, or a method whose `training_kwargs` are
missing, is reported rather than skipped silently. The Best-pick helpers in
`benchmarks.py` (`_record_score`, `rank_records`, `best_record`) are the pure
rule it uses to choose between outputs: the higher mean candidate score wins,
unscored records go last, and ties keep the input order.

### Cancelling a run

`POST /api/labs/automated/jobs/{job_id}/cancel` asks a running job to stop. It
is owner-only (`policy.require_owner`) and gated by the `automated-training`
switch, like the start route. It answers `{"job_id": ..., "cancelling": true}`
as soon as the flag is set — it does not wait for the run to reach a boundary —
and answers 404 (`Unknown or finished automated job.`) for a job this process is
not running: an unknown id, a job that already finished, or one a previous
process started. The flags sit in a bounded in-process registry
(`_AUTOMATED_CANCEL`, capped at 64, oldest dropped first), so a restart or
enough newer jobs means there is nothing left to cancel and the route says 404
instead of claiming it stopped something.

The route passes `should_stop=cancel.is_set` into the engine, which polls it
between rounds, before every candidate run and before every continuation run.
Once it returns `True` nothing new starts: the continuation runs of the current
round are skipped, the rounds that already finished are kept, the manifest is
written with `"cancelled": true`, the last progress stage is `cancelled`
instead of `done`, and `run()` returns normally. A cancellation is a result, not
an error — the job record completes with that result, and a `KeyboardInterrupt`
or `SystemExit` raised by an adapter still propagates. A round stopped before it
produced a single candidate is not recorded at all rather than reported as a
failed round. The flag is only read between runs, so it never kills a training
subprocess mid-step: the run already in flight finishes first.

The panel shows a **Cancel** action while a job is running. It is live only for
a `queued` or `running` job (`isAutomatedJobCancellable`) and shows
*Cancelling…* until the poll sees a terminal status. A refused request hands
the button back with the route's reason printed beside it, and the poll keeps
running; a cancelled result reports `cancelled: true`, which changes the summary
headline from "finished" to "cancelled".

### Resuming

`run(..., resume_from=...)` continues a run that was cancelled, interrupted or
simply stopped. `resume_from` names the earlier output directory, or its
`run_manifest.json` directly, and that manifest supplies what to reuse:

* `rounds` already recorded, keyed by their 1-based `index`. A planned round
  whose index is already recorded is skipped instead of re-run.
* `winners`, which the new rounds are appended to, so the new manifest holds the
  whole history.
* `totals` (`train_runs`, `benchmark_runs`, `failed_runs`) as the starting values
  the new rounds add to.
* `best` (read from the top level, or from `artifacts.best`), used as the
  checkpoint a resumed round has to beat. It is ranked with `order = -1`, below
  every candidate of a resumed round, so an equal score keeps the checkpoint the
  earlier run already produced and only a better score takes the lead.

A `resume` progress stage names how many planned rounds were skipped before the
first new one starts.

A missing manifest, unreadable JSON, a payload that is not a JSON object, a
`rounds` block that is not a list, or a record without a usable `index` are each
ignored with a warning in the returned summary, and the run proceeds from
scratch; resuming never becomes a failed start for a sick manifest. Without
`resume_from` nothing changes.

Resume is an engine parameter only so far. `POST /api/labs/automated/run`
accepts `spec` and `output_dir` (unknown fields are refused), no route passes
`resume_from`, and the page has no Resume control, so continuing a cancelled job
from the UI is not wired yet.

### Timeouts

Timeouts belong to the adapter, not the engine. The engine carries
`timeout_seconds` (0–604800, default 0 meaning "the adapter decides") into every
step spec and into `plan()["projection"]`, and never enforces it. The CPT/FFT
adapter turns a wait past its bound into an error, which the engine records as a
failed run: the round keeps its other candidates and still picks a winner. Only
a round in which every candidate failed raises.

### What a round writes

`run_manifest.json` in the output directory, with the ordinary manifest fields
plus `spec`, `rounds` (per-round datasets, runs, continuations, failures and
winner), `winners`, `totals` (`train_runs`, `benchmark_runs`, `failed_runs`,
`rounds`), `warnings` and `cancelled`. `write_manifest` also stamps the NRC-3
notice (`LICENSE-NRC-3.txt`) beside it. Each run and continuation writes into
its own `round-NN/run-NN-<method>` or `round-NN/continue-NN` folder, so a failed
or abandoned round is still traceable.

## CMT is one job and one experiment

CMT (`core/labs/cmt.py`) loads the base on CPU, splits every local dataset into
its own train and held-out rows, expands the checkpoint with
`force_small_growth=True`, freezes every inherited parameter, and trains only
the inserted blocks. Each held-out split becomes one held-out perplexity task in
the same `cpt_train` call, so the per-dataset delta comes from the training run
itself rather than a second eval pass.

The sizing rule is `clamp(((1.7 * megatokens + 12) / 10) * 1M, 1M, 700M)`, with
the row fallback at 1M per 10k rows. It is a heuristic, and the page says so.
Two limits are honest, not bugs:

* The engine inserts whole blocks. If the requested growth is smaller than one
  block, the plan reports the block's size as the real growth.
* These additions are routinely below the 2x range where growth gains have been
  measured. That is allowed by the Scaler now, but the manifest still records
  `floor_bypassed: true` so the run is marked as outside the evidenced range.

CMT v1 reads local `.jsonl`, `.json`, `.parquet`, `.csv` and `.txt` files. It
does not download Hub datasets during planning.

### CMT is also a method in the trainer

While the `cmt` experiment is on, the wizard's method picker offers **CMT** next
to LoRA/QLoRA/Full/CPT. Picking it swaps the dataset, preset, parameter and
advanced cards for the CMT card, because CMT owns its datasets, per-dataset
splits, eval and steps. The trainer's Start button is replaced by a note: CMT
never posts `/api/training/start`, and `startFreshTrainingRun()` refuses the
method before a payload is built. Everything the run needs travels to
`/api/labs/cmt/run` instead.

Two consequences are deliberate:

* CMT is offered only on a CUDA-class host. It loads in fp16, expands the
  checkpoint and trains dense new blocks; CPU and Apple Silicon hosts cannot.
* Turning the experiment off resets a persisted `cmt` selection to `lora`, so a
  stale browser cannot sit on an engine whose routes now answer 403.

`toBackendTrainingType("cmt")` throws. That is the boundary: the trainer's
`training_type` literal has no CMT member, and the tests pin the refusal.

## Scale before training

`core/labs/autoscale.py` shares CMT's sizing rule and then stops: it expands the
checkpoint, saves it, and leaves training to the method the user picks. It is
the data-sized alternative to the Scaler's slider.

The wizard shows it as **Scale from the dataset** whenever `auto-scale` is on
and the method is not CMT (CMT does the same sizing internally). It accepts one
local dataset path per line, plans without loading a weight, then runs one job
that writes a `scaled/` checkpoint and its manifest. *Use as the training model*
points the wizard at that folder; the run itself stays a separate, normal start.

Keeping growth and training separate is on purpose. Re-scaling from inside every
Start would grow the model again on each retry and make an accidental rerun
permanent. The card makes the growth a step the user can see and keep.

## Why Pipelines does not run

The planner prices a chain from `config.json` arithmetic only: parameter
projections, fp16 bytes, the bytes each GGUF level writes, and the refusals the
engines already enforce (growth policy, cap, tokenizer mismatch, quantize
ordering). It never loads weights.

`STAGE_KINDS` is `merge`, `scale`, `train`, `automated`, `quantize`,
`benchmark`, and that is the order the chain has to respect: a stage after
`quantize` is refused unless it is `benchmark`, so an `automated` stage has to
come before the quantization.

`automated` is the new kind, labelled **Automated train**, and it prices an
automated-training run the way the engine plans it rather than the way it
feels. It needs a local source folder, and like `scale` it falls back to the
pipeline-level source when the stage names none; a Hub id or a missing folder
blocks it. It reads three optional numbers, mirroring the engine's own bounds:
`runs_per_round` 2–72 (default 8), `rounds` 1–12 (default 2) and
`continue_runs` 2–72 (default 4). The plan is
`(runs_per_round + continue_runs) * rounds` training runs, each saved at the
current fp16 size, so `output_bytes` is that count times `current * 2`.
`params_after` stays at the current parameter count: the stage retrains the same
size unless a `scale` stage before it changed it.

The number is a disk figure, not a memory figure, and the plan says so in a
warning: the engine trains and benchmarks one candidate at a time, keeps the
round winner and moves on, so the chain prices all saved candidates and does not
add up concurrent runs. Nothing in the plan executes — the stage is a projection
of what a run of that shape would write, and a value outside those bounds blocks
the stage instead of starting anything.

Executing the chain needs durability the job layer does not have yet — jobs live
in an in-memory dict, leases are taken on the request thread, and a restart
loses the run. Shipping a run button on top of that would lose an overnight
pipeline. The plan is the part that can be right today, and it is what tells a
user a chain is impossible before they spend the GPU hours.

## Adding an experiment

1. Add one entry to `FEATURES` with an `id`, a label, a description, a status
   (`alpha` or `beta`) and `studio`.
2. Call `labs.require("<id>")` at the top of every route the experiment adds.
   Hiding the page is not gating it.
3. Gate the page with `<LabsFeatureGate feature="<id>">`; a deep link then shows
   the switch and where to flip it.
4. Add the route to `CHAT_ONLY_ALLOWED` only if it works with no accelerator.
5. Write the test: the route is 403 with the experiment off and 200 with it on.

## What Labs must never do

* Rewrite or migrate an existing artifact in place. An experiment may write a
  new folder; older builds have to be able to read what they already had.
* Stand in for a safety review. Destructive behavior gets built properly, not
  hidden behind the switch.
* Delete outputs when Labs is turned off. The switch hides entry points; the
  files stay where they are.

## Tests

Backend (`studio/backend/tests/`):

```bash
pytest tests/test_labs.py tests/test_labs_artifacts.py \
  tests/test_labs_benchmarks.py tests/test_labs_cmt.py \
  tests/test_labs_autoscale.py tests/test_labs_housekeeping.py -q
```

Frontend (`studio/frontend/`):

```bash
node --experimental-strip-types --test tests/labs-feature-gate.test.ts
node --experimental-strip-types --test tests/training-method-cmt.test.ts
node --experimental-strip-types --test tests/sidebar-nav-backend-parity.test.ts
```

The sidebar parity test compares `SIDEBAR_NAV_ITEM_DEFAULTS` (backend) with
`DEFAULT_CUSTOMIZATION.sidebarNav` (frontend), including order: the `labs` row
was added to both lists and to `SIDEBAR_NAV_ITEM_IDS`, and the backend's
`SidebarNavItemId` literal, in the same position.
