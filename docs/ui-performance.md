# UI performance

The Studio's UI-speed work is measured with the instrumentation in
`studio/frontend/src/lib/perf/`: tiny wrappers over User Timing, `longtask`
observation, and interaction latency, all inert unless a session opts in. This
page is the reader's guide to that code — how to turn it on, what each module
records, the bounds that keep measuring itself from leaking, and the checks that
pin those decisions. It explains how to measure the UI; it reports no numbers.

## Turning it on

Instrumentation is off by default. `perfEnabled()` OR's three switch sources and
caches the answer at the first check, so every later call costs one cached
boolean comparison:

| Switch | Set | Seen |
| --- | --- | --- |
| `globalThis.__UNSLOTH_PERF__ = true` | at runtime | only before the first check |
| `VITE_UNSLOTH_PERF=1` (or `true`) | at build time, via Vite's `import.meta.env` | at the first check |
| `?perf=1` in the URL | in the page URL | at the first check |

`installPerf()` in `studio/frontend/src/main.tsx` runs the first check at
startup, so the URL and build switches are the ones visible at boot; the runtime
flag takes effect only if it is set before that first check (tests, workers, an
early injected script). `setPerfEnabled(value)` overrides the cached answer for
tests and for a toggle that must act without a reload.

Off, the wrappers stay out of the way: `mark` returns without touching the
platform, `measure` returns `null`, `time`/`timeAsync` are exactly the wrapped
call, and `trackInteraction` returns a finisher that schedules no frame.

The result is read through `reportLines()` — one `"name  12.3ms"` line per
measurement, slowest first. No profiler and no special build are involved; the
module is imported (`@/lib/perf`), not attached to `globalThis`, so a session
reads it from instrumented code or from the tests.

## What each module measures

### `marks.ts` — User Timing wrappers

`mark` places a `u8:`-prefixed mark (the prefix lets devtools filter and clear
the instrumentation without touching the app's own marks); `measure` records a
`PerfEntry` (`name`, `durationMs`, `at`) and returns the duration; `time` and
`timeAsync` bracket a unit of work, recording first and rethrowing the work's own
error unchanged; `snapshot`, `reset` and `reportLines` read and clear the ring.
The optional `mark` start time accepts an earlier timestamp, such as a click's
`event.timeStamp`. Disabled, everything is a no-op and `measure` returns `null`.
No React and no imports at all: the module runs in workers and plain node.

```ts
mark("click", event.timeStamp);
measure("click-to-open", "click", "open");
const rows = time("train-mount", () => mountTrainPage());
await timeAsync("load-runs", () => fetchRuns());
snapshot();                                        // entries, oldest first
reset();
reportLines();                                     // "name  12.3ms", slowest first
```

### `long-task.ts` — main-thread stalls

`observeLongTasks({ thresholdMs })` starts a `PerformanceObserver` on the
`longtask` entry type (`buffered: true`, so tasks from before the call are seen)
and returns a disposer that is safe to call more than once. Tasks shorter than
`thresholdMs` are ignored even when the engine reports them. `longTaskStats()`
aggregates the ring into `{ count, totalMs, worstMs, worstAttribution }`, where
the attribution is one container label (the most specific hint the entry offers:
name, id, src, type). `resetLongTasks()` forgets the records; it does not touch a
live observer. Importing the module arms nothing.

```ts
const stop = observeLongTasks({ thresholdMs: 50 });
longTaskStats(); // { count, totalMs, worstMs, worstAttribution }
resetLongTasks();
```

### `interaction.ts` — INP-style latency

`trackInteraction(name, startTime?)` brackets an interaction (a click, a
keystroke burst) and returns a finisher, safe to call more than once. The
finisher records at the second `requestAnimationFrame` — the first fires before
the next paint, the second after it — so the latency runs from the interaction's
start to the paint the user actually waited for. Every finish also goes through
`measure`, so a slow click lands in the same `snapshot()` and `reportLines()` as
everything else. `inpStats()` reports `{ count, worstMs, names }` with at most
five names, deduplicated so one hot name cannot crowd out the rest.
`resetInteractions()` forgets the records. Where `requestAnimationFrame` does
not exist (node, workers), the finisher records at once. Disabled, the finisher
does nothing at all.

```ts
const done = trackInteraction("save-settings", event.timeStamp);
done();
inpStats(); // { count, worstMs, names }
resetInteractions();
```

### `index.ts` — `installPerf()`

The entry point re-exports the three modules name by name. `installPerf()` is
called once in `src/main.tsx`; enabled, it arms long-task observation and returns
one combined disposer (idempotent, safe for React strict mode's double dispose),
the place future hooks join. Disabled, it arms nothing and returns an inert
disposer — a production build pays the one boolean check and nothing else.

```ts
const disposePerf = installPerf();
```

## The bounds, and why

Every buffer is bounded, and the bounds are the part that is easiest to undo by
accident. The obvious "keep more for debugging" change to any of these
reintroduces the leak each one prevents.

| Bound | What it stops |
| --- | --- |
| `MAX_ENTRIES = 200`, shared by all three rings | The instrumentation's own memory. A measurement log that grows all session becomes the leak it was installed to find. Oldest entries drop first; `reportLines` sorts by duration, so the lost tail is the noise. |
| `measure` consumes its marks | Platform-side growth. `performance.measure` leaves both marks in the performance timeline until cleared, so measuring all session without `clearMarks` piles up timeline entries forever. The module clears each mark the moment a measurement has used it. |
| Long-task stats keep numbers and one attribution string | DOM retention. A `longtask` entry points back at the container node it came from; keeping the entry keeps that DOM alive for the session. The ring stores `{ durationMs, attribution }` primitives and the entry is dropped. |
| Interaction records keep `name` and `durationMs`; `inpStats` names at most `MAX_NAMES = 5` | Same ring bound as marks, plus a bounded report: names are deduplicated and the five slowest win. |

## What it does not do

- **No upload.** Nothing in `src/lib/perf/` sends data anywhere; `marks.ts` has
  no imports at all. Everything stays in the page.
- **No sampling policy.** Every completed measurement is recorded. The only
  eviction is the rings' oldest-first drop past `MAX_ENTRIES`.
- **No React Profiler integration.** No React imports anywhere in the perf
  modules; the API is plain functions callable from event handlers, stores and
  workers.
- **Nothing automatic in release builds.** The code ships in the bundle, but a
  session records nothing unless it opted in. There is no auto-enable on slow
  hardware, no persistent flag, no background collection.
- **No hard failure on old engines.** Browsers without the `longtask` entry type
  (Safari, Firefox) or without `PerformanceObserver` get an inert disposer from
  `observeLongTasks` — nothing is constructed, nothing throws. An `observe()`
  that rejects the entry type degrades the same way. Every wrapper swallows
  platform errors; `time` and `timeAsync` still rethrow the wrapped work's own
  errors unchanged.

## Worked example: is the Train page slow to open?

1. Run the Studio with `?perf=1` (or a build with `VITE_UNSLOTH_PERF=1`).
2. Bracket the suspect mount with the existing API:

   ```ts
   time("train-mount", () => mountTrainPage());
   ```

3. Open the Train page, then read the report — for example a temporary
   `console.log(reportLines())` next to the instrumentation. Lines come back
   slowest first:

   ```text
   train-mount  183.4ms
   ```

4. Then read `longTaskStats()` to see what that cost looked like on the main
   thread. `count: 1` with `worstMs` near `totalMs` means one long task: the open
   blocked the thread once. A large `count` with `worstMs` well below `totalMs`
   means many small tasks: the cost is spread across renders, fetches, or
   handlers rather than one stall. `worstAttribution` names the container of the
   worst task when the engine attributes one.
5. If the question is click-to-paint rather than mount, bracket the click with
   `trackInteraction("open-train", event.timeStamp)` and read `inpStats()` for
   the worst interactions of the session.

## Bundle budget

From `studio/frontend/`, `npm run bundle:check` runs
`node --experimental-strip-types scripts/check-bundle-budget.ts` against `dist/`
(run `npm run build` first). The script reads `dist/index.html` and collects the
eager startup set: Vite's entry `<script type="module">`, one
`<link rel="modulepreload">` per chunk in the entry's static import closure, and
parser-blocking classic `<script src>` tags such as `public/theme-boot.js`. It
then sums raw bytes and transfer bytes (gzip level 6 for `assets/`, matching the
backend's `/assets` middleware; raw otherwise) and asserts two limits from the
`BUDGET` constant at the top of `scripts/check-bundle-budget.ts`:
`transferBytes` (2,058,777) and `rawBytes` (6,837,690). It also fails when
React-preview runtime markers (`__unslothModules`, `__vite_ssr_import__`) appear
in an eager chunk, and when the parsed HTML yields too few Vite chunks to be
trustworthy — a shape change that would otherwise report a comfortable pass on a
path it never measured. The chunk count is printed but not budgeted: splitting a
page out of the entry raises it while lowering the bytes, which is the behaviour
the gate rewards. Raising `BUDGET` is normal; doing it in the same change as the
import that needed it, with the measurement, is the point.

## Tests

The three perf suites run under node's own test runner. From `studio/frontend/`:

```bash
node --experimental-strip-types --test tests/perf-marks.test.ts tests/perf-long-task.test.ts tests/perf-interaction.test.ts
```

`npm test` runs these along with the rest of `tests/**/*.test.ts`.

| File | What it pins |
| --- | --- |
| `tests/perf-marks.test.ts` | Disabled calls touch no platform API at all (not even a mark); the enabled answer is cached from the first read; names are `u8:`-prefixed; `measure` consumes both marks and records nothing when a mark is missing; the ring drops the oldest past `MAX_ENTRIES`; `time`/`timeAsync` record the span and then rethrow the original error or rejection; `reportLines` reads slowest first; `reset` forgets. |
| `tests/perf-long-task.test.ts` | No `PerformanceObserver`, no `longtask` entry type, and an `observe()` that throws all degrade to an inert disposer; stats aggregate durations and the most specific attribution label; the record ring is bounded at `MAX_ENTRIES`; the disposer is idempotent and stops delivery; `installPerf` arms nothing while off and tears down once while on. |
| `tests/perf-interaction.test.ts` | Disabled tracking schedules no frame and records nothing; a finisher records exactly once at the second `requestAnimationFrame` (the frame after the paint) and the latency lands in the shared snapshot; with no `requestAnimationFrame` the finisher records at once; `inpStats` names the five slowest interactions with names deduplicated; `resetInteractions` clears. |
