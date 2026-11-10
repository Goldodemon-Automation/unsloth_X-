// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * User Timing substrate for the "make the UI faster" work: tiny wrappers over
 * `performance.mark`/`performance.measure` that stay a true no-op unless the
 * session opted in, and that cannot break the app or leak while measuring it.
 *
 * No React and no imports at all, so this runs in workers and in plain node
 * alike; the tests drive it through faked globals.
 */

/** One recorded measurement: the caller's name, its duration, and when it landed. */
export type PerfEntry = {
  name: string;
  durationMs: number;
  at: number;
};

/**
 * Measurements retained. A ring, not a log: instrumentation that grows without
 * bound becomes the very leak it was installed to find, and the tail of a long
 * session is noise anyway. `reportLines` sorts, so dropping the oldest loses
 * nothing worth reporting.
 */
export const MAX_ENTRIES = 200;

/**
 * Every mark/measure name gets this prefix, so devtools can filter -- and
 * clear -- the instrumentation without touching the app's own marks.
 */
const MARK_PREFIX = "u8:";

const entries: PerfEntry[] = [];

/** Null until the first read, then cached: the hot path is one comparison. */
let enabledCache: boolean | null = null;

const runtimeFlag = (): unknown =>
  (globalThis as Record<string, unknown>).__UNSLOTH_PERF__;

const buildFlag = (): string => {
  try {
    // vite injects import.meta.env; plain node (the tests) has none and
    // reading through it throws, which must not take the app down.
    const value: unknown = import.meta.env.VITE_UNSLOTH_PERF;
    return typeof value === "string" ? value : "";
  } catch {
    return "";
  }
};

const queryFlag = (): boolean => {
  if (typeof location === "undefined") {
    return false;
  }
  try {
    return new URLSearchParams(location.search).get("perf") === "1";
  } catch {
    return false;
  }
};

/**
 * True when the session opted in: `__UNSLOTH_PERF__ = true` from devtools,
 * `VITE_UNSLOTH_PERF=1|true` baked in at build time, or `?perf=1` in the URL.
 * Cached after the first read; {@link setPerfEnabled} overrides the cache for
 * tests and for a dev toggle that must take effect without a reload.
 */
export function perfEnabled(): boolean {
  if (enabledCache === null) {
    const build = buildFlag();
    enabledCache =
      runtimeFlag() === true ||
      build === "1" ||
      build === "true" ||
      queryFlag();
  }
  return enabledCache;
}

/** Force the cached answer; see {@link perfEnabled}. */
export function setPerfEnabled(value: boolean): void {
  enabledCache = value;
}

const timeline = (): Performance | null =>
  typeof performance === "undefined" ? null : performance;

const nowMs = (): number => {
  const tl = timeline();
  return tl === null ? Date.now() : tl.now();
};

const push = (entry: PerfEntry): void => {
  entries.push(entry);
  if (entries.length > MAX_ENTRIES) {
    // Oldest out: the slowest work is what matters, and the report sorts.
    entries.shift();
  }
};

/**
 * Drop a mark once a measurement has consumed it. Marks live in the
 * performance timeline until cleared, so measuring all session long would
 * otherwise pile up entries forever -- the same leak the ring bound above
 * prevents on our side. A platform without `clearMarks` just keeps them,
 * which is not worth throwing over.
 */
const consumeMark = (tl: Performance, name: string): void => {
  try {
    tl.clearMarks(name);
  } catch {
    // Instrumentation never breaks the app.
  }
};

/**
 * A mark named `u8:<name>`, at `startTime` (default: now). The optional start
 * time exists for work that began before the call -- a click's
 * `event.timeStamp` -- and stays optional so the plain case reads like the
 * platform API. A no-op while instrumentation is off.
 */
export function mark(name: string, startTime?: number): void {
  if (!perfEnabled()) {
    return;
  }
  const tl = timeline();
  if (tl === null) {
    return;
  }
  try {
    if (startTime === undefined) {
      tl.mark(MARK_PREFIX + name);
    } else {
      tl.mark(MARK_PREFIX + name, { startTime });
    }
  } catch {
    // Instrumentation never breaks the app.
  }
}

/**
 * Measure from `startMark` to `endMark` (or to now) and record the result,
 * returning the duration. Null when disabled, when a mark is missing, or when
 * the platform refused the measurement: a missing measurement must never look
 * like a crash. Both marks are consumed afterwards.
 */
export function measure(
  name: string,
  startMark: string,
  endMark?: string,
): number | null {
  if (!perfEnabled()) {
    return null;
  }
  const tl = timeline();
  if (tl === null) {
    return null;
  }
  const prefixedStart = MARK_PREFIX + startMark;
  const prefixedEnd = endMark === undefined ? undefined : MARK_PREFIX + endMark;
  try {
    const measured =
      prefixedEnd === undefined
        ? tl.measure(MARK_PREFIX + name, prefixedStart)
        : tl.measure(MARK_PREFIX + name, prefixedStart, prefixedEnd);
    push({ name, durationMs: measured.duration, at: nowMs() });
    consumeMark(tl, prefixedStart);
    if (prefixedEnd !== undefined) {
      consumeMark(tl, prefixedEnd);
    }
    return measured.duration;
  } catch {
    // A missing mark makes the number meaningless; record nothing.
    return null;
  }
}

/**
 * Measure a unit of work under `name`. Disabled, this is exactly `fn()`. An
 * error is recorded first -- the slow, failing work is exactly what one wants
 * to see -- and then rethrown unchanged.
 */
export function time<T>(name: string, fn: () => T): T {
  if (!perfEnabled()) {
    return fn();
  }
  const startMark = `${name}:start`;
  const endMark = `${name}:end`;
  mark(startMark);
  try {
    return fn();
  } finally {
    mark(endMark);
    measure(name, startMark, endMark);
  }
}

/**
 * {@link time} for async work: measured across the whole span, recorded even
 * when the promise rejects, and the rejection rethrown unchanged.
 */
export async function timeAsync<T>(
  name: string,
  fn: () => Promise<T>,
): Promise<T> {
  if (!perfEnabled()) {
    return fn();
  }
  const startMark = `${name}:start`;
  const endMark = `${name}:end`;
  mark(startMark);
  try {
    return await fn();
  } finally {
    mark(endMark);
    measure(name, startMark, endMark);
  }
}

/** Recorded measurements, oldest first. */
export function snapshot(): PerfEntry[] {
  return [...entries];
}

/** Forget everything recorded so far. */
export function reset(): void {
  entries.length = 0;
}

/** One `"name  12.3ms"` line per measurement, slowest first. */
export function reportLines(): string[] {
  return [...entries]
    .sort((left, right) => right.durationMs - left.durationMs)
    .map((entry) => `${entry.name}  ${entry.durationMs.toFixed(1)}ms`);
}
