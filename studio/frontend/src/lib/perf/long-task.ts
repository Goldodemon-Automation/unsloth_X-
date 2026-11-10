// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Long-task observation: `longtask` PerformanceObserver entries, the main
 * thread stalls over ~50ms that make a UI feel slow. Importing this module
 * arms nothing -- observation starts when {@link observeLongTasks} is called.
 *
 * Only numbers and one attribution label are kept. A `longtask` entry points
 * back at the container (DOM node) it came from, and retaining entries retains
 * that DOM: the classic way an observer becomes the leak it was watching for.
 */

import { MAX_ENTRIES } from "./marks.ts";

/** Aggregate over the bounded ring; `worstAttribution` describes the worst task. */
export type LongTaskStats = {
  count: number;
  totalMs: number;
  worstMs: number;
  worstAttribution: string | null;
};

export type LongTaskObserverOptions = {
  /** Ignore tasks shorter than this even though the platform reported them. */
  thresholdMs?: number;
};

/**
 * The slice of a `longtask` entry we read. Structural and minimal on purpose:
 * enough to record a number and a label, never enough to hold the entry.
 */
type LongTaskEntry = {
  duration: number;
  attribution?: readonly {
    containerType?: string;
    containerSrc?: string;
    containerId?: string;
    containerName?: string;
  }[];
};

/** What the ring stores: primitives only, never the entry. */
type LongTaskRecord = {
  durationMs: number;
  attribution: string | null;
};

const records: LongTaskRecord[] = [];

/**
 * The shared bound (see `marks.ts`): one ring budget for every instrumentation
 * buffer, so no single feed can quietly become the biggest thing in memory.
 */
const push = (record: LongTaskRecord): void => {
  records.push(record);
  if (records.length > MAX_ENTRIES) {
    records.shift();
  }
};

/**
 * A short label for where a task ran: the most specific container hint the
 * entry offers. A label is safe to keep; the entry it came from is not.
 */
const attributionLabel = (entry: LongTaskEntry): string | null => {
  const container = entry.attribution?.[0];
  if (container === undefined) {
    return null;
  }
  return (
    container.containerName ??
    container.containerId ??
    container.containerSrc ??
    container.containerType ??
    null
  );
};

const createObserver = (thresholdMs: number): PerformanceObserver | null => {
  try {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration < thresholdMs) {
          continue;
        }
        push({
          durationMs: entry.duration,
          attribution: attributionLabel(entry),
        });
      }
    });
    observer.observe({ type: "longtask", buffered: true });
    return observer;
  } catch {
    // An engine that advertises nothing may still reject the entry type.
    return null;
  }
};

/**
 * Start recording long tasks and return a disposer that is safe to call more
 * than once. Returns an inert disposer when the engine has no `longtask` entry
 * type (Safari, Firefox) or no PerformanceObserver at all: observation is
 * best-effort and must never throw, least of all during startup.
 */
export function observeLongTasks(
  options?: LongTaskObserverOptions,
): () => void {
  const thresholdMs = options?.thresholdMs ?? 0;
  if (typeof PerformanceObserver === "undefined") {
    return () => undefined;
  }
  const supported = PerformanceObserver.supportedEntryTypes;
  if (supported !== undefined && !supported.includes("longtask")) {
    return () => undefined;
  }
  const observer = createObserver(thresholdMs);
  if (observer === null) {
    return () => undefined;
  }
  let disposed = false;
  return () => {
    if (disposed) {
      // React strict mode and friends dispose twice; the second must be free.
      return;
    }
    disposed = true;
    observer.disconnect();
  };
}

/** Aggregate over the bounded ring, not the whole session. */
export function longTaskStats(): LongTaskStats {
  let totalMs = 0;
  let worstMs = 0;
  let worstAttribution: string | null = null;
  for (const record of records) {
    totalMs += record.durationMs;
    if (record.durationMs > worstMs) {
      worstMs = record.durationMs;
      worstAttribution = record.attribution;
    }
  }
  return { count: records.length, totalMs, worstMs, worstAttribution };
}

/** Forget every recorded task; does not touch a live observer. */
export function resetLongTasks(): void {
  records.length = 0;
}
