// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Interaction latency, INP-style: from an interaction's start to the paint
 * after its next frame -- what the user actually waited for. Every finish goes
 * through `marks.measure`, so a slow click lands in the same `snapshot()` and
 * `reportLines()` as everything else; a small local ring feeds {@link inpStats}.
 *
 * No React: usable from event handlers, stores, and workers alike, and the
 * tests run it in plain node where `requestAnimationFrame` does not exist.
 */

import { MAX_ENTRIES, mark, measure, perfEnabled } from "./marks.ts";

/** How many distinct interaction names a report bothers to name. */
const MAX_NAMES = 5;

type InteractionRecord = {
  name: string;
  durationMs: number;
};

const records: InteractionRecord[] = [];

const nowMs = (): number =>
  typeof performance === "undefined" ? Date.now() : performance.now();

const push = (record: InteractionRecord): void => {
  records.push(record);
  if (records.length > MAX_ENTRIES) {
    // Oldest out; `inpStats` aggregates over what remains.
    records.shift();
  }
};

/**
 * Run after the next paint: rAF twice, because the first callback fires
 * before the next paint and the second after it. Where rAF does not exist
 * (node, workers) the wall clock at the call is the honest fallback.
 */
const afterPaint = (callback: () => void): void => {
  if (typeof requestAnimationFrame !== "function") {
    callback();
    return;
  }
  requestAnimationFrame(() => {
    requestAnimationFrame(callback);
  });
};

const record = (name: string, startedAt: number, endedAt: number): void => {
  const startMark = `${name}:start`;
  const endMark = `${name}:end`;
  // Marks carry the interaction's own timeline points, so the shared measure
  // holds the real latency even when the start predates this call.
  mark(startMark, startedAt);
  mark(endMark, endedAt);
  measure(name, startMark, endMark);
  push({ name, durationMs: endedAt - startedAt });
};

/**
 * Bracket an interaction (a click, a keystroke burst). `startTime` may be the
 * triggering event's timestamp; it defaults to now. The returned finisher
 * records the latency at the next double-rAF boundary and is safe to call
 * more than once. Disabled sessions get a finisher that does nothing at all:
 * not even a frame is scheduled.
 */
export function trackInteraction(name: string, startTime?: number): () => void {
  if (!perfEnabled()) {
    return () => undefined;
  }
  const startedAt = startTime ?? nowMs();
  let finished = false;
  return () => {
    if (finished) {
      // The interaction ran once; recording it twice would double-count it.
      return;
    }
    finished = true;
    afterPaint(() => {
      record(name, startedAt, nowMs());
    });
  };
}

/**
 * The worst interactions of the session: how many were recorded, the worst
 * latency seen, and up to {@link MAX_NAMES} names of the slowest ones. Names
 * are deduplicated -- one hot button repeated all morning should still leave
 * room to name the other slow work.
 */
export function inpStats(): {
  count: number;
  worstMs: number;
  names: string[];
} {
  let worstMs = 0;
  const worstByName = new Map<string, number>();
  for (const record of records) {
    worstMs = Math.max(worstMs, record.durationMs);
    const known = worstByName.get(record.name) ?? 0;
    if (record.durationMs > known) {
      worstByName.set(record.name, record.durationMs);
    }
  }
  const names = [...worstByName.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, MAX_NAMES)
    .map(([name]) => name);
  return { count: records.length, worstMs, names };
}

/** Forget every recorded interaction. */
export function resetInteractions(): void {
  records.length = 0;
}
