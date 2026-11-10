// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The measurement substrate itself. Everything runs against a stub timeline
// so the durations are exact numbers instead of whatever the host clock felt
// like, and so the tests can see what reached the platform (prefixed names,
// consumed marks) and what never should (anything, while disabled).

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_ENTRIES,
  mark,
  measure,
  perfEnabled,
  reportLines,
  reset,
  setPerfEnabled,
  snapshot,
  time,
  timeAsync,
} from "../src/lib/perf/marks.ts";

// Top-level so the pattern is compiled once, not on every throw assertion.
const KABOOM = /kaboom/;
const NOPE = /nope/;

type StubTimeline = {
  /** Names currently on the platform timeline, by mark name. */
  marks: Map<string, number>;
  /** Every measure the platform was asked to make. */
  measures: { name: string; durationMs: number }[];
  advance(ms: number): void;
  restore(): void;
};

/**
 * Replace `globalThis.performance` with a controllable clock. The node global
 * is accessor-backed with a setter, so save/assign/restore is enough; the
 * fake only implements the four members the module reaches for.
 */
function installStubTimeline(): StubTimeline {
  let current = 1000;
  const marks = new Map<string, number>();
  const measures: { name: string; durationMs: number }[] = [];
  const previous = globalThis.performance;
  const stub = {
    now: (): number => current,
    mark: (name: string, options?: { startTime?: number }): void => {
      marks.set(name, options?.startTime ?? current);
    },
    measure: (
      name: string,
      start: string,
      end?: string,
    ): { duration: number } => {
      const from = marks.get(start);
      if (from === undefined) {
        throw new Error(`missing mark ${start}`);
      }
      const to = end === undefined ? current : marks.get(end);
      if (to === undefined) {
        throw new Error(`missing mark ${end}`);
      }
      const durationMs = to - from;
      measures.push({ name, durationMs });
      return { duration: durationMs };
    },
    clearMarks: (name?: string): void => {
      if (name === undefined) {
        marks.clear();
      } else {
        marks.delete(name);
      }
    },
  };
  globalThis.performance = stub as unknown as Performance;
  return {
    marks,
    measures,
    advance: (ms: number): void => {
      current += ms;
    },
    restore: (): void => {
      globalThis.performance = previous;
    },
  };
}

// Must run first: the enabled answer is cached from the first read, so the
// flag sources are only observable before anything has warmed the cache.
test("the devtools flag turns the session on, and the answer is cached", () => {
  const scope = globalThis as Record<string, unknown>;
  scope.__UNSLOTH_PERF__ = true;
  try {
    assert.equal(perfEnabled(), true);
    Reflect.deleteProperty(scope, "__UNSLOTH_PERF__");
    assert.equal(perfEnabled(), true); // cached: a late flip cannot lie
  } finally {
    Reflect.deleteProperty(scope, "__UNSLOTH_PERF__");
    setPerfEnabled(false);
  }
});

test("disabled instrumentation touches nothing at all", () => {
  const stub = installStubTimeline();
  try {
    setPerfEnabled(false);
    reset();
    mark("click");
    assert.equal(measure("click", "click"), null);
    assert.equal(
      time("work", () => 42),
      42,
    );
    assert.deepEqual(snapshot(), []);
    assert.equal(stub.marks.size, 0); // not even a mark reached the platform
    assert.equal(stub.measures.length, 0);
  } finally {
    stub.restore();
    setPerfEnabled(false);
    reset();
  }
});

test("disabled async timing is just the work's own promise", async () => {
  setPerfEnabled(false);
  reset();
  assert.equal(await timeAsync("load", async () => "done"), "done");
  assert.deepEqual(snapshot(), []);
});

test("measures land in the snapshot under u8: prefixed names", () => {
  const stub = installStubTimeline();
  try {
    setPerfEnabled(true);
    reset();
    mark("render-start");
    assert.deepEqual([...stub.marks.keys()], ["u8:render-start"]);
    stub.advance(12);
    mark("render-end");
    assert.deepEqual(
      [...stub.marks.keys()],
      ["u8:render-start", "u8:render-end"],
    );
    const durationMs = measure("render", "render-start", "render-end");
    assert.equal(durationMs, 12);
    assert.equal(stub.marks.size, 0); // the measurement consumed both marks
    assert.deepEqual(snapshot(), [
      { name: "render", durationMs: 12, at: 1012 },
    ]);
  } finally {
    stub.restore();
    setPerfEnabled(false);
    reset();
  }
});

test("a measure whose marks are missing records nothing", () => {
  setPerfEnabled(true);
  reset();
  assert.equal(measure("ghost", "never-marked"), null);
  assert.deepEqual(snapshot(), []);
});

test("the ring stays bounded at MAX_ENTRIES", () => {
  const stub = installStubTimeline();
  try {
    setPerfEnabled(true);
    reset();
    for (let i = 0; i < MAX_ENTRIES * 2; i += 1) {
      mark("start");
      mark("end");
      measure(`unit-${i}`, "start", "end");
    }
    const entries = snapshot();
    assert.equal(entries.length, MAX_ENTRIES);
    // The newest half survived: the oldest half was dropped, oldest first.
    assert.equal(entries[0]?.name, `unit-${MAX_ENTRIES}`);
    assert.equal(entries[MAX_ENTRIES - 1]?.name, `unit-${MAX_ENTRIES * 2 - 1}`);
  } finally {
    stub.restore();
    setPerfEnabled(false);
    reset();
  }
});

test("time records the span, then rethrows the original error", () => {
  const stub = installStubTimeline();
  try {
    setPerfEnabled(true);
    reset();
    assert.equal(
      time("ok", () => {
        stub.advance(25);
        return 42;
      }),
      42,
    );
    assert.throws(
      () =>
        time("boom", () => {
          stub.advance(5);
          throw new Error("kaboom");
        }),
      KABOOM,
    );
    const entries = snapshot();
    assert.deepEqual(
      entries.map((entry) => entry.name),
      ["ok", "boom"],
    );
    assert.equal(entries[1]?.durationMs, 5);
  } finally {
    stub.restore();
    setPerfEnabled(false);
    reset();
  }
});

test("timeAsync records the span, then rethrows the rejection", async () => {
  const stub = installStubTimeline();
  try {
    setPerfEnabled(true);
    reset();
    assert.equal(
      await timeAsync("fetch", () => {
        stub.advance(15);
        return Promise.resolve("ok");
      }),
      "ok",
    );
    await assert.rejects(
      timeAsync("fetch-boom", () => {
        stub.advance(7);
        return Promise.reject(new Error("nope"));
      }),
      NOPE,
    );
    const entries = snapshot();
    assert.deepEqual(
      entries.map((entry) => entry.name),
      ["fetch", "fetch-boom"],
    );
    assert.equal(entries[1]?.durationMs, 7);
  } finally {
    stub.restore();
    setPerfEnabled(false);
    reset();
  }
});

test("reportLines reads slowest first", () => {
  const stub = installStubTimeline();
  try {
    setPerfEnabled(true);
    reset();
    const spans: [string, number][] = [
      ["fast", 10],
      ["slow", 80],
      ["mid", 40],
    ];
    for (const [name, ms] of spans) {
      mark("start");
      stub.advance(ms);
      mark("end");
      measure(name, "start", "end");
    }
    assert.deepEqual(reportLines(), [
      "slow  80.0ms",
      "mid  40.0ms",
      "fast  10.0ms",
    ]);
  } finally {
    stub.restore();
    setPerfEnabled(false);
    reset();
  }
});

test("reset forgets recorded measurements", () => {
  const stub = installStubTimeline();
  try {
    setPerfEnabled(true);
    reset();
    mark("start");
    mark("end");
    measure("gone", "start", "end");
    assert.equal(snapshot().length, 1);
    reset();
    assert.deepEqual(snapshot(), []);
  } finally {
    stub.restore();
    setPerfEnabled(false);
    reset();
  }
});
