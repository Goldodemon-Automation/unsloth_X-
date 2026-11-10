// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Interaction latency is measured to the paint, so the interesting part is
// the frame choreography: a finisher must wait for the SECOND
// requestAnimationFrame (the one that lands after the next paint) and record
// exactly once, while a node host with no rAF at all records at once. The
// frame queue is faked on `globalThis`; durations use the real clock with
// lower-bound assertions, because the point is ordering, not nanoseconds.

import assert from "node:assert/strict";
import test from "node:test";

import {
  inpStats,
  reset,
  resetInteractions,
  setPerfEnabled,
  snapshot,
  trackInteraction,
} from "../src/lib/perf/index.ts";

type FakeFrames = {
  pending: Array<() => void>;
  /** Run the callbacks queued for one frame; a callback may queue the next. */
  flush(): void;
  restore(): void;
};

function installFakeFrames(): FakeFrames {
  const pending: Array<() => void> = [];
  globalThis.requestAnimationFrame = (
    callback: FrameRequestCallback,
  ): number => {
    pending.push(() => callback(0));
    return pending.length;
  };
  return {
    pending,
    flush(): void {
      const callbacks = pending.splice(0, pending.length);
      for (const callback of callbacks) {
        callback();
      }
    },
    restore(): void {
      Reflect.deleteProperty(globalThis, "requestAnimationFrame");
    },
  };
}

test("disabled tracking schedules no frame and records nothing", () => {
  setPerfEnabled(false);
  reset();
  resetInteractions();
  const frames = installFakeFrames();
  try {
    const finish = trackInteraction("click");
    finish();
    assert.equal(frames.pending.length, 0); // not even a frame was scheduled
    assert.deepEqual(snapshot(), []);
    assert.deepEqual(inpStats(), { count: 0, worstMs: 0, names: [] });
  } finally {
    frames.restore();
    setPerfEnabled(false);
  }
});

test("a tracked interaction lands at the second frame, once", () => {
  setPerfEnabled(true);
  reset();
  resetInteractions();
  const frames = installFakeFrames();
  try {
    const startedAt = performance.now() - 40;
    const finish = trackInteraction("save-settings", startedAt);
    finish();
    finish(); // the interaction ran once; a second finish records nothing
    assert.equal(frames.pending.length, 1); // first frame armed
    frames.flush();
    assert.equal(frames.pending.length, 1); // second frame armed, still unrecorded
    assert.equal(inpStats().count, 0);
    frames.flush(); // after the paint: now the latency is recorded
    assert.equal(frames.pending.length, 0);
    const stats = inpStats();
    assert.equal(stats.count, 1);
    assert.ok(stats.worstMs >= 40, `expected >= 40ms, got ${stats.worstMs}`);
    assert.deepEqual(stats.names, ["save-settings"]);
    // And it landed in the shared report: one place with everything else.
    const entries = snapshot();
    assert.equal(entries.length, 1);
    assert.equal(entries[0]?.name, "save-settings");
    assert.ok(Math.abs((entries[0]?.durationMs ?? 0) - stats.worstMs) < 0.001);
  } finally {
    frames.restore();
    setPerfEnabled(false);
    reset();
    resetInteractions();
  }
});

test("without requestAnimationFrame the finisher records at once", () => {
  setPerfEnabled(true);
  reset();
  resetInteractions();
  try {
    assert.equal(typeof globalThis.requestAnimationFrame, "undefined");
    const startedAt = performance.now() - 25;
    const finish = trackInteraction("keystroke", startedAt);
    finish(); // no frame to wait for: recorded before the call returns
    const stats = inpStats();
    assert.equal(stats.count, 1);
    assert.ok(stats.worstMs >= 25);
    assert.deepEqual(stats.names, ["keystroke"]);
    assert.equal(snapshot().length, 1);
  } finally {
    setPerfEnabled(false);
    reset();
    resetInteractions();
  }
});

test("inpStats names the slowest five interactions only", () => {
  setPerfEnabled(true);
  reset();
  resetInteractions();
  try {
    for (let i = 0; i < 10; i += 1) {
      const finish = trackInteraction(
        `tap-${i}`,
        performance.now() - (i + 1) * 10,
      );
      finish();
    }
    const stats = inpStats();
    assert.equal(stats.count, 10);
    assert.equal(stats.names.length, 5);
    assert.deepEqual(stats.names, [
      "tap-9",
      "tap-8",
      "tap-7",
      "tap-6",
      "tap-5",
    ]);
  } finally {
    setPerfEnabled(false);
    reset();
    resetInteractions();
  }
});

test("one hot name keeps its slot, and reset clears the slate", () => {
  setPerfEnabled(true);
  reset();
  resetInteractions();
  try {
    for (const span of [10, 30, 20]) {
      const finish = trackInteraction("click", performance.now() - span);
      finish();
    }
    const stats = inpStats();
    assert.equal(stats.count, 3);
    assert.ok(stats.worstMs >= 30);
    assert.deepEqual(stats.names, ["click"]); // named once, not three times
    resetInteractions();
    assert.deepEqual(inpStats(), { count: 0, worstMs: 0, names: [] });
  } finally {
    setPerfEnabled(false);
    reset();
    resetInteractions();
  }
});
