// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Route spans are pure naming plus the shared perf marks: the naming cases are
// asserted directly, the report ordering against fixed entries, and the wiring
// has to stay inert -- never throwing -- while the session is disabled.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type PerfEntry,
  reset,
  setPerfEnabled,
  snapshot,
} from "../src/lib/perf/index.ts";
import {
  describeRouteSpans,
  markRouteCommit,
  markRouteStart,
  routeSpanName,
} from "../src/lib/perf/route-timing.ts";

test("routeSpanName maps the index and Train, and strips trailing slashes", () => {
  assert.equal(routeSpanName("/"), "route:index");
  assert.equal(routeSpanName(""), "route:index");
  assert.equal(routeSpanName("/train"), "route:train");
  assert.equal(routeSpanName("/train/"), "route:train");
  // The Train page lives at /studio; it reports under the same name.
  assert.equal(routeSpanName("/studio/"), "route:train");
  assert.equal(routeSpanName("/chat"), "route:chat");
});

test("routeSpanName folds unknown paths into one span", () => {
  assert.equal(routeSpanName("/not-a-route"), "route:other");
  assert.equal(routeSpanName("/not-a-route/deeper/"), "route:other");
});

test("describeRouteSpans returns the slowest three route lines", () => {
  const entries: PerfEntry[] = [
    { name: "route:chat", durationMs: 20, at: 1 },
    { name: "route:train", durationMs: 120.5, at: 2 },
    { name: "something-else", durationMs: 999, at: 3 },
    { name: "route:index", durationMs: 5, at: 4 },
    { name: "route:other", durationMs: 60, at: 5 },
    { name: "route:hub", durationMs: 35, at: 6 },
  ];
  assert.deepEqual(describeRouteSpans(entries), [
    "route:train  120.5ms",
    "route:other  60.0ms",
    "route:hub  35.0ms",
  ]);
});

test("describeRouteSpans is empty without route spans", () => {
  assert.deepEqual(describeRouteSpans([]), []);
  assert.deepEqual(
    describeRouteSpans([{ name: "click", durationMs: 12, at: 1 }]),
    [],
  );
});

test("route marks never throw while perf is disabled", () => {
  setPerfEnabled(false);
  reset();
  try {
    assert.doesNotThrow(() => {
      markRouteStart("/train");
      markRouteCommit("/train");
      markRouteCommit("/unknown");
    });
    assert.deepEqual(snapshot(), []);
  } finally {
    setPerfEnabled(false);
  }
});
