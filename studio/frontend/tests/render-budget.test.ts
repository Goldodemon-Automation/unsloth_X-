// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The history grid's render budget is invisible in the rendered output: the same cards come out
// windowed or not, memoized or not, and every unit test around the grid still passes either way.
// So the arithmetic the window is built from is pinned here, in the shape of
// thread-delete-render-budget.test.ts: clamping at the empty/one-row/exact boundaries, growth that
// only widens and caps at the list, a predicate that agrees with the window, the style split at the
// same boundary, and keys that survive the poll's re-sort. Undo any of these and the grid still
// renders correctly while quietly re-mounting every card on each poll or painting rows that bounce.

import assert from "node:assert/strict";
import test from "node:test";

import {
  contentVisibilityStyle,
  historyCardKey,
  initialWindow,
  nextWindow,
  shouldRenderRow,
} from "../src/lib/render-budget.ts";

test("initialWindow clamps at 0, 1 and the exact boundary", () => {
  // An empty list mounts nothing; the grid's empty state is a different branch entirely.
  assert.equal(initialWindow(0, 12), 0);
  // The exact boundary: one fetch page of 12 rows against a first screen of 12 mounts all 12...
  assert.equal(initialWindow(12, 12), 12);
  // ...below it the window is the list, and above it the window stays the first screen.
  assert.equal(initialWindow(11, 12), 11);
  assert.equal(initialWindow(13, 12), 12);
  // One row is the floor for a non-empty list: zero rows would leave no frontier to scroll toward,
  // so nothing would ever grow the window again.
  assert.equal(initialWindow(1, 12), 1);
  assert.equal(initialWindow(5, 0), 1);
  assert.equal(initialWindow(1, 0), 1);
});

test("nextWindow grows monotonically and caps at the total", () => {
  assert.equal(nextWindow(0, 100, 12), 12);
  assert.equal(nextWindow(12, 100, 12), 24);
  // A step that overshoots lands exactly on the total, never past it...
  assert.equal(nextWindow(94, 100, 12), 100);
  assert.equal(nextWindow(3, 5, 12), 5);
  // ...and at the cap further growth is a no-op instead of a shrink.
  assert.equal(nextWindow(100, 100, 12), 100);
  assert.equal(nextWindow(0, 0, 12), 0);
  // The list shrinking below the window is the one documented way the window shrinks: those rows
  // are gone either way, and the cap is what keeps the window inside the list.
  assert.equal(nextWindow(10, 5, 6), 5);
  // A stalled step still advances by one row; returning the window unchanged would strand the
  // frontier against the last mounted row with nothing left to intersect it.
  assert.equal(nextWindow(12, 100, 0), 13);
  assert.equal(nextWindow(12, 100, -4), 13);
});

test("a widening walk covers the list in strictly increasing steps", () => {
  let windowRows = initialWindow(100, 12);
  let steps = 0;
  while (windowRows < 100) {
    const next = nextWindow(windowRows, 100, 12);
    assert.ok(
      next > windowRows,
      `nextWindow stalled at ${windowRows} after ${steps} steps`,
    );
    windowRows = next;
    steps += 1;
    assert.ok(steps <= 100, "a widening walk must terminate");
  }
  assert.equal(windowRows, 100);
});

test("the predicate agrees with the window", () => {
  const rendered = initialWindow(40, 12);
  assert.equal(rendered, 12);
  for (let index = 0; index < rendered + 4; index += 1) {
    assert.equal(shouldRenderRow(index, rendered), index < rendered);
  }
  assert.equal(shouldRenderRow(-1, rendered), false);
  // The style helper makes the same cut, so mounting and styling cannot disagree...
  for (let index = 0; index < rendered + 4; index += 1) {
    assert.equal(
      shouldRenderRow(index, rendered),
      contentVisibilityStyle(rendered, index).contentVisibility === "visible",
    );
  }
  // ...and growth moves the cut in one direction only: everything mounted stays mounted.
  const grown = nextWindow(rendered, 40, 12);
  assert.equal(shouldRenderRow(rendered - 1, grown), true);
  assert.equal(shouldRenderRow(grown - 1, grown), true);
  assert.equal(shouldRenderRow(grown, grown), false);
});

test("contentVisibilityStyle: visible inside the window, auto plus an intrinsic size past it", () => {
  assert.deepEqual(contentVisibilityStyle(12, 0), {
    contentVisibility: "visible",
  });
  assert.deepEqual(contentVisibilityStyle(12, 11), {
    contentVisibility: "visible",
  });
  // The first row past the window flips the policy: skip its layout...
  assert.deepEqual(contentVisibilityStyle(12, 12), {
    contentVisibility: "auto",
    // ...but keep its footprint at the card's authored height (h-[calc(11.5rem*...)]), with the
    // `auto` prefix letting the browser remember a measured size. Pinned verbatim so a card-height
    // change cannot silently leave stale placeholders behind.
    containIntrinsicSize: "auto 11.5rem",
  });
  // Far past the window the policy is unchanged, and the zero-row window styles every row the same.
  assert.deepEqual(contentVisibilityStyle(12, 500), {
    contentVisibility: "auto",
    containIntrinsicSize: "auto 11.5rem",
  });
  assert.deepEqual(contentVisibilityStyle(0, 0), {
    contentVisibility: "auto",
    containIntrinsicSize: "auto 11.5rem",
  });
});

test("historyCardKey is stable across a re-sort of the same runs", () => {
  const runs: { id: string; created_at?: string | null }[] = [
    { id: "run-c", created_at: "2026-03-01T00:00:00Z" },
    { id: "run-a", created_at: "2026-01-01T00:00:00Z" },
    { id: "run-b", created_at: null },
  ];
  // The poll re-sorts the array it replaces; a key derived from the row's position would change
  // under every run and re-mount the whole card, which is what this pins against.
  const ascending = [...runs].sort((left, right) =>
    left.id.localeCompare(right.id),
  );
  const descending = [...ascending].reverse();
  const original = new Map(runs.map((run) => [run.id, historyCardKey(run)]));
  for (const run of [...ascending, ...descending]) {
    assert.equal(
      historyCardKey(run),
      original.get(run.id),
      `key for ${run.id} moved with its position`,
    );
  }
  // The key is the id itself, so it also matches the key the grid used before this helper existed.
  assert.deepEqual(runs.map(historyCardKey), ["run-c", "run-a", "run-b"]);
  // The id wins whenever both are present: created_at is not identity for a normal row...
  assert.equal(
    historyCardKey({ id: "run-a", created_at: "2026-01-01T00:00:00Z" }),
    "run-a",
  );
  // ...and is the only identity left for a legacy row an import produced without one.
  assert.equal(
    historyCardKey({ id: "", created_at: "2026-02-01T00:00:00Z" }),
    "created:2026-02-01T00:00:00Z",
  );
});
