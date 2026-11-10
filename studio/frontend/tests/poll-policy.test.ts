// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  nextPollDelay,
  shouldStop,
} from "../src/features/training/lib/poll-policy.ts";

const BASE_MS = 3000;

test("the first delay after a failure is the base delay", () => {
  assert.equal(
    nextPollDelay({
      baseMs: BASE_MS,
      attempt: 0,
      failed: true,
      hidden: false,
    }),
    BASE_MS,
  );
});

test("the backoff doubles with every failed attempt", () => {
  assert.equal(
    nextPollDelay({
      baseMs: BASE_MS,
      attempt: 1,
      failed: true,
      hidden: false,
    }),
    BASE_MS * 2,
  );
  assert.equal(
    nextPollDelay({
      baseMs: BASE_MS,
      attempt: 2,
      failed: true,
      hidden: false,
    }),
    BASE_MS * 4,
  );
});

test("the backoff stops growing at four times the base delay", () => {
  for (const attempt of [3, 4, 12, 40]) {
    assert.equal(
      nextPollDelay({
        baseMs: BASE_MS,
        attempt,
        failed: true,
        hidden: false,
      }),
      BASE_MS * 4,
    );
  }
});

test("a healthy poll keeps the base delay whatever the attempt count was", () => {
  assert.equal(
    nextPollDelay({
      baseMs: BASE_MS,
      attempt: 0,
      failed: false,
      hidden: false,
    }),
    BASE_MS,
  );
  assert.equal(
    nextPollDelay({
      baseMs: BASE_MS,
      attempt: 5,
      failed: false,
      hidden: false,
    }),
    BASE_MS,
  );
});

test("a hidden tab waits longer than a visible one", () => {
  const visible = nextPollDelay({
    baseMs: BASE_MS,
    attempt: 0,
    failed: false,
    hidden: false,
  });
  const hidden = nextPollDelay({
    baseMs: BASE_MS,
    attempt: 0,
    failed: false,
    hidden: true,
  });
  assert.ok(hidden > visible);

  // The same holds while the backoff is running: hidden is longer than the
  // visible delay at the same attempt.
  assert.ok(
    nextPollDelay({
      baseMs: BASE_MS,
      attempt: 1,
      failed: true,
      hidden: true,
    }) >
      nextPollDelay({
        baseMs: BASE_MS,
        attempt: 1,
        failed: true,
        hidden: false,
      }),
  );
});

test("the poll stops once the run it watches has finished", () => {
  assert.equal(
    shouldStop({ finished: true, attempts: 0, maxAttempts: 8 }),
    true,
  );
});

test("the poll stops at the attempt ceiling", () => {
  assert.equal(
    shouldStop({ finished: false, attempts: 8, maxAttempts: 8 }),
    true,
  );
  assert.equal(
    shouldStop({ finished: false, attempts: 7, maxAttempts: 8 }),
    false,
  );
});

test("an endless budget never stops on the ceiling", () => {
  assert.equal(
    shouldStop({ finished: false, attempts: 10_000, maxAttempts: Infinity }),
    false,
  );
});
