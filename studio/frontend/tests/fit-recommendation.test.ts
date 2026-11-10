// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

// The helpers are pure, but the module they live in fetches: stub the two network
// imports so the test drives the real helpers with no server and no auth state.
const fitApi =
  loadWithStubs<typeof import("../src/features/training/api/fit-api.ts")>(
    new URL("../src/features/training/api/fit-api.ts", import.meta.url),
    {
      "@/features/auth": { authFetch: async () => new Response("{}") },
      "@/lib/format-fastapi-error": { readFastApiError: async () => "error" },
    },
  );

// What the registry recommends for an integrated GPU: batch 1, accumulation 8,
// bfloat16, checkpointing on (backend core/training/device_targets.py).
const IGPU = {
  recommended: {
    batch_size: 1,
    grad_accum: 8,
    dtype: "bfloat16",
    gradient_checkpointing: true,
  },
};

test("the wizard's fields pass through unchanged when the recommendation is not requested", () => {
  const input = {
    model: "unsloth/Qwen3-4B",
    batchSize: 4,
    gradientAccumulation: 4,
    dtype: "float16",
    gradientCheckpointing: "none" as const,
  };
  // No options at all: applying a device's sizing is opt-in, so the request the
  // wizard would send is byte-identical to the input.
  assert.deepEqual(fitApi.withTargetRecommendation(input, IGPU), {
    ...input,
    dtypeApplied: false,
  });
  assert.deepEqual(
    fitApi.withTargetRecommendation(input, IGPU, { apply: false }),
    { ...input, dtypeApplied: false },
  );
});

test("the igpu recommendation is applied to the request fields", () => {
  const input = {
    model: "unsloth/Qwen3-4B",
    batchSize: 4,
    gradientAccumulation: 4,
    gradientCheckpointing: "unsloth" as const,
  };
  const sized = fitApi.withTargetRecommendation(input, IGPU, { apply: true });
  assert.equal(sized.batchSize, 1);
  assert.equal(sized.gradientAccumulation, 8);
  // Everything the recommendation does not name rides along untouched.
  assert.equal(sized.model, "unsloth/Qwen3-4B");
  assert.equal(sized.gradientCheckpointing, "unsloth");
});

test("the diff lists exactly the fields the recommendation would change", () => {
  // Only batch differs here; accumulation, precision and checkpointing match
  // ("unsloth" is checkpointing on, which is what the recommendation asks for).
  assert.deepEqual(
    fitApi.recommendationDiff(
      {
        batchSize: 4,
        gradientAccumulation: 8,
        dtype: "bfloat16",
        gradientCheckpointing: "unsloth",
      },
      IGPU,
    ),
    [{ field: "batchSize", current: 4, recommended: 1 }],
  );
  // All four disagree: all four are listed, in the recommendation's order.
  assert.deepEqual(
    fitApi.recommendationDiff(
      {
        batchSize: 4,
        gradientAccumulation: 4,
        dtype: "float16",
        gradientCheckpointing: "none",
      },
      IGPU,
    ),
    [
      { field: "batchSize", current: 4, recommended: 1 },
      { field: "gradientAccumulation", current: 4, recommended: 8 },
      { field: "dtype", current: "float16", recommended: "bfloat16" },
      { field: "gradientCheckpointing", current: "none", recommended: true },
    ],
  );
  // An exact match is an empty diff, not a diff of equal values.
  assert.deepEqual(
    fitApi.recommendationDiff(
      {
        batchSize: 1,
        gradientAccumulation: 8,
        dtype: "bfloat16",
        gradientCheckpointing: "true",
      },
      IGPU,
    ),
    [],
  );
});

test("a target without a recommendation leaves the input alone", () => {
  const input = {
    batchSize: 4,
    gradientAccumulation: 4,
    dtype: "float16",
    gradientCheckpointing: "true" as const,
  };
  for (const target of [{}, { recommended: null }]) {
    assert.deepEqual(
      fitApi.withTargetRecommendation(input, target, { apply: true }),
      { ...input, dtypeApplied: false },
    );
    assert.deepEqual(fitApi.recommendationDiff(input, target), []);
  }
});

test("dtypeApplied is true only when the recommended precision differs", () => {
  const matching = fitApi.withTargetRecommendation(
    { batchSize: 4, gradientAccumulation: 4, dtype: "bfloat16" },
    IGPU,
    { apply: true },
  );
  assert.equal(matching.dtypeApplied, false);
  const differing = fitApi.withTargetRecommendation(
    { batchSize: 4, gradientAccumulation: 4, dtype: "float16" },
    IGPU,
    { apply: true },
  );
  assert.equal(differing.dtypeApplied, true);
  // With no precision on the input there is nothing to compare, so nothing applied.
  const unknown = fitApi.withTargetRecommendation(
    { batchSize: 4, gradientAccumulation: 4 },
    IGPU,
    { apply: true },
  );
  assert.equal(unknown.dtypeApplied, false);
});
