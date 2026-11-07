// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import assert from "node:assert/strict";
import test from "node:test";
import {
  AUTOMATED_CATEGORIES,
  AUTOMATED_LIMITS,
  AUTOMATED_METHODS,
  clampContinueRuns,
  clampDatasetsPerRound,
  clampNumber,
  clampRounds,
  clampRunsPerRound,
  clampSeed,
  clampStepsPerRun,
  defaultAutomatedTrainingSpec,
  isAutomatedCategory,
  isAutomatedMethod,
  normalizeAutomatedTrainingSpec,
  validateAutomatedTrainingSpec,
} from "../src/features/labs/lib/automated-training-spec.ts";

type Spec = ReturnType<typeof defaultAutomatedTrainingSpec>;

function validSpec(): Spec {
  return {
    ...defaultAutomatedTrainingSpec(),
    source: "C:/models/base",
    datasets: [
      {
        path: "C:/data/a.jsonl",
        name: "a",
        category: "code",
        eval_fraction: 0.1,
      },
      {
        path: "C:/data/b.jsonl",
        name: "b",
        category: "reasoning",
        eval_fraction: 0.2,
      },
    ],
  };
}

test("default spec starts with two dataset slots inside every bound", () => {
  const spec = defaultAutomatedTrainingSpec();
  assert.equal(spec.datasets.length, 2);
  assert.ok(spec.datasets.length >= AUTOMATED_LIMITS.datasetsPerRound.min);
  assert.ok(spec.datasets.length <= AUTOMATED_LIMITS.datasetsPerRound.max);
  assert.equal(spec.datasets_per_round, 2);
  assert.equal(spec.runs_per_round, 4);
  assert.ok(spec.runs_per_round! >= AUTOMATED_LIMITS.runsPerRound.min);
  assert.ok(spec.runs_per_round! <= AUTOMATED_LIMITS.runsPerRound.max);
  assert.ok(spec.rounds! >= AUTOMATED_LIMITS.rounds.min);
  assert.ok(spec.rounds! <= AUTOMATED_LIMITS.rounds.max);
});

test("clampNumber truncates and pulls both ends into range", () => {
  assert.equal(clampNumber(5.9, 0, 10), 5);
  assert.equal(clampNumber(-4, 0, 10), 0);
  assert.equal(clampNumber(40, 0, 10), 10);
  assert.equal(clampNumber(Number.NaN, 3, 10), 3);
});

test("datasets per round clamps to 2-8", () => {
  assert.equal(clampDatasetsPerRound(0), 2);
  assert.equal(clampDatasetsPerRound(1), 2);
  assert.equal(clampDatasetsPerRound(5), 5);
  assert.equal(clampDatasetsPerRound(99), 8);
});

test("runs per round and continue runs clamp to 2-72", () => {
  assert.equal(clampRunsPerRound(1), 2);
  assert.equal(clampRunsPerRound(100), 72);
  assert.equal(clampContinueRuns(0), 2);
  assert.equal(clampContinueRuns(72), 72);
  assert.equal(clampContinueRuns(500), 72);
});

test("rounds clamp to 1-12 and seed/steps clamp to their bounds", () => {
  assert.equal(clampRounds(0), 1);
  assert.equal(clampRounds(50), 12);
  assert.equal(clampSeed(-1), 0);
  assert.equal(clampSeed(4_294_967_296), 4_294_967_295);
  assert.equal(clampStepsPerRun(0), 1);
  assert.equal(clampStepsPerRun(5_000_000), 1_000_000);
});

test("categories are limited to the four known values", () => {
  assert.deepEqual([...AUTOMATED_CATEGORIES], [
    "code",
    "syntax",
    "reasoning",
    "general",
  ]);
  for (const category of AUTOMATED_CATEGORIES) {
    assert.equal(isAutomatedCategory(category), true);
  }
  assert.equal(isAutomatedCategory("math"), false);
  assert.equal(isAutomatedCategory(""), false);
  assert.equal(isAutomatedCategory(3), false);
});

test("methods are limited to cmt, cpt and fft", () => {
  assert.deepEqual([...AUTOMATED_METHODS], ["cmt", "cpt", "fft"]);
  for (const method of AUTOMATED_METHODS) {
    assert.equal(isAutomatedMethod(method), true);
  }
  assert.equal(isAutomatedMethod("cora"), false);
  assert.equal(isAutomatedMethod(null), false);
});

test("a valid spec passes validation with no errors", () => {
  assert.deepEqual(validateAutomatedTrainingSpec(validSpec()), []);
});

test("a dataset without a path or repo id is rejected", () => {
  const spec = validSpec();
  spec.datasets[0] = { name: "empty" };
  const errors = validateAutomatedTrainingSpec(spec);
  assert.ok(errors.some((error) => error.includes("Dataset 1")));
});

test("an unknown category and method are both reported", () => {
  const spec = validSpec();
  spec.datasets[0].category = "math" as never;
  spec.methods = ["cpt", "gan" as never];
  const errors = validateAutomatedTrainingSpec(spec);
  assert.ok(errors.some((error) => error.includes("unknown category")));
  assert.ok(errors.some((error) => error.includes("Unknown training method")));
});

test("out-of-range numbers produce one error per field", () => {
  const bad = {
    source: "",
    datasets: [],
    methods: [],
    datasets_per_round: 1,
    runs_per_round: 100,
    rounds: 0,
    continue_runs: 1,
    seed: -5,
    steps_per_run: -1,
  } as unknown as Spec;
  const errors = validateAutomatedTrainingSpec(bad);
  assert.ok(errors.some((error) => error.includes("source checkpoint")));
  assert.ok(errors.some((error) => error.includes("at least two datasets")));
  assert.ok(errors.some((error) => error.includes("at least one")));
  assert.ok(errors.some((error) => error.includes("Datasets per round")));
  assert.ok(errors.some((error) => error.includes("Runs per round")));
  assert.ok(errors.some((error) => error.includes("Rounds")));
  assert.ok(errors.some((error) => error.includes("Continue runs")));
  assert.ok(errors.some((error) => error.includes("Seed")));
  assert.ok(errors.some((error) => error.includes("Steps per run")));
  assert.ok(errors.length >= 9);
});

test("normalize clamps numbers and coerces unknown values", () => {
  const spec = defaultAutomatedTrainingSpec();
  const out = normalizeAutomatedTrainingSpec({
    ...spec,
    datasets_per_round: 99,
    runs_per_round: 1,
    rounds: 50,
    continue_runs: 0,
    seed: -3,
    steps_per_run: 5_000_000,
    methods: ["cpt", "bogus" as never],
    datasets: [{ path: "C:/data/x.jsonl", category: "bogus" as never }],
  });
  assert.equal(out.datasets_per_round, 8);
  assert.equal(out.runs_per_round, 2);
  assert.equal(out.rounds, 12);
  assert.equal(out.continue_runs, 2);
  assert.equal(out.seed, 0);
  assert.equal(out.steps_per_run, 1_000_000);
  assert.deepEqual(out.methods, ["cpt"]);
  assert.equal(out.datasets[0].category, "general");
});
