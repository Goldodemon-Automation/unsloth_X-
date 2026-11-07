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
  isAutomatedJobCancellable,
  isAutomatedMethod,
  normalizeAutomatedTrainingSpec,
  summarizeAutomatedProgress,
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

test("only a queued or running job is cancellable", () => {
  assert.equal(isAutomatedJobCancellable({ status: "running" }), true);
  assert.equal(isAutomatedJobCancellable({ status: "queued" }), true);
});

test("a terminal job is not cancellable", () => {
  for (const status of ["completed", "failed", "cancelled", "idle"]) {
    assert.equal(
      isAutomatedJobCancellable({ status }),
      false,
      `${status} must not offer Cancel`,
    );
  }
});

test("a missing job or status is not cancellable", () => {
  assert.equal(isAutomatedJobCancellable(null), false);
  assert.equal(isAutomatedJobCancellable(undefined), false);
  assert.equal(isAutomatedJobCancellable({}), false);
  assert.equal(isAutomatedJobCancellable({ status: "" }), false);
});

test("an unknown status is not cancellable", () => {
  for (const status of ["starting", "paused", "unknown", "runn1ng"]) {
    assert.equal(isAutomatedJobCancellable({ status }), false);
  }
  // Job records carry what the backend wrote, so case and padding are forgiven.
  assert.equal(isAutomatedJobCancellable({ status: " RUNNING " }), true);
  assert.equal(isAutomatedJobCancellable({ status: "Queued" }), true);
});

test("a training stage becomes a headline naming the round", () => {
  const summary = summarizeAutomatedProgress(
    {
      stage: "training",
      round: 2,
      message: "Round 2: training run-01-cmt on a, b.",
      label: "run-01-cmt",
      method: "cmt",
    },
    null,
  );
  assert.match(summary.headline, /training/i);
  assert.match(summary.headline, /round 2/);
  assert.equal(summary.detail, "Round 2: training run-01-cmt on a, b.");
  assert.deepEqual(summary.failures, []);
});

test("a benchmark stage becomes a benchmarking headline", () => {
  const summary = summarizeAutomatedProgress(
    { stage: "benchmark", round: 1, label: "run-02-cpt" },
    null,
  );
  assert.match(summary.headline, /benchmark/i);
  assert.match(summary.headline, /round 1/);
  // Without a message the label is the best detail available.
  assert.equal(summary.detail, "run-02-cpt");
});

test("round-done keeps the round and reports the completed rounds", () => {
  const summary = summarizeAutomatedProgress(
    { stage: "round-done", round: 2, total: 3 },
    null,
  );
  assert.match(summary.headline, /round 2/i);
  assert.match(summary.headline, /complete/i);
  assert.equal(summary.done, 2);
  assert.equal(summary.total, 3);
});

test("done and cancelled are terminal headlines with no bar", () => {
  const done = summarizeAutomatedProgress(
    { stage: "done", message: "Automated training finished." },
    null,
  );
  assert.match(done.headline, /finished/i);
  assert.equal(done.total, null);
  assert.equal(done.done, 0);

  const cancelled = summarizeAutomatedProgress(
    { stage: "cancelled", rounds: 1, total: 3 },
    null,
  );
  assert.match(cancelled.headline, /cancelled/i);
  assert.equal(cancelled.done, 1);
  assert.equal(cancelled.total, 3);
});

test("done and total parse from numbers and strings", () => {
  const summary = summarizeAutomatedProgress(
    { stage: "round-start", done: "2", total: "4" },
    null,
  );
  assert.equal(summary.done, 2);
  assert.equal(summary.total, 4);

  const clamped = summarizeAutomatedProgress(
    { stage: "round-start", done: 9, total: 4 },
    null,
  );
  assert.equal(clamped.done, 4);
  assert.equal(clamped.total, 4);
});

test("failures come from result.failed_runs", () => {
  const summary = summarizeAutomatedProgress(
    { stage: "done" },
    {
      failed_runs: ["run-01-cmt did not train", "run-02-cpt benchmark failed"],
      warnings: ["no benchmark adapter"],
    },
  );
  assert.deepEqual(summary.failures, [
    "run-01-cmt did not train",
    "run-02-cpt benchmark failed",
  ]);
});

test("a failed-run count is detailed from the round records", () => {
  const summary = summarizeAutomatedProgress(
    { stage: "done" },
    {
      failed_runs: 2,
      rounds: [
        {
          index: 1,
          failures: [
            {
              round: 1,
              label: "run-01-cmt",
              stage: "train",
              error: "ValueError: boom",
            },
          ],
        },
        { index: 2, failures: [] },
        {
          index: 3,
          failures: [
            {
              round: 3,
              label: "continue-01",
              stage: "benchmark",
              error: "TimeoutError: slow",
            },
          ],
        },
      ],
    },
  );
  assert.equal(summary.failures.length, 2);
  assert.match(summary.failures[0], /run-01-cmt/);
  assert.match(summary.failures[0], /boom/);
  assert.match(summary.failures[1], /continue-01/);
  assert.match(summary.failures[1], /slow/);
});

test("a count without detail still reports the failures", () => {
  const summary = summarizeAutomatedProgress(
    { stage: "done" },
    { failed_runs: 3 },
  );
  assert.deepEqual(summary.failures, ["3 runs failed."]);
});

test("missing progress is a safe idle summary", () => {
  for (const progress of [null, undefined]) {
    const summary = summarizeAutomatedProgress(progress, null);
    assert.deepEqual(summary, {
      headline: "Idle",
      detail: null,
      done: 0,
      total: null,
      failures: [],
    });
  }
});

test("an unknown stage falls back to a neutral headline", () => {
  const summary = summarizeAutomatedProgress({ stage: "warming-up" }, null);
  assert.equal(summary.headline, "Automated training");
  assert.equal(summary.total, null);
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
