// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import type { AutomatedTrainingSpec } from "../api";

/** The dataset categories the automated loop can route on. */
export const AUTOMATED_CATEGORIES = [
  "code",
  "syntax",
  "reasoning",
  "general",
] as const;
export type AutomatedCategory = (typeof AUTOMATED_CATEGORIES)[number];

/** The training methods the automated loop can schedule, in a fixed order. */
export const AUTOMATED_METHODS = ["cmt", "cpt", "fft"] as const;
export type AutomatedMethod = (typeof AUTOMATED_METHODS)[number];

export interface AutomatedLimit {
  min: number;
  max: number;
}

/** One place for every numeric bound so the UI and the plan agree. */
export const AUTOMATED_LIMITS = {
  datasetsPerRound: { min: 2, max: 8 },
  runsPerRound: { min: 2, max: 72 },
  rounds: { min: 1, max: 12 },
  continueRuns: { min: 2, max: 72 },
  seed: { min: 0, max: 4_294_967_295 },
  stepsPerRun: { min: 1, max: 1_000_000 },
} as const;

/** Truncate to an integer and pull it back inside [min, max]. */
export function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

export function clampDatasetsPerRound(value: number): number {
  return clampNumber(
    value,
    AUTOMATED_LIMITS.datasetsPerRound.min,
    AUTOMATED_LIMITS.datasetsPerRound.max,
  );
}

export function clampRunsPerRound(value: number): number {
  return clampNumber(
    value,
    AUTOMATED_LIMITS.runsPerRound.min,
    AUTOMATED_LIMITS.runsPerRound.max,
  );
}

export function clampRounds(value: number): number {
  return clampNumber(
    value,
    AUTOMATED_LIMITS.rounds.min,
    AUTOMATED_LIMITS.rounds.max,
  );
}

export function clampContinueRuns(value: number): number {
  return clampNumber(
    value,
    AUTOMATED_LIMITS.continueRuns.min,
    AUTOMATED_LIMITS.continueRuns.max,
  );
}

export function clampSeed(value: number): number {
  return clampNumber(
    value,
    AUTOMATED_LIMITS.seed.min,
    AUTOMATED_LIMITS.seed.max,
  );
}

export function clampStepsPerRun(value: number): number {
  return clampNumber(
    value,
    AUTOMATED_LIMITS.stepsPerRun.min,
    AUTOMATED_LIMITS.stepsPerRun.max,
  );
}

export function isAutomatedCategory(
  value: unknown,
): value is AutomatedCategory {
  return (
    typeof value === "string" &&
    (AUTOMATED_CATEGORIES as readonly string[]).includes(value)
  );
}

export function isAutomatedMethod(value: unknown): value is AutomatedMethod {
  return (
    typeof value === "string" &&
    (AUTOMATED_METHODS as readonly string[]).includes(value)
  );
}

/**
 * A safe starting spec: two empty dataset slots so the first Plan explains what
 * is missing, and every numeric field already inside its bound.
 */
export function defaultAutomatedTrainingSpec(): AutomatedTrainingSpec {
  return {
    source: "",
    datasets: [
      { path: "", name: "", category: "general", eval_fraction: 0.1 },
      { path: "", name: "", category: "general", eval_fraction: 0.1 },
    ],
    random_datasets: false,
    methods: ["cmt", "cpt"],
    datasets_per_round: 2,
    runs_per_round: 4,
    rounds: 3,
    continue_runs: 4,
    seed: 42,
    steps_per_run: 100,
  };
}

/** Clamp every numeric field and drop unknown categories/methods in place. */
export function normalizeAutomatedTrainingSpec(
  spec: AutomatedTrainingSpec,
): AutomatedTrainingSpec {
  return {
    ...spec,
    random_datasets: spec.random_datasets ?? false,
    datasets: (spec.datasets ?? []).map((dataset) => ({
      ...dataset,
      category: isAutomatedCategory(dataset.category)
        ? dataset.category
        : "general",
    })),
    methods: (spec.methods ?? []).filter(isAutomatedMethod),
    datasets_per_round: clampDatasetsPerRound(spec.datasets_per_round ?? 2),
    runs_per_round: clampRunsPerRound(spec.runs_per_round ?? 4),
    rounds: clampRounds(spec.rounds ?? 3),
    continue_runs: clampContinueRuns(spec.continue_runs ?? 4),
    seed: clampSeed(spec.seed ?? 42),
    steps_per_run: clampStepsPerRun(spec.steps_per_run ?? 100),
  };
}

/**
 * Return every problem that would stop a plan from running. An empty array
 * means the spec is runnable; the order is stable so a caller can show the
 * first message as the primary reason.
 */
export function validateAutomatedTrainingSpec(
  spec: AutomatedTrainingSpec,
): string[] {
  const errors: string[] = [];
  if (!spec.source || spec.source.trim() === "") {
    errors.push("A source checkpoint is required.");
  }

  const datasets = spec.datasets ?? [];
  if (!spec.random_datasets) {
    if (datasets.length < AUTOMATED_LIMITS.datasetsPerRound.min) {
      errors.push("Automated training needs at least two datasets.");
    }
    if (datasets.length > AUTOMATED_LIMITS.datasetsPerRound.max) {
      errors.push("Automated training accepts at most eight datasets.");
    }
    datasets.forEach((dataset, index) => {
      const hasPath = Boolean(dataset.path && dataset.path.trim() !== "");
      const hasRepo = Boolean(dataset.repo_id && dataset.repo_id.trim() !== "");
      if (!hasPath && !hasRepo) {
        errors.push(`Dataset ${index + 1} needs a path or a Hub repo id.`);
      }
      if (
        dataset.category !== undefined &&
        !isAutomatedCategory(dataset.category)
      ) {
        errors.push(`Dataset ${index + 1} has an unknown category.`);
      }
    });
  }

  const methods = spec.methods ?? [];
  if (methods.length === 0) {
    errors.push("Pick at least one training method.");
  }
  methods.forEach((method) => {
    if (!isAutomatedMethod(method)) {
      errors.push(`Unknown training method: ${String(method)}.`);
    }
  });

  const numeric: Array<[string, number | undefined, AutomatedLimit]> = [
    [
      "Datasets per round",
      spec.datasets_per_round,
      AUTOMATED_LIMITS.datasetsPerRound,
    ],
    ["Runs per round", spec.runs_per_round, AUTOMATED_LIMITS.runsPerRound],
    ["Rounds", spec.rounds, AUTOMATED_LIMITS.rounds],
    ["Continue runs", spec.continue_runs, AUTOMATED_LIMITS.continueRuns],
    ["Seed", spec.seed, AUTOMATED_LIMITS.seed],
    ["Steps per run", spec.steps_per_run, AUTOMATED_LIMITS.stepsPerRun],
  ];
  for (const [label, value, limit] of numeric) {
    if (value === undefined) continue;
    if (!Number.isFinite(value) || value < limit.min || value > limit.max) {
      errors.push(`${label} must be between ${limit.min} and ${limit.max}.`);
    }
  }

  return errors;
}
