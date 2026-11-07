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
 * Whether a Cancel button should be live for this job. Only a job that is still
 * waiting or running has work left to stop; a finished, failed or cancelled job
 * has none, and an unknown status must not offer an action the route answers
 * with a 404. Case and surrounding spaces are tolerated because the value comes
 * from a job record, not from this UI.
 */
export function isAutomatedJobCancellable(
  job: { status?: string } | null | undefined,
): boolean {
  const status = job?.status;
  if (typeof status !== "string") return false;
  const normalized = status.trim().toLowerCase();
  return normalized === "running" || normalized === "queued";
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
 * The shape the panel renders while a job runs and once it finished. `done` and
 * `total` are rounds; `total` is null until the engine tells us how many rounds
 * the plan has, so a caller can hide the bar instead of showing a fake one.
 */
export interface AutomatedProgressSummary {
  headline: string;
  detail: string | null;
  done: number;
  total: number | null;
  failures: string[];
}

/** Engine stage name -> the short line the panel shows. */
const AUTOMATED_STAGE_HEADLINE: Record<string, string> = {
  queued: "Starting automated training",
  "round-start": "Starting a round",
  training: "Training",
  benchmark: "Benchmarking",
  pick: "Picking the round winner",
  continue: "Continue-training",
  "round-done": "Round complete",
  done: "Automated training finished",
  cancelled: "Automated training cancelled",
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function asText(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

/** One engine failure record -> one readable line. */
function describeFailure(failure: Record<string, unknown>): string | null {
  const error = asText(failure.error) ?? asText(failure.message);
  const round = asNumber(failure.round);
  const label = asText(failure.label);
  const stage = asText(failure.stage);
  const parts: string[] = [];
  if (round !== null) parts.push(`Round ${round}`);
  if (label) parts.push(label);
  if (stage) parts.push(stage);
  const prefix = parts.join(" ");
  if (!error) return prefix ? `${prefix} failed.` : null;
  return prefix ? `${prefix}: ${error}` : error;
}

/**
 * Every failed run a finished result knows about. The engine reports the count
 * in `failed_runs` and the detail per round; a plain string list is also
 * accepted so a caller can hand over an already-flattened summary.
 */
function failureMessages(result: Record<string, unknown> | null): string[] {
  if (!result) return [];
  const failures: string[] = [];
  const seen = new Set<string>();
  const push = (message: string | null) => {
    const text = message?.trim();
    if (!text || seen.has(text)) return;
    seen.add(text);
    failures.push(text);
  };

  const reported = result.failed_runs;
  if (Array.isArray(reported)) {
    for (const entry of reported) {
      if (typeof entry === "string") {
        push(entry);
        continue;
      }
      const record = asRecord(entry);
      push(record ? describeFailure(record) : null);
    }
    return failures;
  }

  const rounds = Array.isArray(result.rounds) ? result.rounds : [];
  for (const entry of rounds) {
    const round = asRecord(entry);
    if (!round || !Array.isArray(round.failures)) continue;
    for (const failure of round.failures) {
      const record = asRecord(failure);
      if (record) push(describeFailure(record));
    }
  }

  const declared = asNumber(reported);
  if (declared !== null && declared > failures.length) {
    const missing = declared - failures.length;
    push(missing === 1 ? "1 run failed." : `${missing} runs failed.`);
  }
  return failures;
}

/** True once the result carries anything only a finished job can know. */
function looksFinished(result: Record<string, unknown> | null): boolean {
  if (!result) return false;
  return (
    Array.isArray(result.rounds) ||
    asRecord(result.best) !== null ||
    asNumber(result.train_runs) !== null ||
    result.cancelled !== undefined
  );
}

/**
 * Turn the engine's stage names into one headline plus an optional detail line,
 * a done/total pair for a progress bar and the failure lines of a finished
 * result. Pure: no React and no i18n, so the panel and the tests share it.
 */
export function summarizeAutomatedProgress(
  progress: Record<string, unknown> | null | undefined,
  result: Record<string, unknown> | null | undefined,
): AutomatedProgressSummary {
  const finished = result ?? null;
  const failures = failureMessages(finished);

  if (!progress) {
    if (looksFinished(finished)) {
      const cancelled = finished?.cancelled === true;
      const rounds = Array.isArray(finished?.rounds)
        ? finished.rounds.length
        : null;
      const best = asRecord(finished?.best);
      return {
        headline: cancelled
          ? "Automated training cancelled"
          : "Automated training finished",
        detail: asText(best?.output_dir) ?? asText(finished?.output_dir),
        done: rounds ?? 0,
        total: rounds,
        failures,
      };
    }
    return { headline: "Idle", detail: null, done: 0, total: null, failures };
  }

  const stage = asText(progress.stage)?.toLowerCase().replaceAll("_", "-");
  const round = asNumber(progress.round);
  const base = stage ? AUTOMATED_STAGE_HEADLINE[stage] : undefined;
  const terminal = stage === "done" || stage === "cancelled";
  const headline = !base
    ? "Automated training"
    : round === null || terminal
      ? base
      : `${base} \u00b7 round ${round}`;

  const explicitDone = asNumber(progress.done);
  const completedRounds = asNumber(progress.rounds);
  let done = explicitDone ?? completedRounds ?? 0;
  if (explicitDone === null && completedRounds === null && round !== null) {
    done = stage === "round-done" ? round : Math.max(0, round - 1);
  }
  done = Math.max(0, Math.trunc(done));

  const reportedTotal = asNumber(progress.total);
  const total =
    reportedTotal !== null && reportedTotal > 0
      ? Math.trunc(reportedTotal)
      : null;

  return {
    headline,
    detail:
      asText(progress.message) ??
      asText(progress.label) ??
      asText(progress.method),
    done: total === null ? done : Math.min(done, total),
    total,
    failures,
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
