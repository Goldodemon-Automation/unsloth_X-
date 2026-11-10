// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What a failed training run means, in one testable place.
 *
 * A failure arrives as free text. The backend passes torch's own wording for an
 * out-of-memory (`core/training/worker.py`), unsloth's wording for a host with no
 * accelerator (`core/training/trainer.py`), the Hub's for a gated repo, and the disk
 * preflight's sentence for a run that would not fit on disk
 * (`core/training/disk_preflight.py`). Shown verbatim, that is a log to read rather
 * than a setting to change.
 *
 * So each recognised failure names the concrete setting to change, in the same
 * vocabulary the fit check uses: batch size, sequence length, 4-bit, gradient
 * checkpointing. An unrecognised message keeps its own text as the title and claims
 * no cause — a guess the user then acts on is worse than none.
 *
 * Pure and dependency-free, so a node test imports it directly, the way the
 * compute-target contract does (`src/features/compute-targets/types.ts`).
 *
 * It sits beside `training-failure-postmortem.ts`, which buckets a log into a kind
 * (oom, nan, disk, auth, dataset, timeout) for translated card copy. This module
 * names the causes that list has no bucket for — a host with no accelerator, a run
 * the user stopped — and adds the answer the start actions need: whether running it
 * again unchanged could help. Neither reads the other, so a test pins each on its
 * own.
 */

/** The failures this module can name, plus the case it leaves unexplained. */
export type TrainingFailureKind =
  | "oom"
  | "no_accelerator"
  | "dataset"
  | "gated"
  | "disk"
  | "interrupted"
  | "unknown";

/** Everything except the unrecognised case: the failures with a fixed reading. */
export type RecognisedTrainingFailureKind = Exclude<
  TrainingFailureKind,
  "unknown"
>;

/** Every kind `explainTrainingFailure` can answer with, for exhaustive tests. */
export const TRAINING_FAILURE_KINDS: readonly TrainingFailureKind[] = [
  "oom",
  "no_accelerator",
  "dataset",
  "gated",
  "disk",
  "interrupted",
  "unknown",
];

export interface TrainingFailureInput {
  /** The backend's error text, as it arrived. */
  message?: string | null;
  /**
   * A machine-readable token the backend sent with the failure (`error_kind`,
   * `error_code`), when it sent one. It outranks the text; an unknown token is
   * ignored, so a newer backend still gets a reading of its message rather than
   * nothing.
   */
  kind?: string | null;
}

export interface TrainingFailureExplanation {
  /** One short line: what happened, or the run's own words when nothing matched. */
  title: string;
  /** The setting or action that changes the outcome, in the fit check's vocabulary. */
  hint: string;
  /**
   * Whether starting this run again, unchanged, could get anywhere. False means the
   * fix lives outside the run: a license to accept, or space to free.
   */
  retryable: boolean;
}

interface FailureRule {
  kind: RecognisedTrainingFailureKind;
  patterns: readonly string[];
}

/**
 * Ordered: the narrowest readings first, so one string cannot answer for two causes.
 * A gated download and a full disk both fail inside a dataset or model load, and
 * their advice is nothing like the load's own.
 */
const RULES: readonly FailureRule[] = [
  {
    kind: "interrupted",
    patterns: [
      "training stopped",
      "training cancelled",
      "training canceled",
      "training start was cancelled",
      "stopped by the user",
      "stopped by user",
      "cancelled by the user",
      "canceled by the user",
      "interrupted by the user",
      "user requested stop",
      "stop requested",
      "keyboardinterrupt",
    ],
  },
  {
    kind: "gated",
    patterns: [
      // Never a bare "gated": "aggregated" and "mitigated" both contain it.
      "gated repo",
      "gatedrepo",
      "gated or private",
      "gated hugging face",
      "access denied",
      "access to model",
      "access to it at",
      "accept its license",
      "401 client error",
      "403 client error",
      "unauthorized",
      "invalid token",
      "forbidden",
    ],
  },
  {
    kind: "disk",
    patterns: [
      "no space left",
      "disk quota",
      "errno 28",
      "os error 112",
      "edquot",
      "enospc",
      "not enough space",
      "disk full",
      "out of disk",
      "free disk space",
      "would stop mid-training",
      "is free on",
    ],
  },
  {
    kind: "oom",
    patterns: [
      "out of memory",
      "outofmemoryerror",
      "out of vram",
      "ran out of vram",
      "unable to allocate",
      "cannot allocate",
      "memory allocation failed",
      "cublas_status_alloc_failed",
      "insufficient memory",
      "not enough memory",
      "does not fit",
    ],
  },
  {
    kind: "no_accelerator",
    patterns: [
      "no accelerator",
      "has no gpu",
      "no gpu",
      "need an accelerator",
      "requires apple silicon",
      "cuda is not available",
      "mps is not available",
      "no compatible device",
      "no backward pass",
    ],
  },
  {
    kind: "dataset",
    patterns: [
      "failed to load dataset",
      "could not load dataset",
      "failed to check dataset format",
      "dataset format",
      "dataset scripts are no longer supported",
      "dataset snapshot",
      "no local dataset files",
      "no dataset specified",
      "no columns",
      "empty dataset",
      "empty text cell",
      "rows have an empty",
    ],
  },
];

/**
 * Machine tokens this build understands, including the ones the existing failure
 * diagnosis and the Hub errors use ("auth" for any access refusal, "nan" never
 * mapped — a non-finite loss has no reading here yet, so it stays unclaimed).
 */
const KIND_TOKENS: Record<string, RecognisedTrainingFailureKind | undefined> = {
  oom: "oom",
  out_of_memory: "oom",
  vram: "oom",
  auth: "gated",
  gated: "gated",
  gated_repo: "gated",
  access_denied: "gated",
  disk: "disk",
  disk_full: "disk",
  no_space: "disk",
  dataset: "dataset",
  dataset_load: "dataset",
  no_accelerator: "no_accelerator",
  no_device: "no_accelerator",
  cpu_only: "no_accelerator",
  interrupted: "interrupted",
  cancelled: "interrupted",
  canceled: "interrupted",
  stopped: "interrupted",
  user_stopped: "interrupted",
  training_start_cancelled: "interrupted",
};

const EXPLANATIONS: Record<
  RecognisedTrainingFailureKind,
  TrainingFailureExplanation
> = {
  oom: {
    title: "Out of memory",
    hint: "This run's peak went over the device's memory. Drop the batch size to 1 and cut the sequence length to 1024 or 2048 — activations scale with both — turn on gradient checkpointing, or load in 4-bit (QLoRA), then start the run again.",
    retryable: true,
  },
  no_accelerator: {
    title: "No accelerator for this run",
    hint: "This training method needs a device with a backward pass. Switch the method to QLoRA or CoRA, which train on the CPU, or pick a different Compute target, then start the run again.",
    retryable: true,
  },
  dataset: {
    title: "The dataset could not be read",
    hint: "Re-check the dataset's name, subset and split under Training data, and the columns the rows are mapped to — every row needs text in them. Pick another dataset if the rows themselves are the problem, then start the run again.",
    retryable: true,
  },
  gated: {
    title: "Access to this model is gated",
    hint: "Accept the model's terms on Hugging Face with the account your token belongs to, then set that token under Settings > Hugging Face. Starting the run before both are done fails the same way.",
    retryable: false,
  },
  disk: {
    title: "No room left on the outputs disk",
    hint: "Free space on the drive holding the outputs folder, or point Output directory at a larger disk, and save less often (a larger Save steps keeps fewer checkpoints around). The run cannot start until the files fit.",
    retryable: false,
  },
  interrupted: {
    title: "Interrupted by the user",
    hint: "Nothing failed: this run was stopped. Start it again with Resume to keep the steps it already trained, or without it to train from scratch.",
    retryable: true,
  },
};

/**
 * The unrecognised case is a reading of the message, not a fixed card: the title
 * quotes the run, the hint offers the settings to check without blaming any of them.
 */
const UNKNOWN_TITLE = "Unrecognised failure";
const UNKNOWN_HINT =
  "Nothing in this log matches a failure Unsloth Studio knows, so no cause is claimed. Compare the run's own shape with the fit check — batch size, sequence length, 4-bit loading, gradient checkpointing — and fix what the log names before starting it again.";

/** Python prints tracebacks; a title should not open with one. */
const LOG_NOISE_PREFIXES = [
  "traceback (most recent call last)",
  'file "',
  "during handling of the above exception",
  "the above exception was the direct cause",
];

const TITLE_MAX_CHARS = 90;

const TRAILING_PUNCTUATION = /[,:;.]$/;

function normalize(text: string | null | undefined): string {
  return (text ?? "").trim().toLowerCase();
}

/**
 * The run's own first real line, as the title. Nothing is added to it: an
 * unrecognised message is quoted, not explained. `text` is non-empty.
 */
function titleFromMessage(text: string): string {
  const trimmed = text.trim();
  const line = trimmed
    .split("\n")
    .map((entry) => entry.trim())
    .find(
      (entry) =>
        entry.length > 0 &&
        !LOG_NOISE_PREFIXES.some((noise) =>
          entry.toLowerCase().startsWith(noise),
        ),
    );
  const subject = line ?? trimmed;
  if (subject.length <= TITLE_MAX_CHARS) {
    return subject;
  }
  const cut = subject.slice(0, TITLE_MAX_CHARS);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > 40 ? cut.slice(0, lastSpace) : cut).replace(TRAILING_PUNCTUATION, "")}…`;
}

function kindFromToken(
  kind: string | null | undefined,
): RecognisedTrainingFailureKind | null {
  const token = normalize(kind);
  if (token.length === 0) {
    return null;
  }
  return KIND_TOKENS[token] ?? null;
}

function kindFromMessage(text: string): RecognisedTrainingFailureKind | null {
  for (const rule of RULES) {
    if (rule.patterns.some((pattern) => text.includes(pattern))) {
      return rule.kind;
    }
  }
  return null;
}

/**
 * Explain a failed run. Always answers: unlike a diagnosis, which can stay silent
 * when it has nothing to say, this feeds a block that is only shown once there is an
 * error on screen.
 */
export function explainTrainingFailure(
  input: TrainingFailureInput,
): TrainingFailureExplanation {
  const message = (input.message ?? "").trim();
  const kind =
    kindFromToken(input.kind) ?? kindFromMessage(message.toLowerCase());
  if (kind !== null) {
    return EXPLANATIONS[kind];
  }
  return {
    title: message.length > 0 ? titleFromMessage(message) : UNKNOWN_TITLE,
    hint: UNKNOWN_HINT,
    retryable: true,
  };
}
