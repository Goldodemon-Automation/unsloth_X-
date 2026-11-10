// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  TRAINING_FAILURE_KINDS,
  explainTrainingFailure,
} from "../src/features/training/lib/failure-hints.ts";

/** Verbatim shapes of what the backend puts in the run's error field. */
const BACKEND_MESSAGES: Record<string, string> = {
  // core/training/worker.py, the OOM branch of the run's top-level handler.
  oom: "GPU ran out of VRAM during training.\nTo fix: reduce max_seq_length (e.g. 2048-4096), enable gradient_checkpointing=True, lower per_device_train_batch_size, or use a smaller model / higher quantization.",
  oomTorch:
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 7.79 GiB total capacity)",
  // core/training/trainer.py, the CPU host asked for a method it cannot run.
  no_accelerator:
    "This machine has no GPU, so only CoRA and QLoRA can train here. Select one of those methods; plain LoRA, full finetuning, CPT and RL need an accelerator and would crash here.",
  // core/training/worker.py, dataset loading.
  dataset:
    "Failed to load dataset: Dataset scripts are no longer supported, but found dataset.py",
  datasetEmpty: "All 1,204 rows have an empty text cell.",
  // core/training/trainer.py, the gated-repo check on model load.
  gated:
    "Access denied for 'meta-llama/Llama-3.1-8B'. This model is gated or private. Please add a Hugging Face token with access and try again.",
  gatedHub: "OSError: 401 Client Error. You are trying to access a gated repo.",
  // core/training/disk_preflight.py's blocker, and the OS error it prevents.
  disk: "The run needs about 61.00 GB but only 12.40 GB is free on ./outputs, so it would stop mid-training.",
  diskErrno: "OSError: [Errno 28] No space left on device",
  // core/training/training.py, a run the user stopped.
  interrupted: "Training stopped.",
};

const CASES: ReadonlyArray<{
  kind: (typeof TRAINING_FAILURE_KINDS)[number];
  messages: readonly string[];
  /** Words the hint must contain: the setting or action to change. */
  mentions: readonly string[];
}> = [
  {
    kind: "oom",
    messages: [BACKEND_MESSAGES.oom, BACKEND_MESSAGES.oomTorch],
    mentions: [
      "batch size",
      "sequence length",
      "gradient checkpointing",
      "4-bit",
    ],
  },
  {
    kind: "no_accelerator",
    messages: [BACKEND_MESSAGES.no_accelerator],
    mentions: ["compute target", "qlora"],
  },
  {
    kind: "dataset",
    messages: [BACKEND_MESSAGES.dataset, BACKEND_MESSAGES.datasetEmpty],
    mentions: ["dataset", "split"],
  },
  {
    kind: "gated",
    messages: [BACKEND_MESSAGES.gated, BACKEND_MESSAGES.gatedHub],
    mentions: ["accept", "token"],
  },
  {
    kind: "disk",
    messages: [BACKEND_MESSAGES.disk, BACKEND_MESSAGES.diskErrno],
    mentions: ["output directory", "space"],
  },
  {
    kind: "interrupted",
    messages: [BACKEND_MESSAGES.interrupted],
    mentions: ["resume"],
  },
];

const SENTENCE_END = /[.!?]$/;

test("each failure the backend emits gets its own reading", () => {
  for (const { kind, messages, mentions } of CASES) {
    for (const message of messages) {
      const explanation = explainTrainingFailure({ message });
      assert.notEqual(
        explanation.title,
        "Unrecognised failure",
        `${message} was left unexplained`,
      );
      // The title is a fixed sentence for a recognised cause, not the log line.
      assert.ok(
        explanation.title.length > 0 && !message.includes(explanation.title),
        `${kind} quoted the log instead of naming the cause`,
      );
      for (const mention of mentions) {
        assert.match(
          explanation.hint.toLowerCase(),
          new RegExp(mention),
          `${kind} does not name ${mention}`,
        );
      }
    }
  }
});

test("recognised causes stay distinct from one another", () => {
  const titles = CASES.map(
    ({ messages }) => explainTrainingFailure({ message: messages[0] }).title,
  );
  assert.equal(new Set(titles).size, titles.length);
  // A gated download that failed inside a dataset load is still an access problem.
  assert.equal(
    explainTrainingFailure({
      message:
        "Failed to load dataset: 401 Client Error. You are trying to access a gated repo.",
    }).title,
    explainTrainingFailure({ message: BACKEND_MESSAGES.gated }).title,
  );
});

test("an unknown message keeps its own words and claims no cause", () => {
  const message =
    "RuntimeError: expected all tensors on same device, but got cuda:0 and cpu";
  const explanation = explainTrainingFailure({ message });
  assert.equal(explanation.title, message);
  assert.equal(explanation.retryable, true);
  const claimed = [
    "out of memory",
    "vram",
    "gated",
    "disk",
    "accelerator",
    "dataset",
    "stopped",
    "interrupted",
  ];
  const text = `${explanation.title} ${explanation.hint}`.toLowerCase();
  for (const cause of claimed) {
    assert.ok(
      !text.includes(cause),
      `an unrecognised failure claimed ${cause}`,
    );
  }
});

test("a title quotes the run, not the traceback around it", () => {
  const explanation = explainTrainingFailure({
    message: [
      "Traceback (most recent call last):",
      '  File "/env/train.py", line 12, in <module>',
      "AssertionError: heads must divide hidden_size",
    ].join("\n"),
  });
  assert.equal(
    explanation.title,
    "AssertionError: heads must divide hidden_size",
  );

  const long = explainTrainingFailure({
    message: `ValueError: ${"an explanation far too long for a heading ".repeat(6)}`,
  });
  assert.ok(long.title.length <= 91, `title was ${long.title.length} chars`);
  assert.ok(long.title.endsWith("…"));

  const blank = explainTrainingFailure({ message: "   " });
  assert.equal(blank.title, "Unrecognised failure");
  assert.equal(blank.retryable, true);
});

test("retryable is false only when the fix lives outside the run", () => {
  assert.equal(
    explainTrainingFailure({ message: BACKEND_MESSAGES.gated }).retryable,
    false,
  );
  assert.equal(
    explainTrainingFailure({ message: BACKEND_MESSAGES.disk }).retryable,
    false,
  );
  for (const kind of ["oom", "no_accelerator", "dataset", "interrupted"]) {
    const entry = CASES.find((item) => item.kind === kind);
    assert.equal(
      explainTrainingFailure({ message: (entry?.messages ?? [""])[0] })
        .retryable,
      true,
      `${kind} cannot be retried`,
    );
  }
});

test("every hint names a concrete setting or action", () => {
  const concrete = [
    "batch size",
    "sequence length",
    "4-bit",
    "gradient checkpointing",
    "compute target",
    "qlora",
    "cora",
    "dataset",
    "split",
    "token",
    "output directory",
    "save steps",
    "space",
    "resume",
    "settings",
    "fit check",
  ];
  for (const kind of TRAINING_FAILURE_KINDS) {
    const explanation = explainTrainingFailure({ kind });
    assert.ok(explanation.hint.length > 40, `${kind} has no real hint`);
    assert.ok(
      concrete.some((term) => explanation.hint.toLowerCase().includes(term)),
      `${kind} names nothing to change: ${explanation.hint}`,
    );
    assert.ok(
      SENTENCE_END.test(explanation.hint.trim()),
      `${kind}'s hint is not a sentence`,
    );
  }
});

test("a machine-readable kind outranks the text, and a new token does not silence it", () => {
  assert.deepEqual(
    explainTrainingFailure({
      message: "Training stopped.",
      kind: "out_of_memory",
    }),
    explainTrainingFailure({ kind: "oom" }),
  );
  assert.equal(
    explainTrainingFailure({ message: BACKEND_MESSAGES.oom, kind: null }).title,
    "Out of memory",
  );
  // A token this build does not know falls back to reading the message.
  assert.equal(
    explainTrainingFailure({
      message: BACKEND_MESSAGES.disk,
      kind: "some_future_token",
    }).title,
    explainTrainingFailure({ kind: "disk" }).title,
  );
  assert.equal(
    explainTrainingFailure({ message: null, kind: null }).title,
    "Unrecognised failure",
  );
});
