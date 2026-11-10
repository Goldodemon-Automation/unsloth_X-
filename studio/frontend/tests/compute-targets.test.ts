// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type * as ComputeApi from "../src/features/compute-targets/api.ts";
import { useComputeTargetStore } from "../src/features/compute-targets/compute-target-store.ts";
import {
  type ComputeTarget,
  type ComputeTargetsResponse,
  type NpuTrainingCapability,
  computeTargetError,
  computeTargetKindLabel,
  describeReason,
  describeReasons,
  formatMemoryBytes,
  isTargetSelectable,
} from "../src/features/compute-targets/types.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const CPU_TARGET: ComputeTarget = {
  id: "cpu",
  device: "cpu",
  kind: "cpu",
  name: "AMD Ryzen 9 7950X",
  vendor: "amd",
  memory_bytes: 68719476736,
  unified_memory: true,
  supported: true,
  reasons: [],
  capabilities: {
    bf16: false,
    fp16: false,
    bnb4bit: false,
    triton: false,
    grad_checkpoint: true,
    compile: false,
    flash_attn: false,
  },
  recommended: {
    batch_size: 1,
    grad_accum: 16,
    dtype: "float32",
    gradient_checkpointing: true,
  },
  notes: [
    "No accelerator: training runs on the CPU and is roughly 50-100x slower than a GPU run.",
  ],
};

const UNREACHABLE = /Could not reach Unsloth Studio/;

function target(overrides: Partial<ComputeTarget>): ComputeTarget {
  return { ...CPU_TARGET, ...overrides };
}

function npu(mode: string): NpuTrainingCapability {
  return {
    mode,
    backward_pass: false,
    tasks: ["tokenize", "eval"],
    reasons: ["no_backward_pass"],
    notes:
      "The NPU has no backward pass, so training uses it for tokenize and eval only.",
    providers: [],
  };
}

const REASONS = [
  "arch_unknown",
  "no_backward_pass",
  "disabled",
  "absent",
  "unsupported_family",
  "no_triton",
  "no_bnb",
  "npu_is_primary",
  "no_runtime",
];

function client(reply: () => Response | Promise<Response>) {
  const requests: { path: string; init?: RequestInit }[] = [];
  const api = loadWithStubs<typeof ComputeApi>(
    new URL("../src/features/compute-targets/api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: (path: string, init?: RequestInit) => {
          requests.push({ path, init });
          return Promise.resolve(reply());
        },
      },
      "@/lib/format-fastapi-error": formatFastApiError,
    },
    { relativePassthrough: true },
  );
  return { api, requests };
}

async function rejection(run: () => Promise<unknown>): Promise<unknown> {
  try {
    await run();
  } catch (error) {
    return error;
  }
  throw new Error("expected the call to reject");
}

test("every reason token reads as a sentence", () => {
  for (const token of REASONS) {
    const sentence = describeReason(token);
    assert.notEqual(sentence, token);
    assert.equal(sentence.includes("_"), false);
    assert.equal(sentence.endsWith("."), true);
  }
});

test("an unknown reason token falls back to itself", () => {
  assert.equal(describeReason("hyperdrive_offline"), "hyperdrive_offline");
});

test("a target's reasons read as prose", () => {
  assert.equal(describeReasons([]), "");
  assert.equal(
    describeReasons(["absent", "nope"]),
    `${describeReason("absent")} nope`,
  );
});

test("a device's memory reads in GB to one decimal", () => {
  assert.equal(formatMemoryBytes(null), "—");
  assert.equal(formatMemoryBytes(0), "0.0 GB");
  assert.equal(formatMemoryBytes(536870912), "0.5 GB");
  assert.equal(formatMemoryBytes(1073741824), "1.0 GB");
  assert.equal(formatMemoryBytes(68719476736), "64.0 GB");
});

test("each device kind has a badge, and an unknown one keeps its own name", () => {
  assert.equal(computeTargetKindLabel(target({ kind: "cpu" })), "CPU");
  assert.equal(
    computeTargetKindLabel(target({ kind: "igpu" })),
    "Integrated GPU",
  );
  assert.equal(computeTargetKindLabel(target({ kind: "dgpu" })), "GPU");
  assert.equal(computeTargetKindLabel(target({ kind: "npu" })), "NPU");
  assert.equal(
    computeTargetKindLabel(target({ kind: "hole_punch" })),
    "hole_punch",
  );
});

test("an unsupported target stays listed and is never selectable", () => {
  assert.equal(isTargetSelectable(target({}), npu("assist")), true);
  assert.equal(isTargetSelectable(target({}), null), true);
  assert.equal(
    isTargetSelectable(
      target({
        id: "dgpu",
        kind: "dgpu",
        supported: false,
        reasons: ["no_triton"],
      }),
      npu("assist"),
    ),
    false,
  );
});

test("an unavailable NPU cannot be picked, and blocks nothing else", () => {
  const npuTarget = target({ id: "npu", kind: "npu" });
  assert.equal(isTargetSelectable(npuTarget, npu("unavailable")), false);
  assert.equal(isTargetSelectable(npuTarget, npu("assist")), true);
  assert.equal(isTargetSelectable(target({}), npu("unavailable")), true);
});

test("reading the targets takes the route's answer", async () => {
  const body: ComputeTargetsResponse = {
    targets: [CPU_TARGET],
    default: "cpu",
    resolved: "cpu",
    npu: npu("assist"),
    budget: {
      integrated: true,
      dedicated_bytes: 536870912,
      system_memory_bytes: 68719476736,
      shared_fraction: 0.6,
      usable_bytes: 41231686041,
      notes: "This GPU shares system memory.",
    },
  };
  const c = client(() => Response.json(body));
  assert.deepEqual(await c.api.getComputeTargets(), body);
  assert.equal(c.requests[0].path, "/api/compute/targets");
});

test("choosing a target posts the contract's body and takes the reply", async () => {
  const chosen = target({ id: "igpu", kind: "igpu", name: "Radeon 780M" });
  const c = client(() => Response.json({ target: chosen, resolved: "igpu" }));
  assert.deepEqual(await c.api.setComputeTarget("igpu"), {
    target: chosen,
    resolved: "igpu",
  });
  assert.equal(c.requests[0].path, "/api/compute/target");
  assert.equal(c.requests[0].init?.method, "POST");
  assert.equal(c.requests[0].init?.body, JSON.stringify({ target: "igpu" }));
});

test("a FastAPI detail reaches the UI as the rejection message", async () => {
  const c = client(() =>
    Response.json(
      { detail: "The NPU runtime is not installed." },
      { status: 400 },
    ),
  );
  const error = await rejection(() => c.api.setComputeTarget("npu"));
  assert.deepEqual(computeTargetError(error), {
    message: "The NPU runtime is not installed.",
    code: "http_400",
  });
});

test("an unreachable backend rejects with a sentence, not a bare failure", async () => {
  const c = client(() => {
    throw new TypeError("Failed to fetch");
  });
  const error = await rejection(() => c.api.getComputeTargets());
  const { message, code } = computeTargetError(error);
  assert.equal(code, "network_unreachable");
  assert.match(message, UNREACHABLE);
});

test("an error with nothing readable still carries a code and a sentence", () => {
  assert.deepEqual(computeTargetError({}), {
    message: "The compute targets could not be read from the backend.",
    code: "request_failed",
  });
  assert.deepEqual(computeTargetError(undefined), {
    message: "The compute targets could not be read from the backend.",
    code: "request_failed",
  });
  assert.deepEqual(computeTargetError("Disk full"), {
    message: "Disk full",
    code: "request_failed",
  });
});

test("the selection store holds one id and can clear it", () => {
  useComputeTargetStore.getState().setSelected("igpu");
  assert.equal(useComputeTargetStore.getState().selectedId, "igpu");
  useComputeTargetStore.getState().setSelected(null);
  assert.equal(useComputeTargetStore.getState().selectedId, null);
});
