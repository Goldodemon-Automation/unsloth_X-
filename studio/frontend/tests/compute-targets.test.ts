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
import type * as ComputeTargetHooks from "../src/features/compute-targets/use-compute-targets.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

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
  // The registry emits this one; it rendered as the raw token until it was added here.
  "torch_missing",
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

// ---------------------------------------------------------------------------
// The "Run on" surface fails gracefully: a probe that fails leaves a retryable
// notice behind, and retrying must not be answered by the probe it replaced.

interface NoticeModule {
  ComputeTargetsNotice: (props: {
    error: string | null;
    loading: boolean;
    onRetry: () => void;
    hasData?: boolean;
  }) => StubElement | null;
  noticeState: (input: {
    error: string | null;
    loading: boolean;
    hasData: boolean;
  }) => "loading" | "error" | "none";
}

const notice = loadWithStubs<NoticeModule>(
  new URL(
    "../src/features/compute-targets/compute-targets-notice.tsx",
    import.meta.url,
  ),
  {
    "react/jsx-runtime": stubJsxRuntime(),
    "@/lib/utils": {
      cn: (...parts: unknown[]) => parts.filter(Boolean).join(" "),
    },
    "@/components/ui/button": {
      Button: (props: Record<string, unknown>) => props,
    },
  },
);

function targetsBody(): ComputeTargetsResponse {
  return {
    targets: [CPU_TARGET],
    default: "cpu",
    resolved: "cpu",
    npu: npu("assist"),
    budget: {
      integrated: true,
      dedicated_bytes: null,
      system_memory_bytes: 68719476736,
      shared_fraction: null,
      usable_bytes: 68719476736,
      notes: "Training on the CPU plans against system RAM.",
    },
  };
}

/** A hook whose probe answers only when the test says so, one queued reply per request. */
function targetsHook() {
  const pending: {
    resolve: (response: Response) => void;
    reject: (error: unknown) => void;
  }[] = [];
  const c = client(
    () =>
      new Promise<Response>((resolve, reject) => {
        pending.push({ resolve, reject });
      }),
  );
  const published: ComputeTargetHooks.ComputeTargetsSnapshot[] = [];
  const hooks = loadWithStubs<typeof ComputeTargetHooks>(
    new URL(
      "../src/features/compute-targets/use-compute-targets.ts",
      import.meta.url,
    ),
    {
      react: {
        useState: (initial: unknown) => {
          const value =
            typeof initial === "function"
              ? (initial as () => unknown)()
              : initial;
          return [
            value,
            (next: unknown) => {
              published.push(next as ComputeTargetHooks.ComputeTargetsSnapshot);
            },
          ];
        },
        // Run the effect so the hook really subscribes to the module's publishes.
        useEffect: (effect: () => unknown) => {
          effect();
        },
        useCallback: <T>(fn: T): T => fn,
      },
      "./api": c.api,
    },
    { relativePassthrough: true },
  );
  return { hooks, pending, published };
}

/** Let the fetch promise, `.json()`, the handler chain and `.finally()` all settle. */
async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test("the notice prefers a failure over a retry in flight", () => {
  assert.equal(
    notice.noticeState({ error: null, loading: false, hasData: false }),
    "none",
  );
  assert.equal(
    notice.noticeState({ error: null, loading: true, hasData: false }),
    "loading",
  );
  assert.equal(
    notice.noticeState({
      error: "Backend down",
      loading: false,
      hasData: false,
    }),
    "error",
  );
  // Pressing "Try again" starts a probe; the message and its button stay put while it runs.
  assert.equal(
    notice.noticeState({
      error: "Backend down",
      loading: true,
      hasData: false,
    }),
    "error",
  );
  // A list already on screen keeps its own stale-entry line instead of the notice.
  assert.equal(
    notice.noticeState({
      error: "Backend down",
      loading: false,
      hasData: true,
    }),
    "none",
  );
  assert.equal(
    notice.noticeState({ error: null, loading: true, hasData: true }),
    "none",
  );
});

test("the notice says what it is doing, and offers a way back", () => {
  const retries: number[] = [];
  const onRetry = () => retries.push(1);

  assert.equal(
    notice.ComputeTargetsNotice({ error: null, loading: false, onRetry }),
    null,
  );

  const loading = notice.ComputeTargetsNotice({
    error: null,
    loading: true,
    onRetry,
  });
  assert.ok(loading);
  assert.equal(loading.props.role, "status");

  const failed = notice.ComputeTargetsNotice({
    error: "Backend down",
    loading: false,
    onRetry,
  });
  assert.ok(failed);
  assert.equal(failed.type, "div");
  assert.equal(failed.props.role, "alert");
  const children = failed.props.children as StubElement[];
  assert.equal(children[0].props.children, "Backend down");
  const retryButton = children.find(
    (child) => typeof child.props.onClick === "function",
  );
  const onClick = retryButton?.props.onClick as (() => void) | undefined;
  onClick?.();
  assert.deepEqual(retries, [1]);
});

test("retry asks the backend again and clears the error only on success", async () => {
  const body = targetsBody();
  const { hooks, pending, published } = targetsHook();
  const first = hooks.useComputeTargets(false);
  first.refresh();
  assert.equal(pending.length, 1);

  pending[0].reject(new TypeError("Failed to fetch"));
  await settled();

  const failed = hooks.useComputeTargets(false);
  assert.equal(failed.data, null);
  assert.equal(failed.loading, false);
  assert.match(failed.error ?? "", UNREACHABLE);

  failed.retry();

  // The failure outranks the retry in flight: the message and its button stay put while the
  // fresh probe runs, and only the answer clears them.
  const retrying = hooks.useComputeTargets(false);
  assert.match(retrying.error ?? "", UNREACHABLE);
  assert.equal(retrying.loading, true);
  assert.equal(pending.length, 2);
  assert.match(published.at(-1)?.error ?? "", UNREACHABLE);
  assert.equal(published.at(-1)?.loading, true);

  pending[1].resolve(Response.json(body));
  await settled();

  const recovered = hooks.useComputeTargets(false);
  assert.equal(recovered.error, null);
  assert.equal(recovered.loading, false);
  assert.deepEqual(recovered.data, body);
});

test("a retry takes its own answer, not the probe it bypassed", async () => {
  const body = targetsBody();
  const { hooks, pending } = targetsHook();
  const first = hooks.useComputeTargets(false);
  first.refresh();
  assert.equal(pending.length, 1);

  const second = hooks.useComputeTargets(false);
  assert.equal(second.loading, true);
  second.retry();
  assert.equal(pending.length, 2);

  pending[1].resolve(Response.json(body));
  await settled();

  const recovered = hooks.useComputeTargets(false);
  assert.deepEqual(recovered.data, body);
  assert.equal(recovered.loading, false);

  // The older probe fails only after the retry replaced it: its answer must not repaint.
  pending[0].reject(new TypeError("Failed to fetch"));
  await settled();

  const after = hooks.useComputeTargets(false);
  assert.deepEqual(after.data, body);
  assert.equal(after.error, null);
  assert.equal(after.loading, false);
});

// ---------------------------------------------------------------------------
// The review's three device-card states: a first paint before any probe, a retry
// that must not flash back to "checking", and a pick that must not keep the
// previous device's numbers.

test("a hook with no probe yet reports loading, not a blank card", () => {
  const { hooks } = targetsHook();
  // The first render lands before the effect can start a probe, and an inactive hook never
  // starts one at all: both must still paint the notice's "checking" line instead of nothing.
  const before = hooks.useComputeTargets(true);
  assert.equal(before.data, null);
  assert.equal(before.error, null);
  assert.equal(before.loading, true);
  assert.equal(
    notice.noticeState({
      error: before.error,
      loading: before.loading,
      hasData: before.data !== null,
    }),
    "loading",
  );

  const inactive = hooks.useComputeTargets(false);
  assert.equal(inactive.data, null);
  assert.equal(inactive.error, null);
  assert.equal(inactive.loading, true);
  assert.equal(
    notice.noticeState({
      error: inactive.error,
      loading: inactive.loading,
      hasData: inactive.data !== null,
    }),
    "loading",
  );
});

test("a retry keeps the failure on screen until its probe settles", async () => {
  const { hooks, pending, published } = targetsHook();
  const first = hooks.useComputeTargets(false);
  first.refresh();
  pending[0].reject(new TypeError("Failed to fetch"));
  await settled();

  hooks.useComputeTargets(false).retry();

  // Mid-flight the failure still outranks the retry: the notice keeps its message and its
  // button instead of flashing back to "checking".
  const retrying = hooks.useComputeTargets(false);
  assert.match(retrying.error ?? "", UNREACHABLE);
  assert.equal(retrying.loading, true);
  assert.equal(
    notice.noticeState({
      error: retrying.error,
      loading: retrying.loading,
      hasData: retrying.data !== null,
    }),
    "error",
  );
  assert.match(published.at(-1)?.error ?? "", UNREACHABLE);

  // A probe that fails again replaces the message; the surface never falls back to "checking".
  pending[1].resolve(Response.json({ detail: "Disk full" }, { status: 500 }));
  await settled();
  const failed = hooks.useComputeTargets(false);
  assert.equal(failed.error, "Disk full");
  assert.equal(failed.loading, false);
});

test("choosing a target re-reads the payload so budget and npu follow it", async () => {
  const gpuBody: ComputeTargetsResponse = {
    ...targetsBody(),
    targets: [
      CPU_TARGET,
      target({ id: "dgpu", kind: "dgpu", name: "Radeon RX 7900 XTX" }),
    ],
    resolved: "dgpu",
    budget: {
      integrated: false,
      dedicated_bytes: 25769803776,
      system_memory_bytes: 68719476736,
      shared_fraction: 0.6,
      usable_bytes: 41231686041,
      notes: "This GPU shares system memory.",
    },
    npu: npu("assist"),
  };
  const cpuBody: ComputeTargetsResponse = {
    ...targetsBody(),
    resolved: "cpu",
    requested: "cpu",
    npu: { ...npu("unavailable"), notes: "The NPU cannot assist a CPU run." },
  };
  const { hooks, pending } = targetsHook();
  const first = hooks.useComputeTargets(false);
  first.refresh();
  pending[0].resolve(Response.json(gpuBody));
  await settled();
  assert.equal(
    hooks.useComputeTargets(false).data?.budget.notes,
    gpuBody.budget.notes,
  );

  const pick = hooks.useComputeTargets(false).choose("cpu");
  await settled();
  assert.equal(pending.length, 2); // the POST
  pending[1].resolve(
    Response.json({ target: CPU_TARGET, resolved: "cpu", requested: "cpu" }),
  );
  await settled();
  // The POST reply carries no budget or NPU capability, so the hook re-reads the whole payload.
  assert.equal(pending.length, 3);
  pending[2].resolve(Response.json(cpuBody));
  assert.equal(await pick, true);

  const after = hooks.useComputeTargets(false);
  assert.equal(after.data?.resolved, "cpu");
  // The card must not keep printing the GPU's shared-memory sentence under a CPU pick.
  assert.equal(after.data?.budget.notes, cpuBody.budget.notes);
  assert.notEqual(after.data?.budget.notes, gpuBody.budget.notes);
  assert.deepEqual(after.data?.npu, cpuBody.npu);
  assert.deepEqual(after.data, cpuBody);
  assert.equal(after.error, null);
  assert.equal(after.loading, false);
});
