// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as NpuAssistModule from "../src/features/npu/npu-training-assist.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const ASSIST = {
  enabled: true,
  mode: "assist",
  tasks: ["tokenize", "eval"],
  providers: ["openvino"],
  refused_reason: null,
  notes: [
    "The NPU has no backward pass, so it cannot train the model on its own.",
    "Training runs on the discrete GPU; the NPU's work is off the critical path of a training step.",
  ],
  capability: {
    mode: "assist",
    backward_pass: false,
    tasks: ["tokenize", "eval", "inference"],
    reasons: [],
  },
} as const;

let reply: () => Response = () => Response.json(ASSIST);
const urls: string[] = [];

// The stub below forwards to globalThis.fetch, so this is the transport the
// module under test actually talks to.
globalThis.fetch = (async (input: RequestInfo | URL) => {
  urls.push(String(input));
  return reply();
}) as typeof fetch;

function client() {
  return loadWithStubs<typeof NpuAssistModule>(
    new URL("../src/features/npu/npu-training-assist.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: (path: string, init?: RequestInit) =>
          globalThis.fetch(path, init),
      },
      "@/lib/format-fastapi-error": formatFastApiError,
    },
  );
}

const api = client();

function assist(
  overrides: Partial<NpuAssistModule.NpuTrainingAssist> = {},
): NpuAssistModule.NpuTrainingAssist {
  return { ...ASSIST, ...overrides } as NpuAssistModule.NpuTrainingAssist;
}

const REFUSALS = [
  "disabled",
  "absent",
  "no_runtime",
  "no_supported_task",
  "no_backward_pass",
  "npu_is_primary",
  "not_measured",
] as const;

test("every refusal reads as a sentence, and none leaks the token", () => {
  const sentences = REFUSALS.map((reason) => api.describeRefusal(reason));
  for (const sentence of sentences) {
    assert.match(sentence, /\.$/);
    assert.doesNotMatch(sentence, /_/);
    assert.ok(sentence.length > 10, sentence);
  }
  assert.equal(new Set(sentences).size, REFUSALS.length);
  assert.equal(
    api.describeRefusal("no_backward_pass"),
    "The NPU has no backward pass, so it cannot train the model on its own.",
  );
});

test("an unknown or missing refusal admits it does not know", () => {
  const fallback = "The NPU cannot help with this run.";
  for (const reason of ["who_knows", "", null, undefined]) {
    assert.equal(api.describeRefusal(reason), fallback);
    assert.doesNotMatch(api.describeRefusal(reason), /_/);
  }
});

test("each NPU task is described in plain words", () => {
  assert.equal(
    api.describeTask("tokenize"),
    "Reading and tokenizing the dataset",
  );
  assert.equal(
    api.describeTask("eval"),
    "Scoring checkpoints while training runs",
  );
  assert.equal(
    api.describeTask("qat_calibrate"),
    "Calibrating quantized weights",
  );
});

test("the headline lists the work when enabled and the refusal when not", () => {
  assert.equal(
    api.assistHeadline(assist()),
    "Your NPU will handle: reading and tokenizing the dataset and scoring checkpoints while training runs.",
  );
  assert.equal(
    api.assistHeadline(assist({ tasks: ["qat_calibrate"] })),
    "Your NPU will handle: calibrating quantized weights.",
  );
  for (const reason of REFUSALS) {
    const refused = assist({
      enabled: false,
      mode: "unavailable",
      refused_reason: reason,
      tasks: [],
      providers: [],
    });
    assert.equal(api.assistHeadline(refused), api.describeRefusal(reason));
  }
});

test("the summary keeps its order: headline, notes, then who serves it", () => {
  assert.deepEqual(api.assistSummaryLines(assist()), [
    "Your NPU will handle: reading and tokenizing the dataset and scoring checkpoints while training runs.",
    ...ASSIST.notes,
    "Served by: OpenVINO.",
  ]);
});

test("provider ids become product names, and an unknown one stays as it came", () => {
  assert.equal(api.describeProvider("openvino"), "OpenVINO");
  assert.equal(api.describeProvider("onnxruntime_qnn"), "ONNX Runtime QNN");
  assert.equal(api.describeProvider("vitis_ai"), "Vitis AI");
  assert.equal(api.describeProvider("fastflowlm"), "FastFlowLM");
  assert.equal(api.describeProvider("lemonade"), "Lemonade");
  assert.equal(api.describeProvider("windows_ml"), "Windows ML");
  assert.equal(api.describeProvider("some_new_runtime"), "some_new_runtime");
  const lines = api.assistSummaryLines(
    assist({ providers: ["openvino", "windows_ml"] }),
  );
  assert.equal(lines[lines.length - 1], "Served by: OpenVINO and Windows ML.");
});

test("a refusal lists no server and no extra line", () => {
  const lines = api.assistSummaryLines(
    assist({
      enabled: false,
      mode: "unavailable",
      refused_reason: "no_backward_pass",
      tasks: [],
      providers: ["openvino"],
      notes: [],
    }),
  );
  assert.deepEqual(lines, [
    "The NPU has no backward pass, so it cannot train the model on its own.",
  ]);
});

test("the same intent asks the same URL, and only non-defaults appear", async () => {
  urls.length = 0;
  reply = () => Response.json(ASSIST);
  await api.getNpuTrainingAssist();
  await api.getNpuTrainingAssist();
  assert.equal(urls[0], urls[1]);
  assert.equal(urls[0], "/api/compute/npu/assist");

  await api.getNpuTrainingAssist({ evalEnabled: true });
  assert.equal(urls[2], "/api/compute/npu/assist?eval_enabled=true");

  await api.getNpuTrainingAssist({
    evalEnabled: true,
    qat: true,
    requested: "assist",
  });
  assert.equal(
    urls[3],
    "/api/compute/npu/assist?requested=assist&eval_enabled=true&qat=true",
  );
});

test("the request key names every field that changes the answer", () => {
  assert.equal(api.npuAssistKey(), "auto|false|false");
  assert.deepEqual(
    api.npuAssistKey({ requested: "full", qat: true }),
    "full|false|true",
  );
  assert.equal(
    api.npuAssistKey({ evalEnabled: true }),
    api.npuAssistKey({ evalEnabled: true }),
  );
});

test("an error body that carries a detail rejects with that detail", async () => {
  reply = () => Response.json({ detail: "The NPU is busy." }, { status: 503 });
  await assert.rejects(api.getNpuTrainingAssist(), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.equal(error.message, "The NPU is busy.");
    return true;
  });
});

test("a different device is a different key, and the same device is stable", () => {
  const ids = ["cpu", "igpu", "dgpu"];
  const keys = ids.map((primaryTargetId) =>
    api.npuAssistKey({ primaryTargetId }),
  );
  assert.equal(new Set(keys).size, ids.length);
  // Stable: the same device always keys the same entry, whichever order it is named in.
  for (let index = 0; index < ids.length; index += 1) {
    assert.equal(
      api.npuAssistKey({ primaryTargetId: ids[index] }),
      keys[index],
    );
  }
  // The device composes with the rest of the intent instead of replacing it.
  assert.equal(
    api.npuAssistKey({
      requested: "assist",
      qat: true,
      primaryTargetId: "dgpu",
    }),
    "assist|false|true|dgpu",
  );
  assert.notEqual(
    api.npuAssistKey({
      requested: "assist",
      qat: true,
      primaryTargetId: "dgpu",
    }),
    api.npuAssistKey({
      requested: "assist",
      qat: true,
      primaryTargetId: "igpu",
    }),
  );
  // An unresolved device keeps the key a session had before one was picked.
  assert.equal(
    api.npuAssistKey({ primaryTargetId: "auto" }),
    api.npuAssistKey(),
  );
  assert.notEqual(
    api.npuAssistKey({ primaryTargetId: "dgpu" }),
    api.npuAssistKey(),
  );
});

test("the device never reaches the query string", async () => {
  urls.length = 0;
  reply = () => Response.json(ASSIST);
  await api.getNpuTrainingAssist({ primaryTargetId: "dgpu" });
  await api.getNpuTrainingAssist({
    requested: "assist",
    evalEnabled: true,
    primaryTargetId: "igpu",
  });
  assert.equal(urls[0], "/api/compute/npu/assist");
  assert.equal(
    urls[1],
    "/api/compute/npu/assist?requested=assist&eval_enabled=true",
  );
  // Two devices asking the same intent hit the same URL, and differ only in the key.
  assert.equal(
    api.npuAssistQuery({ primaryTargetId: "dgpu" }),
    api.npuAssistQuery({ primaryTargetId: "cpu" }),
  );
  assert.equal(api.npuAssistQuery({ primaryTargetId: "dgpu" }), "");
});
