// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/**
 * What the NPU can honestly do for a training run.
 *
 * The NPU has no backward pass, so it never trains the model: it only takes
 * work off the GPU's critical path (tokenizing, scoring checkpoints,
 * calibrating QAT weights). The backend decides that, and this module renders
 * the answer without ever promising more than the device can deliver.
 */

/** Jobs the NPU may take over while a training run is going. */
export type NpuAssistTask = "tokenize" | "eval" | "qat_calibrate";

/** What the user asked the NPU to do. */
export type NpuAssistRequested = "auto" | "off" | "assist" | "full";

/** The modes an answer can come back in ("unavailable" is a refusal). */
export type NpuAssistMode = NpuAssistRequested | "unavailable";

/** Every reason the backend may refuse to use the NPU for training. */
export type NpuAssistRefusal =
  | "disabled"
  | "absent"
  | "no_runtime"
  | "no_supported_task"
  | "no_backward_pass"
  | "npu_is_primary"
  | "not_measured";

export interface NpuAssistCapability {
  mode: NpuAssistMode;
  backward_pass: boolean;
  tasks: string[];
  reasons: string[];
}

export interface NpuTrainingAssist {
  enabled: boolean;
  mode: NpuAssistMode;
  tasks: NpuAssistTask[];
  providers: string[];
  refused_reason: NpuAssistRefusal | string | null;
  notes: string[];
  /** Absent from servers that predate the capability block. */
  capability?: NpuAssistCapability | null;
}

export interface NpuTrainingAssistOptions {
  evalEnabled?: boolean;
  qat?: boolean;
  requested?: NpuAssistRequested;
  /**
   * The device a run would use, or "auto" when none was resolved yet. The backend answers for the
   * machine's primary device and names it in `notes[0]`, so a switch from one device to another
   * changes the sentence -- but the device is server state, not an argument: it is part of the
   * cache key and never of the query string.
   */
  primaryTargetId?: string;
  signal?: AbortSignal;
}

const ASSIST_PATH = "/api/compute/npu/assist";

/**
 * The query string for one request, defaults omitted. A cached GET is only
 * reusable if the same intent produces the exact same URL, so `requested=auto`,
 * `eval_enabled=false` and `qat=false` never appear.
 */
export function npuAssistQuery(options: NpuTrainingAssistOptions = {}): string {
  const { evalEnabled = false, qat = false, requested = "auto" } = options;
  const params = new URLSearchParams();
  if (requested !== "auto") params.set("requested", requested);
  if (evalEnabled) params.set("eval_enabled", "true");
  if (qat) params.set("qat", "true");
  const query = params.toString();
  return query ? `?${query}` : "";
}

/**
 * A stable key for one request, for caching and for in-flight deduping.
 *
 * The primary device belongs in here because it is an input to the answer: the backend's sentence
 * describes the device the run would use, so the same `requested` on a different device is a
 * different answer, not the same one. It is not in {@link npuAssistQuery}, because the backend reads
 * the device from its own state and a parameter would only be able to disagree with it.
 * Like the query string, the default is left off, so a session that never resolved a device keeps
 * the key it always had and no real device id can collide with it (`resolved` is never "auto").
 */
export function npuAssistKey(
  options: Pick<
    NpuTrainingAssistOptions,
    "evalEnabled" | "qat" | "requested" | "primaryTargetId"
  > = {},
): string {
  const {
    evalEnabled = false,
    qat = false,
    requested = "auto",
    primaryTargetId = "auto",
  } = options;
  const base = `${requested}|${evalEnabled}|${qat}`;
  return primaryTargetId === "auto" ? base : `${base}|${primaryTargetId}`;
}

export async function getNpuTrainingAssist(
  options: NpuTrainingAssistOptions = {},
): Promise<NpuTrainingAssist> {
  const response = await authFetch(`${ASSIST_PATH}${npuAssistQuery(options)}`, {
    signal: options.signal,
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not read NPU training assist"),
    );
  }
  return (await response.json()) as NpuTrainingAssist;
}

const REFUSAL_SENTENCES = new Map<string, string>([
  ["disabled", "NPU acceleration for training is turned off."],
  ["absent", "No NPU was found on this machine."],
  ["no_runtime", "The NPU runtime is not installed yet."],
  ["no_supported_task", "The NPU does not support any of the training tasks."],
  [
    "no_backward_pass",
    "The NPU has no backward pass, so it cannot train the model on its own.",
  ],
  [
    "npu_is_primary",
    "The NPU is your main compute device, so training runs on it directly instead.",
  ],
  ["not_measured", "This NPU's speed has not been measured yet."],
  // Published by the backend for a generation no runtime here supports; without an entry it fell
  // through to the generic sentence.
  ["unsupported_family", "This NPU generation is not supported yet."],
]);

const UNKNOWN_REFUSAL = "The NPU cannot help with this run.";

/** A plain sentence for a refusal token, never the token itself. */
export function describeRefusal(
  reason: NpuAssistRefusal | string | null | undefined,
): string {
  const token = reason?.trim().toLowerCase();
  if (!token) return UNKNOWN_REFUSAL;
  return REFUSAL_SENTENCES.get(token) ?? UNKNOWN_REFUSAL;
}

const TASK_LABELS = new Map<string, string>([
  ["tokenize", "Reading and tokenizing the dataset"],
  ["eval", "Scoring checkpoints while training runs"],
  ["qat_calibrate", "Calibrating quantized weights"],
]);

/**
 * A plain description of one job the NPU can take over. Task ids a newer
 * backend adds fall back to a neutral line rather than leaking the id.
 */
export function describeTask(task: NpuAssistTask | string): string {
  return TASK_LABELS.get(task) ?? "Helping with this run";
}

const PROVIDER_LABELS = new Map<string, string>([
  ["openvino", "OpenVINO"],
  ["onnxruntime_qnn", "ONNX Runtime QNN"],
  ["vitis_ai", "Vitis AI"],
  ["fastflowlm", "FastFlowLM"],
  ["lemonade", "Lemonade"],
  ["windows_ml", "Windows ML"],
]);

/** A runtime's product name, or the raw id when this build does not know it. */
export function describeProvider(provider: string): string {
  return PROVIDER_LABELS.get(provider.trim().toLowerCase()) ?? provider;
}

function joinWords(parts: string[]): string {
  if (parts.length <= 1) return parts[0] ?? "";
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

function lowerFirst(text: string): string {
  return text.length === 0
    ? text
    : `${text.charAt(0).toLowerCase()}${text.slice(1)}`;
}

/** One sentence: what the NPU takes over, or why it takes nothing. */
export function assistHeadline(assist: NpuTrainingAssist): string {
  // A deliberate "off" is not a refusal: the backend sends `refused_reason: null` for it, so the
  // generic "cannot help" sentence used to appear directly above the backend's own "assist is off"
  // note, which reads as a fault rather than a choice.
  if (!assist.enabled && assist.mode === "off" && !assist.refused_reason) {
    return "NPU assist is off for this run.";
  }
  if (!assist.enabled) return describeRefusal(assist.refused_reason);
  const tasks = (assist.tasks ?? []).map((task) =>
    lowerFirst(describeTask(task)),
  );
  if (tasks.length === 0) return "Your NPU will help with this run.";
  return `Your NPU will handle: ${joinWords(tasks)}.`;
}

/** The head of the panel: headline, the backend's notes, then who serves it. */
export function assistSummaryLines(assist: NpuTrainingAssist): string[] {
  const lines = [assistHeadline(assist)];
  for (const note of assist.notes ?? []) {
    if (note.trim()) lines.push(note);
  }
  const providers = (assist.providers ?? []).filter((name) => name.trim());
  if (assist.enabled && providers.length > 0) {
    lines.push(`Served by: ${joinWords(providers.map(describeProvider))}.`);
  }
  return lines;
}

/**
 * A refusal only reads as a fact rather than a bug if the user learns that the
 * device's missing backward pass is a hardware property. This says so once.
 */
export function assistRefusalNote(
  assist: NpuTrainingAssist | null,
): string | null {
  if (!assist || assist.enabled) return null;
  if (assist.refused_reason !== "no_backward_pass") return null;
  return "NPU-only training becomes possible when a vendor ships a backward pass.";
}
