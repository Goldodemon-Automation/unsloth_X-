// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The compute-target contract: the devices a machine can train on, what each one
 * can do, and the pure helpers the picker and the capability card render from.
 *
 * The presentation helpers live here, not in the components, so they stay pure:
 * a node test imports this module directly, with no React and no network layer.
 * Nothing in this file may import another module for that reason.
 */

export type ComputeTargetId = "cpu" | "igpu" | "dgpu" | "npu";

/** What the backend's kernels may use on one device. */
export interface ComputeTargetCapabilities {
  bf16: boolean;
  fp16: boolean;
  bnb4bit: boolean;
  triton: boolean;
  grad_checkpoint: boolean;
  compile: boolean;
  flash_attn: boolean;
}

/** The settings the backend would start a run with on this device. */
export interface ComputeTargetRecommendation {
  batch_size: number;
  grad_accum: number;
  dtype: string;
  gradient_checkpointing: boolean;
}

export interface ComputeTarget {
  id: ComputeTargetId;
  /** The torch device family a run is placed on ("cpu", "gpu", "npu"). */
  device: string;
  /** The device's shape: "cpu", "igpu", "dgpu" or "npu". Text, since the backend may add one. */
  kind: string;
  name: string;
  vendor: string;
  /** Null when the backend could not read a size for this device. */
  memory_bytes: number | null;
  unified_memory: boolean;
  supported: boolean;
  /** Machine-readable tokens: describeReason turns them into prose. */
  reasons: string[];
  capabilities: ComputeTargetCapabilities;
  recommended: ComputeTargetRecommendation;
  notes: string[];
}

export interface NpuProvider {
  name: string;
  installed: boolean;
  version: string | null;
}

/**
 * What the NPU may do for a run. The NPU has no backward pass, so it assists
 * rather than trains; the backend may add modes, so `mode` stays text.
 */
export interface NpuTrainingCapability {
  mode: string;
  backward_pass: boolean;
  tasks: string[];
  reasons: string[];
  notes: string;
  providers: NpuProvider[];
}

/** An NPU in this mode is reported, but never selectable. */
export const NPU_MODE_UNAVAILABLE = "unavailable";

/** How much memory a run may use, and where that memory comes from. */
export interface ComputeBudget {
  integrated: boolean;
  dedicated_bytes: number | null;
  system_memory_bytes: number | null;
  shared_fraction: number | null;
  usable_bytes: number | null;
  notes: string;
}

/** `GET /api/compute/targets`. */
export interface ComputeTargetsResponse {
  targets: ComputeTarget[];
  default: ComputeTargetId;
  resolved: ComputeTargetId;
  /**
   * What was asked for: "auto" or a target id. Kept separate from `resolved` so a run started from
   * this page keeps meaning "auto" -- a laptop that later gains an eGPU must not stay pinned to the
   * CPU it resolved to once.
   */
  requested?: ComputeTargetId | "auto";
  /** Set when a remembered target could not be served on this machine and the default was used. */
  fallback_reason?: string;
  npu: NpuTrainingCapability;
  budget: ComputeBudget;
}

export interface ComputeTargetError {
  message: string;
  code: string;
}

export const COMPUTE_TARGET_ERROR_REQUEST_FAILED = "request_failed";
export const COMPUTE_TARGET_ERROR_NETWORK_UNREACHABLE = "network_unreachable";

const COMPUTE_TARGET_FALLBACK_MESSAGE =
  "The compute targets could not be read from the backend.";

/** A failed call, carrying a message a UI can show and a stable code. */
export class ComputeTargetApiError extends Error {
  code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = "ComputeTargetApiError";
    this.code = code;
  }
}

/** The route bodies carry only `detail`, so the code names the HTTP outcome. */
export function computeTargetHttpError(
  status: number,
  detail: string,
): ComputeTargetApiError {
  return new ComputeTargetApiError(detail, `http_${status}`);
}

export function computeTargetNetworkError(): ComputeTargetApiError {
  return new ComputeTargetApiError(
    "Could not reach Unsloth Studio. Check that the backend is running, then try again.",
    COMPUTE_TARGET_ERROR_NETWORK_UNREACHABLE,
  );
}

/** Reduce anything a compute-target call throws to a message and a code. */
export function computeTargetError(error: unknown): ComputeTargetError {
  if (typeof error === "string" && error.trim()) {
    return { message: error, code: COMPUTE_TARGET_ERROR_REQUEST_FAILED };
  }
  if (error && typeof error === "object") {
    const { message, code } = error as { message?: unknown; code?: unknown };
    return {
      message:
        typeof message === "string" && message.trim()
          ? message
          : COMPUTE_TARGET_FALLBACK_MESSAGE,
      code:
        typeof code === "string" && code
          ? code
          : COMPUTE_TARGET_ERROR_REQUEST_FAILED,
    };
  }
  return {
    message: COMPUTE_TARGET_FALLBACK_MESSAGE,
    code: COMPUTE_TARGET_ERROR_REQUEST_FAILED,
  };
}

/**
 * The backend's reason tokens, in plain words. An unknown token falls back to
 * itself so a newer backend still reads as something rather than vanishing.
 */
const REASON_SENTENCES: Record<string, string | undefined> = {
  arch_unknown:
    "Unsloth Studio could not tell which instruction set this processor supports, so it cannot pick a safe kernel for it.",
  no_backward_pass:
    "This device has no backward pass, so it cannot train a model.",
  disabled:
    "This device is switched off in the operating system or in Unsloth Studio's settings.",
  absent: "No device of this kind was found on this machine.",
  unsupported_family: "This device's family is not supported yet.",
  no_triton:
    "The Triton kernel compiler is not installed, so this GPU cannot run Unsloth's training kernels.",
  no_bnb:
    "The 4-bit bitsandbytes library is not installed, so this GPU cannot train in 4-bit.",
  npu_is_primary:
    "This machine's NPU is its main accelerator, and training needs a device with a backward pass.",
  no_runtime: "The runtime this device needs is not installed.",
  // Emitted by the registry when a GPU is listed but PyTorch is absent from this environment, which
  // is a repairable install rather than a missing device -- the distinction the user needs.
  torch_missing:
    "PyTorch is not installed in this environment, so this GPU cannot be used for training yet.",
};

export function describeReason(reason: string): string {
  return REASON_SENTENCES[reason] ?? reason;
}

/** Every reason a target gives, as sentences. */
export function describeReasons(reasons: readonly string[]): string {
  return reasons.map(describeReason).join(" ");
}

const BYTES_PER_GB = 1024 ** 3;

/** A device's memory, in GB to one decimal; `—` when the backend has no figure. */
export function formatMemoryBytes(bytes: number | null): string {
  if (bytes === null || !Number.isFinite(bytes) || bytes < 0) return "—";
  return `${(bytes / BYTES_PER_GB).toFixed(1)} GB`;
}

const KIND_LABELS: Record<string, string | undefined> = {
  cpu: "CPU",
  igpu: "Integrated GPU",
  dgpu: "GPU",
  npu: "NPU",
};

/** The one-line badge a row shows, e.g. "Integrated GPU". */
export function computeTargetKindLabel(target: ComputeTarget): string {
  return KIND_LABELS[target.kind] ?? target.kind;
}

export function isNpuTarget(target: ComputeTarget): boolean {
  return target.kind === "npu" || target.id === "npu";
}

/**
 * Whether a row may be picked. Unsupported devices stay listed so the user can
 * see the machine has one; an unavailable NPU cannot even be assisted with.
 */
export function isTargetSelectable(
  target: ComputeTarget,
  npu: NpuTrainingCapability | null | undefined,
): boolean {
  if (!target.supported) return false;
  return !(isNpuTarget(target) && npu?.mode === NPU_MODE_UNAVAILABLE);
}
