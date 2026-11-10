// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import type { ComputeTargetRecommendation } from "@/features/compute-targets/types";
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { GradientCheckpointing, TrainingMethod } from "@/types/training";

export interface TrainingFitReport {
  model: string;
  method: string;
  params: number;
  params_label: string;
  trainable_params: number;
  config: {
    hidden_size: number;
    num_layers: number;
    vocab_size: number;
    intermediate_size: number;
    batch_size: number;
    gradient_accumulation: number;
    max_seq_length: number;
    lora_rank: number;
    load_in_4bit: boolean;
    gradient_checkpointing: boolean;
  };
  budget_gb: number | null;
  /**
   * Where the ceiling came from: "explicit", "probe", "target:cpu|igpu|dgpu|npu", "unresolved" or
   * "unreported". The UI says "this card shares system memory" instead of presenting a shared
   * budget as if it were private VRAM.
   */
  budget_source?: string;
  /** Plain sentences the budget module wrote, e.g. an integrated GPU sharing system memory. */
  target_notes?: string[];
  required_gb: number;
  headroom_gb: number | null;
  verdict: "fits" | "tight" | "over" | "unknown";
  breakdown: {
    weights_gb: number;
    adapter_gb: number;
    optimizer_gb: number;
    activations_gb: number;
    logits_gb: number;
    overhead_gb: number;
    total_gb: number;
  };
  suggestions: string[];
  assumptions: string[];
}

/**
 * The device's own sizing, pure. `GET /api/compute/targets` recommends a batch,
 * accumulation, precision and checkpointing per target; these helpers compare that
 * against the wizard's fields and fold it into one fit request, without touching the
 * wizard's values. No fetch and no store, so a node test pins them directly.
 */

/** The wizard fields a device recommendation can re-size. */
export interface FitRecommendationInput {
  batchSize: number;
  gradientAccumulation: number;
  /**
   * The precision these fields run at ("bfloat16", "float16", ...). Optional and
   * never sent: the fit request carries no precision, so a caller that does not
   * track one omits it and no dtype entry can differ.
   */
  dtype?: string;
  /** The wizard's checkpointing setting; "none" is off, exactly as the request sends it. */
  gradientCheckpointing?: GradientCheckpointing;
}

/** A compute target's recommendation, read defensively: an older backend omits it. */
export interface RecommendationSource {
  recommended?: ComputeTargetRecommendation | null;
}

export interface FitRecommendationOptions {
  /**
   * Fold the device's sizing into the request fields. Off by default: a caller that
   * has not asked for the device's sizing sends exactly the wizard's fields.
   */
  apply?: boolean;
}

export interface RecommendationDiffEntry {
  field:
    | "batchSize"
    | "gradientAccumulation"
    | "dtype"
    | "gradientCheckpointing";
  current: string | number | boolean;
  recommended: string | number | boolean;
}

/**
 * The fit request sized by `target`'s recommendation. With `apply` off (the default),
 * or with no recommendation to read, the wizard's fields pass through untouched.
 */
export function withTargetRecommendation<T extends FitRecommendationInput>(
  input: T,
  target: RecommendationSource | null | undefined,
  options?: FitRecommendationOptions,
): T & {
  batchSize: number;
  gradientAccumulation: number;
  dtypeApplied: boolean;
} {
  const recommendation = target?.recommended ?? null;
  if (!options?.apply || !recommendation) {
    return {
      ...input,
      batchSize: input.batchSize,
      gradientAccumulation: input.gradientAccumulation,
      dtypeApplied: false,
    };
  }
  return {
    ...input,
    batchSize: recommendation.batch_size,
    gradientAccumulation: recommendation.grad_accum,
    // True only when the recommendation's precision is not the input's: there is
    // then something to apply. The fit request carries no precision, so the flag
    // rides along for a caller that tracks one.
    dtypeApplied:
      input.dtype !== undefined && input.dtype !== recommendation.dtype,
  };
}

/**
 * Where the wizard's fields disagree with `target`'s recommendation: one entry per
 * differing field, in the recommendation's order. A field the input does not carry
 * (dtype, sometimes checkpointing) cannot differ and is left out; a target without
 * a recommendation leaves everything out. Checkpointing compares on/off, so
 * "unsloth" and "true" both count as on.
 */
export function recommendationDiff(
  input: FitRecommendationInput,
  target: RecommendationSource | null | undefined,
): RecommendationDiffEntry[] {
  const recommendation = target?.recommended ?? null;
  if (!recommendation) return [];
  const diff: RecommendationDiffEntry[] = [];
  if (input.batchSize !== recommendation.batch_size) {
    diff.push({
      field: "batchSize",
      current: input.batchSize,
      recommended: recommendation.batch_size,
    });
  }
  if (input.gradientAccumulation !== recommendation.grad_accum) {
    diff.push({
      field: "gradientAccumulation",
      current: input.gradientAccumulation,
      recommended: recommendation.grad_accum,
    });
  }
  if (input.dtype !== undefined && input.dtype !== recommendation.dtype) {
    diff.push({
      field: "dtype",
      current: input.dtype,
      recommended: recommendation.dtype,
    });
  }
  if (
    input.gradientCheckpointing !== undefined &&
    (input.gradientCheckpointing !== "none") !==
      recommendation.gradient_checkpointing
  ) {
    diff.push({
      field: "gradientCheckpointing",
      current: input.gradientCheckpointing,
      recommended: recommendation.gradient_checkpointing,
    });
  }
  return diff;
}

function fitMethod(method: TrainingMethod): string {
  if (method === "full" || method === "cpt") return method;
  return method === "qlora" ? "qlora" : "lora";
}

export async function fetchTrainingFit(input: {
  model: string;
  method: TrainingMethod;
  loadIn4Bit: boolean;
  maxSeqLength: number;
  batchSize: number;
  gradientAccumulation: number;
  loraRank: number;
  gradientCheckpointing: GradientCheckpointing;
  gpuMemoryGb?: number | null;
  /** The compute target the estimate is for; omitted, the server uses the stored choice. */
  computeTarget?: string | null;
}): Promise<TrainingFitReport> {
  const response = await authFetch("/api/train/fit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: input.model,
      method: fitMethod(input.method),
      load_in_4bit: input.loadIn4Bit,
      max_seq_length: input.maxSeqLength,
      batch_size: input.batchSize,
      gradient_accumulation: input.gradientAccumulation,
      lora_rank: input.loraRank,
      gradient_checkpointing: input.gradientCheckpointing !== "none",
      ...(input.computeTarget ? { compute_target: input.computeTarget } : {}),
      gpu_memory_gb: input.gpuMemoryGb ?? null,
    }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as TrainingFitReport;
}
