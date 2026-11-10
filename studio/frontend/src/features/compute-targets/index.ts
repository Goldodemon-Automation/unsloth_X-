// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export {
  ComputeTargetApiError,
  NPU_MODE_UNAVAILABLE,
  computeTargetError,
  computeTargetKindLabel,
  describeReason,
  describeReasons,
  formatMemoryBytes,
  isNpuTarget,
  isTargetSelectable,
} from "./types";
export type {
  ComputeBudget,
  ComputeTarget,
  ComputeTargetCapabilities,
  ComputeTargetError,
  ComputeTargetId,
  ComputeTargetRecommendation,
  ComputeTargetsResponse,
  NpuProvider,
  NpuTrainingCapability,
} from "./types";
export { getComputeTargets, setComputeTarget } from "./api";
export type { SetComputeTargetResult } from "./api";
export { useComputeTargets } from "./use-compute-targets";
export type {
  ComputeTargetsSnapshot,
  UseComputeTargetsResult,
} from "./use-compute-targets";
export { useComputeTargetStore } from "./compute-target-store";
export type { ComputeTargetSelectionState } from "./compute-target-store";
export { ComputeTargetPicker } from "./compute-target-picker";
export type { ComputeTargetPickerProps } from "./compute-target-picker";
export {
  ComputeTargetsNotice,
  noticeState,
} from "./compute-targets-notice";
export type {
  ComputeTargetsNoticeInput,
  ComputeTargetsNoticeProps,
  ComputeTargetsNoticeState,
} from "./compute-targets-notice";
export { ComputeCapabilityCard } from "./compute-capability-card";
export type { ComputeCapabilityCardProps } from "./compute-capability-card";
