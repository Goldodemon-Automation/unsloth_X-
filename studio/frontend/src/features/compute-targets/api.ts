// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type ComputeTarget,
  type ComputeTargetId,
  type ComputeTargetsResponse,
  computeTargetHttpError,
  computeTargetNetworkError,
} from "./types";

const TARGETS_PATH = "/api/compute/targets";
const TARGET_PATH = "/api/compute/target";

export interface SetComputeTargetResult {
  target: ComputeTarget;
  resolved: ComputeTargetId;
  /** What was asked for ("auto" or a target id), which is what the run should keep meaning. */
  requested?: ComputeTargetId | "auto";
}

/** What this machine can train on, how much memory each device has and what it supports. */
export async function getComputeTargets(): Promise<ComputeTargetsResponse> {
  const response = await request(
    TARGETS_PATH,
    "Could not read the compute targets",
  );
  return (await response.json()) as ComputeTargetsResponse;
}

/** Ask the backend to train on `target`; the reply carries that target's details. */
export async function setComputeTarget(
  target: string,
): Promise<SetComputeTargetResult> {
  const response = await request(
    TARGET_PATH,
    "Could not switch the compute target",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    },
  );
  return (await response.json()) as SetComputeTargetResult;
}

/**
 * The one place a compute-target call turns a transport or HTTP failure into a
 * message a UI can show: never a bare status, never a swallowed error.
 */
async function request(
  path: string,
  fallbackPrefix: string,
  init?: RequestInit,
): Promise<Response> {
  let response: Response;
  try {
    response = await authFetch(path, init);
  } catch {
    throw computeTargetNetworkError();
  }
  if (!response.ok) {
    throw computeTargetHttpError(
      response.status,
      await readFastApiError(response, fallbackPrefix),
    );
  }
  return response;
}
