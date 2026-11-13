// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export const LLAMA_ARCHITECTURES = [
  "normal",
  "dory",
  "bee",
  "neurall",
  "ik",
  "x",
] as const;

export type LlamaArchitectureId = (typeof LLAMA_ARCHITECTURES)[number];

export function isLlamaArchitectureId(
  value: unknown,
): value is LlamaArchitectureId {
  return (
    typeof value === "string" &&
    (LLAMA_ARCHITECTURES as readonly string[]).includes(value)
  );
}

export interface LlamaArchitectureProfile {
  id: LlamaArchitectureId;
  label: string;
  repo: string | null;
  sourceBuild: boolean;
  sourceCommit: string | null;
  available: boolean;
  installable: boolean;
  selectable: boolean;
  installed: boolean;
  active: boolean;
  reason: string | null;
  asset: string | null;
  releaseTag: string | null;
  latestTag: string | null;
  installedTag: string | null;
  installedAt: string | null;
  downloadSizeBytes: number | null;
  updateAvailable: boolean;
  binaryPath: string | null;
  installDir: string | null;
  description: string;
}

export interface LlamaArchitectureJob {
  state: "idle" | "running" | "success" | "error";
  architecture: LlamaArchitectureId | null;
  message: string;
  error: string | null;
  progress: number | null;
  fromTag: string | null;
  toTag: string | null;
  activated: boolean | null;
  startedAt: string | null;
  finishedAt: string | null;
}

export interface LlamaArchitecturesStatus {
  active: LlamaArchitectureId;
  runtimeLocked: boolean;
  runtimeLockReason: string | null;
  profiles: LlamaArchitectureProfile[];
  job: LlamaArchitectureJob;
}

export interface LlamaArchitectureInstallStarted {
  started: boolean;
  reason: string | null;
  message: string | null;
  job: LlamaArchitectureJob;
}

type ApiObject = Record<string, unknown>;

function asString(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseJob(value: unknown): LlamaArchitectureJob {
  const job = (value ?? {}) as ApiObject;
  const state = job.state;
  return {
    state:
      state === "running" || state === "success" || state === "error"
        ? state
        : "idle",
    architecture: isLlamaArchitectureId(job.architecture)
      ? job.architecture
      : null,
    message: typeof job.message === "string" ? job.message : "",
    error: asString(job.error),
    progress: asNumber(job.progress),
    fromTag: asString(job.from_tag),
    toTag: asString(job.to_tag),
    activated: typeof job.activated === "boolean" ? job.activated : null,
    startedAt: asString(job.started_at),
    finishedAt: asString(job.finished_at),
  };
}

function parseProfile(value: unknown): LlamaArchitectureProfile | null {
  const profile = (value ?? {}) as ApiObject;
  if (!isLlamaArchitectureId(profile.id)) {
    return null;
  }
  return {
    id: profile.id,
    label: typeof profile.label === "string" ? profile.label : profile.id,
    repo: asString(profile.repo),
    sourceBuild: profile.source_build === true,
    sourceCommit: asString(profile.source_commit),
    available: profile.available === true,
    installable: profile.installable === true,
    selectable: profile.selectable === true,
    installed: profile.installed === true,
    active: profile.active === true,
    reason: asString(profile.reason),
    asset: asString(profile.asset),
    releaseTag: asString(profile.release_tag),
    latestTag: asString(profile.latest_tag),
    installedTag: asString(profile.installed_tag),
    installedAt: asString(profile.installed_at_utc),
    downloadSizeBytes: asNumber(profile.download_size_bytes),
    updateAvailable: profile.update_available === true,
    binaryPath: asString(profile.binary_path),
    installDir: asString(profile.install_dir),
    description:
      typeof profile.description === "string" ? profile.description : "",
  };
}

export function parseLlamaArchitecturesStatus(
  value: unknown,
): LlamaArchitecturesStatus {
  const payload = (value ?? {}) as ApiObject;
  const profiles = Array.isArray(payload.profiles) ? payload.profiles : [];
  return {
    active: isLlamaArchitectureId(payload.active) ? payload.active : "normal",
    runtimeLocked: payload.runtime_locked === true,
    runtimeLockReason: asString(payload.runtime_lock_reason),
    profiles: profiles
      .map(parseProfile)
      .filter(
        (profile): profile is LlamaArchitectureProfile => profile !== null,
      ),
    job: parseJob(payload.job),
  };
}

export async function loadLlamaArchitectures(
  forceRefresh = false,
): Promise<LlamaArchitecturesStatus> {
  const response = await authFetch(
    `/api/llama/architectures${forceRefresh ? "?force_refresh=true" : ""}`,
  );
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to load llama.cpp architectures",
      ),
    );
  }
  return parseLlamaArchitecturesStatus(await response.json());
}

export async function installLlamaArchitecture(
  architecture: LlamaArchitectureId,
  activate = true,
): Promise<LlamaArchitectureInstallStarted> {
  const response = await authFetch("/api/llama/architectures/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ architecture, activate }),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to install the llama.cpp architecture",
      ),
    );
  }
  const payload = (await response.json()) as ApiObject;
  return {
    started: payload.started === true,
    reason: asString(payload.reason),
    message: asString(payload.message),
    job: parseJob(payload.job),
  };
}

export async function activateLlamaArchitecture(
  architecture: LlamaArchitectureId,
): Promise<LlamaArchitecturesStatus> {
  const response = await authFetch("/api/llama/architectures/activate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ architecture }),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to switch the llama.cpp architecture",
      ),
    );
  }
  return parseLlamaArchitecturesStatus(await response.json());
}
