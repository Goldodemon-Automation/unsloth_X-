// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * View state for the llama.cpp architecture picker, derived from the server
 * status alone. Kept import-free like llama-backend-payload.ts so the tests can
 * pin the row's decisions without loading a component.
 */

import type { InterpolationValues, TranslationKey } from "@/i18n";
import type {
  LlamaArchitectureId,
  LlamaArchitecturesStatus,
} from "../api/llama-architectures";

/** What the status line says, as a key plus its interpolation values. */
export interface ArchitectureStatusText {
  key: TranslationKey;
  values: InterpolationValues;
}

/** The two things the row formats but cannot compute: a byte count and a profile's display name. */
export interface ArchitectureTextFormat {
  size: (bytes: number) => string;
  profile: (id: LlamaArchitectureId) => string;
}

export interface ArchitectureActionState {
  /** A job is running or an action is in flight: every control stays off. */
  running: boolean;
  canInstall: boolean;
  canActivate: boolean;
}

/**
 * The one status line for the selected profile. Order matters: the runtime lock
 * and a running job describe the row before the profile does, and a profile
 * that is already installed is described by its installed tag, never by the
 * release that would replace it.
 */
export function architectureStatusText(
  selected: LlamaArchitectureId,
  status: LlamaArchitecturesStatus,
  format: ArchitectureTextFormat,
): ArchitectureStatusText | null {
  const values = { profile: format.profile(selected) };
  if (status.runtimeLocked) {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.runtimeLocked",
      values: {},
    };
  }
  const profile = status.profiles.find((entry) => entry.id === selected);
  if (!profile) return null;
  if (status.job.state === "running" && status.job.architecture === selected) {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.installing",
      values,
    };
  }
  if (profile.id === "normal") {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.bundled",
      values: {},
    };
  }
  if (profile.installed) {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.installed",
      values: {
        tag: profile.installedTag ?? profile.latestTag ?? "?",
      },
    };
  }
  if (profile.reason === "build_tools_missing") {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.buildToolsMissing",
      values: {},
    };
  }
  if (profile.sourceBuild === true && !profile.asset) {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.sourceBuild",
      values: { commit: profile.sourceCommit?.slice(0, 8) ?? "pinned" },
    };
  }
  if (profile.reason === "build_not_ready") {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.buildNotReady",
      values: {},
    };
  }
  if (profile.reason === "no_release") {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.noRelease",
      values: {},
    };
  }
  if (profile.reason === "missing_binary") {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.missingBinary",
      values: {},
    };
  }
  if (profile.reason === "release_lookup_failed") {
    return {
      key: "settings.resources.llamaBackend.modelSupport.status.lookupFailed",
      values: {},
    };
  }
  if (profile.installable) {
    return profile.downloadSizeBytes != null
      ? {
          key: "settings.resources.llamaBackend.modelSupport.status.download",
          values: { size: format.size(profile.downloadSizeBytes) },
        }
      : {
          key: "settings.resources.llamaBackend.modelSupport.status.notInstalled",
          values: {},
        };
  }
  return {
    key: "settings.resources.llamaBackend.modelSupport.status.noBuild",
    values: {},
  };
}

/**
 * Which actions the row offers. Both actions are gated on the same hard
 * blockers as the picker itself (a runtime lock, or any job in flight), so the
 * buttons can never promise what the server would refuse.
 */
export function architectureActionState(
  selected: LlamaArchitectureId,
  status: LlamaArchitecturesStatus | null,
  busy: boolean,
): ArchitectureActionState {
  const profile =
    status?.profiles.find((entry) => entry.id === selected) ?? null;
  const running = busy || status?.job.state === "running";
  const runtimeLocked = status?.runtimeLocked === true;
  const canInstall =
    profile != null &&
    profile.id !== "normal" &&
    profile.installable &&
    (!profile.installed || profile.updateAvailable) &&
    !running &&
    !runtimeLocked;
  const canActivate =
    profile != null &&
    profile.id !== "normal" &&
    profile.installed &&
    status?.active !== profile.id &&
    !running &&
    !runtimeLocked;
  return { running, canInstall, canActivate };
}
