// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type SelfTrainingSettings = {
  enabled: boolean;
  defaultEnabled: boolean;
};

type ApiSelfTrainingSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_enabled: boolean;
};

function fromApi(settings: ApiSelfTrainingSettings): SelfTrainingSettings {
  return {
    enabled: settings.enabled,
    defaultEnabled: settings.default_enabled,
  };
}

export async function loadSelfTrainingSettings(): Promise<SelfTrainingSettings> {
  const res = await authFetch("/api/settings/self-training");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load model self-training settings"),
    );
  }
  return fromApi(await res.json());
}

export async function updateSelfTrainingSettings(
  enabled: boolean,
): Promise<SelfTrainingSettings> {
  const res = await authFetch("/api/settings/self-training", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update model self-training settings"),
    );
  }
  return fromApi(await res.json());
}
