// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Opt-in telemetry settings. The endpoint is the collector's report URL; the
// install id is random and belongs to this machine, not to an account.

import { authFetch } from "@/features/auth";

const BASE = "/api/telemetry";

export interface TelemetrySettings {
  enabled: boolean;
  endpoint: string;
  has_ingest_key: boolean;
  interval_hours: number;
  install_id: string;
  last_sent_at: number | null;
  last_attempt_at: number | null;
  last_error: string | null;
  dashboard_url: string;
  window_days: number;
}

export interface TelemetryUpdate {
  enabled?: boolean;
  endpoint?: string;
  ingest_key?: string;
  interval_hours?: number;
  reset_install_id?: boolean;
}

export interface TelemetrySendResult {
  ok: boolean;
  error: string | null;
  skipped?: boolean;
  attempted_at?: number | null;
  sent_at?: number | null;
}

export interface TelemetryPreview {
  schema_version: number;
  install_id: string;
  sent_at: number;
  version: string | null;
  window_days: number;
  device: Record<string, unknown>;
  machine: Record<string, unknown>;
  usage: unknown[];
  models: unknown[];
  trainings: unknown[];
  merges: unknown[];
  features: Record<string, number>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(`${BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // A proxy answered with HTML; the status is all there is.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function loadTelemetrySettings(): Promise<TelemetrySettings> {
  return request<TelemetrySettings>("/settings");
}

export function saveTelemetrySettings(
  update: TelemetryUpdate,
): Promise<TelemetrySettings> {
  return request<TelemetrySettings>("/settings", {
    method: "PUT",
    body: JSON.stringify(update),
  });
}

export function previewTelemetry(): Promise<TelemetryPreview> {
  return request<TelemetryPreview>("/preview");
}

export function sendTelemetryNow(): Promise<TelemetrySendResult> {
  return request<TelemetrySendResult>("/send", { method: "POST" });
}
