// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import {
  type TelemetryPreview,
  type TelemetrySettings,
  loadTelemetrySettings,
  previewTelemetry,
  saveTelemetrySettings,
  sendTelemetryNow,
} from "../api/telemetry";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

const INPUT_CLASS =
  "h-8 min-w-0 rounded-md border border-border bg-background px-2.5 text-xs text-foreground outline-none placeholder:text-muted-foreground/60 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 disabled:opacity-50";

function messageOf(failure: unknown): string {
  return failure instanceof Error ? failure.message : String(failure);
}

function stamp(value: number | null | undefined): string {
  if (!value) return "—";
  try {
    return new Date(value * 1000).toLocaleString();
  } catch {
    return "—";
  }
}

export function TelemetryTab() {
  const t = useT();
  const [settings, setSettings] = useState<TelemetrySettings | null>(null);
  const [endpoint, setEndpoint] = useState("");
  const [ingestKey, setIngestKey] = useState("");
  const [preview, setPreview] = useState<TelemetryPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [sending, setSending] = useState(false);

  const adopt = useCallback((next: TelemetrySettings) => {
    setSettings(next);
    setEndpoint(next.endpoint);
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const next = await loadTelemetrySettings();
        if (!cancelled) adopt(next);
      } catch (failure) {
        if (!cancelled) setError(messageOf(failure));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [adopt]);

  const save = useCallback(
    async (
      update: Parameters<typeof saveTelemetrySettings>[0],
      successMessage?: string,
    ): Promise<boolean> => {
      setSaving(true);
      try {
        adopt(await saveTelemetrySettings(update));
        if (successMessage) toast.success(successMessage);
        return true;
      } catch (failure) {
        toast.error(t("settings.telemetry.saveFailed"), {
          description: messageOf(failure),
        });
        return false;
      } finally {
        setSaving(false);
      }
    },
    [adopt, t],
  );

  const showPreview = useCallback(async () => {
    setPreviewing(true);
    try {
      setPreview(await previewTelemetry());
    } catch (failure) {
      toast.error(t("settings.telemetry.previewFailed"), {
        description: messageOf(failure),
      });
    } finally {
      setPreviewing(false);
    }
  }, [t]);

  const sendNow = useCallback(async () => {
    setSending(true);
    try {
      const result = await sendTelemetryNow();
      if (result.ok) {
        toast.success(t("settings.telemetry.sendOk"));
        adopt(await loadTelemetrySettings());
      } else {
        toast.error(t("settings.telemetry.sendFailed"), {
          description: result.error ?? undefined,
        });
      }
    } catch (failure) {
      toast.error(t("settings.telemetry.sendFailed"), {
        description: messageOf(failure),
      });
    } finally {
      setSending(false);
    }
  }, [adopt, t]);

  return (
    <div className="settings-page">
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.telemetry.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.telemetry.title")}
        </h1>
        <p
          data-settings-label={t("settings.telemetry.description")}
          className="text-xs leading-relaxed text-muted-foreground"
        >
          {t("settings.telemetry.description")}
        </p>
      </header>

      {loading ? (
        <SettingsSection title={t("settings.telemetry.loading")}>
          <SettingsRow label={t("settings.telemetry.loading")}>
            <Spinner />
          </SettingsRow>
        </SettingsSection>
      ) : error ? (
        <SettingsSection title={t("settings.telemetry.unavailable")}>
          <SettingsRow label={t("settings.telemetry.unavailable")} description={error}>
            <Badge variant="outline">{t("settings.telemetry.off")}</Badge>
          </SettingsRow>
        </SettingsSection>
      ) : settings ? (
        <>
          <SettingsSection
            title={t("settings.telemetry.collection")}
            description={t("settings.telemetry.collectionHint", {
              hours: settings.interval_hours,
            })}
          >
            <SettingsRow
              label={t("settings.telemetry.enable")}
              description={
                settings.enabled
                  ? t("settings.telemetry.enabledOn", {
                      hours: settings.interval_hours,
                    })
                  : t("settings.telemetry.offHint")
              }
            >
              <Switch
                checked={settings.enabled}
                disabled={saving}
                onCheckedChange={(next) => void save({ enabled: next })}
                data-testid="telemetry-master-switch"
                aria-label={t("settings.telemetry.enable")}
              />
            </SettingsRow>

            <SettingsRow
              label={t("settings.telemetry.endpoint")}
              description={t("settings.telemetry.endpointHint")}
            >
              <input
                className={`${INPUT_CLASS} w-[min(22rem,60vw)]`}
                value={endpoint}
                onChange={(event) => setEndpoint(event.target.value)}
                onBlur={() => {
                  const next = endpoint.trim();
                  if (next !== settings.endpoint) {
                    void save({ endpoint: next }, t("settings.telemetry.saved"));
                  }
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") event.currentTarget.blur();
                }}
                placeholder="https://collector.example.com/api/v1/report"
                spellCheck={false}
                disabled={saving}
                aria-label={t("settings.telemetry.endpoint")}
              />
            </SettingsRow>

            <SettingsRow
              label={t("settings.telemetry.ingestKey")}
              description={
                settings.has_ingest_key
                  ? t("settings.telemetry.ingestKeySet")
                  : t("settings.telemetry.ingestKeyHint")
              }
            >
              <div className="flex flex-wrap items-center justify-end gap-2">
                <input
                  className={`${INPUT_CLASS} w-[min(14rem,50vw)]`}
                  type="password"
                  value={ingestKey}
                  onChange={(event) => setIngestKey(event.target.value)}
                  placeholder={
                    settings.has_ingest_key
                      ? t("settings.telemetry.ingestKeyReplace")
                      : t("settings.telemetry.ingestKeyPlaceholder")
                  }
                  autoComplete="off"
                  spellCheck={false}
                  disabled={saving}
                  aria-label={t("settings.telemetry.ingestKey")}
                />
                <Button
                  variant="outline"
                  size="sm"
                  disabled={saving || !ingestKey.trim()}
                  onClick={() => {
                    void save(
                      { ingest_key: ingestKey.trim() },
                      t("settings.telemetry.saved"),
                    ).then((ok) => {
                      if (ok) setIngestKey("");
                    });
                  }}
                >
                  {t("settings.telemetry.save")}
                </Button>
                {settings.has_ingest_key ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={saving}
                    onClick={() => void save({ ingest_key: "" })}
                  >
                    {t("settings.telemetry.clear")}
                  </Button>
                ) : null}
              </div>
            </SettingsRow>
          </SettingsSection>

          <SettingsSection
            title={t("settings.telemetry.status")}
            description={t("settings.telemetry.statusHint")}
          >
            <SettingsRow label={t("settings.telemetry.lastSent")}>
              <span className="text-xs text-muted-foreground">
                {settings.last_sent_at
                  ? stamp(settings.last_sent_at)
                  : t("settings.telemetry.never")}
              </span>
            </SettingsRow>
            <SettingsRow label={t("settings.telemetry.installId")}>
              <code className="font-mono text-[11px] text-muted-foreground">
                {settings.install_id}
              </code>
            </SettingsRow>
            <SettingsRow label={t("settings.telemetry.window")}>
              <span className="text-xs text-muted-foreground">
                {t("settings.telemetry.windowValue", {
                  days: settings.window_days,
                })}
              </span>
            </SettingsRow>
            {settings.last_error ? (
              <SettingsRow
                label={t("settings.telemetry.lastError")}
                description={settings.last_error}
              >
                <Badge variant="outline">{t("settings.telemetry.failed")}</Badge>
              </SettingsRow>
            ) : null}
            <SettingsRow
              label={t("settings.telemetry.manual")}
              description={t("settings.telemetry.manualHint")}
            >
              <div className="flex flex-wrap items-center justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={previewing}
                  onClick={() => void showPreview()}
                >
                  {previewing
                    ? t("settings.telemetry.building")
                    : t("settings.telemetry.preview")}
                </Button>
                <Button
                  size="sm"
                  disabled={sending || !settings.enabled}
                  onClick={() => void sendNow()}
                >
                  {sending
                    ? t("settings.telemetry.sending")
                    : t("settings.telemetry.sendNow")}
                </Button>
                {settings.dashboard_url ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() =>
                      window.open(
                        settings.dashboard_url,
                        "_blank",
                        "noopener,noreferrer",
                      )
                    }
                  >
                    {t("settings.telemetry.openDashboard")}
                  </Button>
                ) : null}
              </div>
            </SettingsRow>
          </SettingsSection>

          {preview ? (
            <SettingsSection
              title={t("settings.telemetry.previewTitle")}
              description={t("settings.telemetry.previewSummary", {
                models: preview.models.length,
                trainings: preview.trainings.length,
                merges: preview.merges.length,
                days: preview.usage.length,
              })}
              action={
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setPreview(null)}
                >
                  {t("settings.telemetry.hidePreview")}
                </Button>
              }
            >
              <pre className="max-h-72 overflow-auto rounded-md border border-border bg-muted/30 p-3 font-mono text-[10px] leading-relaxed">
                {JSON.stringify(preview, null, 2)}
              </pre>
            </SettingsSection>
          ) : null}

          <SettingsSection
            title={t("settings.telemetry.privacyTitle")}
            description={t("settings.telemetry.privacyHint")}
          >
            <div className="grid gap-4 py-3 sm:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <h3 className="text-xs font-medium text-foreground">
                  {t("settings.telemetry.sentTitle")}
                </h3>
                <ul className="flex flex-col gap-1 text-[11px] leading-relaxed text-muted-foreground">
                  <li>· {t("settings.telemetry.sentTokens")}</li>
                  <li>· {t("settings.telemetry.sentModels")}</li>
                  <li>· {t("settings.telemetry.sentTraining")}</li>
                  <li>· {t("settings.telemetry.sentHardware")}</li>
                </ul>
              </div>
              <div className="flex flex-col gap-1.5">
                <h3 className="text-xs font-medium text-foreground">
                  {t("settings.telemetry.neverTitle")}
                </h3>
                <ul className="flex flex-col gap-1 text-[11px] leading-relaxed text-muted-foreground">
                  <li>· {t("settings.telemetry.neverPrompts")}</li>
                  <li>· {t("settings.telemetry.neverFiles")}</li>
                  <li>· {t("settings.telemetry.neverIdentity")}</li>
                  <li>· {t("settings.telemetry.neverSecrets")}</li>
                </ul>
              </div>
            </div>
            <SettingsRow
              label={t("settings.telemetry.resetId")}
              description={t("settings.telemetry.resetIdHint")}
              destructive
            >
              <Button
                variant="outline"
                size="sm"
                disabled={saving}
                onClick={() =>
                  void save(
                    { reset_install_id: true },
                    t("settings.telemetry.resetDone"),
                  )
                }
              >
                {t("settings.telemetry.resetIdAction")}
              </Button>
            </SettingsRow>
          </SettingsSection>
        </>
      ) : null}
    </div>
  );
}
