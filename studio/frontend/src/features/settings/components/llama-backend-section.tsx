// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatBytes } from "@/features/hub";
import {
  FolderBrowser,
  invalidateLlamaFlagCatalog,
} from "@/features/model-picker";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  type LlamaArchitecturesStatus,
  activateLlamaArchitecture,
  installLlamaArchitecture,
  loadLlamaArchitectures,
} from "../api/llama-architectures";
import {
  type LlamaBackendOption,
  isLlamaBackend,
  llamaBackendSelectionNeedsApply,
  visibleLlamaBackendOptions,
} from "../api/llama-backend";
import {
  type LlamaCppPathSettings,
  loadLlamaCppPathSettings,
  updateLlamaCppPathSettings,
} from "../api/llama-cpp-path";
import { useLlamaBackendSwitch } from "../hooks/use-llama-backend-switch";
import {
  architectureActionState,
  architectureStatusText,
} from "../lib/llama-architecture-status";
import { backendDisplayName } from "../lib/llama-backend-labels";
import {
  type LlamaModelSupport,
  isLlamaModelSupport,
  useSettingsPanelPrefsStore,
} from "../stores/settings-panel-prefs-store";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const MODEL_SUPPORT_OPTIONS: ReadonlyArray<{
  value: LlamaModelSupport;
  labelKey: TranslationKey;
}> = [
  {
    value: "normal",
    labelKey: "settings.resources.llamaBackend.modelSupport.profiles.normal",
  },
  {
    value: "dory",
    labelKey: "settings.resources.llamaBackend.modelSupport.profiles.dory",
  },
  {
    value: "bee",
    labelKey: "settings.resources.llamaBackend.modelSupport.profiles.bee",
  },
  {
    value: "neurall",
    labelKey: "settings.resources.llamaBackend.modelSupport.profiles.neurall",
  },
  {
    value: "ik",
    labelKey: "settings.resources.llamaBackend.modelSupport.profiles.ik",
  },
  {
    value: "x",
    labelKey: "settings.resources.llamaBackend.modelSupport.profiles.x",
  },
];

const JOB_POLL_MS = 900;

// A dead endpoint must not hold the row on "Installing" for ever: after this
// many unanswered checks (POLL_GIVE_UP_AFTER * JOB_POLL_MS) the row says the
// install could not be tracked and unlocks itself.
const POLL_GIVE_UP_AFTER = 20;

const UNSUPPORTED_REASONS: Record<string, TranslationKey> = {
  not_installed: "settings.resources.llamaBackend.unsupported.notInstalled",
  local_link: "settings.resources.llamaBackend.unsupported.localLink",
  source_build: "settings.resources.llamaBackend.unsupported.sourceBuild",
  no_install_dir: "settings.resources.llamaBackend.unsupported.notInstalled",
  custom_path: "settings.resources.llamaBackend.unsupported.customPath",
  unresolved: "settings.resources.llamaBackend.unsupported.unresolved",
  update_checks_disabled:
    "settings.resources.llamaBackend.unsupported.updateChecksDisabled",
};

function ModelSupportRow({ onChanged }: { onChanged: () => void }) {
  const t = useT();
  const modelSupport = useSettingsPanelPrefsStore(
    (state) => state.llamaModelSupport,
  );
  const setModelSupport = useSettingsPanelPrefsStore(
    (state) => state.setLlamaModelSupport,
  );
  const [status, setStatus] = useState<LlamaArchitecturesStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [installing, setInstalling] = useState<LlamaModelSupport | null>(null);
  // The job's outcome stays on screen after the toast is gone: a run must never
  // end with nothing said.
  const [result, setResult] = useState<{
    tone: "success" | "error";
    text: string;
    detail: string | null;
  } | null>(null);
  const mounted = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const polling = useRef(false);
  const failedPolls = useRef(0);
  // Set only by the confirm button, so closing the dialog to start the job does
  // not look like a cancel and rewind the picker.
  const confirmAccepted = useRef(false);
  const statusId = useId();

  const refresh = useCallback(async (forceRefresh = false) => {
    try {
      const next = await loadLlamaArchitectures(forceRefresh);
      if (mounted.current) {
        setStatus(next);
        setLoadError(null);
      }
      return next;
    } catch (reason) {
      if (mounted.current) {
        setLoadError(reason instanceof Error ? reason.message : String(reason));
      }
      return null;
    }
  }, []);

  // The picker names profiles by their translated option label, never by the
  // server's English one.
  const profileLabel = useCallback(
    (id: LlamaModelSupport | null | undefined) =>
      t(
        MODEL_SUPPORT_OPTIONS.find((option) => option.value === id)?.labelKey ??
          "settings.resources.llamaBackend.modelSupport.profiles.normal",
      ),
    [t],
  );

  const stopPolling = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    if (!mounted.current || pollTimer.current) return;
    failedPolls.current = 0;
    pollTimer.current = setInterval(() => {
      if (polling.current) return;
      polling.current = true;
      void (async () => {
        try {
          // The release lookup is cached; force only on open and final refresh.
          const next = await refresh(false);
          if (!next) {
            // Unanswered checks are budgeted: a dead endpoint must not leave the
            // row frozen on "Installing" with no way out.
            failedPolls.current += 1;
            if (failedPolls.current >= POLL_GIVE_UP_AFTER) {
              stopPolling();
              setBusy(false);
              setInstalling(null);
              setResult({
                tone: "error",
                text: t(
                  "settings.resources.llamaBackend.modelSupport.confirm.stalled",
                ),
                detail: null,
              });
            }
            return;
          }
          failedPolls.current = 0;
          if (next.job.state === "running") return;
          stopPolling();
          setBusy(false);
          setInstalling(null);
          onChanged();
          if (next.job.state === "success") {
            const profile = next.profiles.find(
              (entry) => entry.id === next.job.architecture,
            );
            const profileName = profileLabel(next.job.architecture);
            if (next.job.activated === false) {
              setResult({
                tone: "error",
                text: t(
                  "settings.resources.llamaBackend.modelSupport.status.activateFailed",
                ),
                detail: next.job.message || null,
              });
              toast.error(
                t(
                  "settings.resources.llamaBackend.modelSupport.status.activateFailed",
                ),
                { description: next.job.message || undefined },
              );
            } else {
              const text = t(
                "settings.resources.llamaBackend.modelSupport.status.installedToast",
                { profile: profileName },
              );
              setResult({
                tone: "success",
                text,
                detail: next.job.message || profile?.installedTag || null,
              });
              toast.success(text, {
                description:
                  next.job.message || profile?.installedTag || undefined,
              });
            }
          } else if (next.job.state === "error") {
            const text = t(
              "settings.resources.llamaBackend.modelSupport.status.installFailed",
            );
            setResult({
              tone: "error",
              text,
              detail: (next.job.error ?? next.job.message) || null,
            });
            toast.error(text, {
              description: next.job.error ?? next.job.message,
            });
          }
        } finally {
          polling.current = false;
        }
      })();
    }, JOB_POLL_MS);
  }, [onChanged, profileLabel, refresh, stopPolling, t]);

  useEffect(() => {
    mounted.current = true;
    // Fetching is an external subscription; the guard silences the render-lint
    // heuristic that cannot see the async boundary.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh(true);
    return () => {
      mounted.current = false;
      stopPolling();
    };
  }, [refresh, stopPolling]);

  // The server's stored profile is authoritative: installs, switches from another
  // tab, and a failed activation all update it. A confirmed install is the one
  // exception: it sits between the selection and the job it starts, and snapping
  // the picker back during that round trip would undo the choice just confirmed.
  useEffect(() => {
    if (!status || busy || confirmOpen || status.job.state === "running") return;
    if (isLlamaModelSupport(status.active) && status.active !== modelSupport) {
      setModelSupport(status.active);
    }
  }, [busy, confirmOpen, modelSupport, setModelSupport, status]);

  // A job may have been started by another tab or by the engine picker. Follow
  // it to completion instead of leaving the row frozen on "Installing".
  useEffect(() => {
    if (status?.job.state === "running") startPolling();
  }, [startPolling, status?.job.state]);

  const activateNow = useCallback(
    async (architecture: LlamaModelSupport) => {
      setBusy(true);
      setResult(null);
      try {
        const next = await activateLlamaArchitecture(architecture);
        if (!mounted.current) return;
        setStatus(next);
        setModelSupport(next.active);
        invalidateLlamaFlagCatalog();
        onChanged();
        const text = t(
          "settings.resources.llamaBackend.modelSupport.status.activatedToast",
          { profile: profileLabel(next.active) },
        );
        setResult({ tone: "success", text, detail: next.job.message || null });
        toast.success(text, { description: next.job.message || undefined });
      } catch (reason) {
        if (!mounted.current) return;
        const message =
          reason instanceof Error ? reason.message : String(reason);
        const text = t(
          "settings.resources.llamaBackend.modelSupport.status.activateFailed",
        );
        setResult({ tone: "error", text, detail: message });
        toast.error(text, { description: message });
        await refresh();
      } finally {
        if (mounted.current) setBusy(false);
      }
    },
    [onChanged, profileLabel, refresh, setModelSupport, t],
  );

  const openConfirm = () => {
    // A new confirmation is never the tail of the previous one; clear the
    // accepted flag so cancelling this dialog restores the active row.
    confirmAccepted.current = false;
    setConfirmOpen(true);
  };

  const runInstall = useCallback(
    async (architecture: LlamaModelSupport) => {
      setBusy(true);
      setInstalling(architecture);
      setResult(null);
      try {
        const started = await installLlamaArchitecture(architecture, true);
        if (!mounted.current) return;
        setStatus((current) =>
          current ? { ...current, job: started.job } : current,
        );
        if (!started.started) {
          setBusy(false);
          setInstalling(null);
          const text =
            started.message ??
            t(
              "settings.resources.llamaBackend.modelSupport.status.installFailed",
            );
          setResult({ tone: "error", text, detail: null });
          toast.error(text);
          await refresh();
          return;
        }
        startPolling();
      } catch (reason) {
        if (!mounted.current) return;
        setBusy(false);
        setInstalling(null);
        const message =
          reason instanceof Error ? reason.message : String(reason);
        const text = t(
          "settings.resources.llamaBackend.modelSupport.status.installFailed",
        );
        setResult({ tone: "error", text, detail: message });
        toast.error(text, { description: message });
        await refresh();
      }
    },
    [refresh, startPolling, t],
  );

  const selectArchitecture = (next: LlamaModelSupport) => {
    const profile = status?.profiles.find((entry) => entry.id === next);
    if (status && next !== "normal" && profile && !profile.selectable) {
      toast.error(
        profile.reason === "build_not_ready"
          ? t(
              "settings.resources.llamaBackend.modelSupport.status.buildNotReady",
            )
          : profile.reason === "build_tools_missing"
            ? t(
                "settings.resources.llamaBackend.modelSupport.status.buildToolsMissing",
              )
            : t("settings.resources.llamaBackend.modelSupport.status.noBuild"),
      );
      return;
    }
    setModelSupport(next);
    setResult(null);
    if (next === "normal") {
      void activateNow(next);
      return;
    }
    if (!profile) {
      void refresh();
      return;
    }
    if (profile.installed) {
      void activateNow(next);
      return;
    }
    if (profile.installable) {
      // Selecting is not consenting: the download and the third-party binary
      // wait behind the confirmation, and cancelling snaps the picker back.
      openConfirm();
      return;
    }
    toast.error(
      t("settings.resources.llamaBackend.modelSupport.status.noBuild"),
    );
    setModelSupport(status?.active ?? "normal");
  };

  const selectedProfile = status?.profiles.find(
    (profile) => profile.id === modelSupport,
  );
  const job = status?.job;
  const { running, canInstall, canActivate } = architectureActionState(
    modelSupport,
    status,
    busy,
  );
  const runtimeLocked = status?.runtimeLocked === true;
  const isSourceBuild =
    selectedProfile?.sourceBuild === true && !selectedProfile.asset;
  const installInFlight = installing === modelSupport;

  const statusText = (() => {
    const descriptor = status
      ? architectureStatusText(modelSupport, status, {
          size: formatBytes,
          profile: profileLabel,
        })
      : null;
    if (!descriptor) return null;
    // A live message from the job is more specific than the fallback line.
    if (
      job?.state === "running" &&
      job.architecture === modelSupport &&
      job.message
    ) {
      return job.message;
    }
    return t(descriptor.key, descriptor.values);
  })();

  // The confirmation names the consequence before the button that pays it: what
  // is downloaded or built, how big, and what runs it.
  const confirmProfileName = profileLabel(modelSupport);
  const confirmTitle = t(
    selectedProfile?.installed === true
      ? "settings.resources.llamaBackend.modelSupport.confirm.updateTitle"
      : "settings.resources.llamaBackend.modelSupport.confirm.installTitle",
    { profile: confirmProfileName },
  );
  const confirmBody = isSourceBuild
    ? t("settings.resources.llamaBackend.modelSupport.confirm.buildBody", {
        profile: confirmProfileName,
      })
    : selectedProfile?.downloadSizeBytes != null
      ? t("settings.resources.llamaBackend.modelSupport.confirm.installBody", {
          profile: confirmProfileName,
          size: formatBytes(selectedProfile.downloadSizeBytes),
        })
      : t(
          "settings.resources.llamaBackend.modelSupport.confirm.installBodyUnsized",
          { profile: confirmProfileName },
        );
  const installActionLabel = selectedProfile?.installed
    ? t("settings.resources.llamaBackend.modelSupport.status.update")
    : isSourceBuild
      ? t("settings.resources.llamaBackend.modelSupport.status.buildAndUse")
      : t("settings.resources.llamaBackend.modelSupport.status.installAndUse");

  const closeConfirm = (open: boolean) => {
    setConfirmOpen(open);
    if (open) return;
    if (confirmAccepted.current) {
      confirmAccepted.current = false;
      return;
    }
    // Selecting an installable fork is not the same as confirming its download.
    // Closing the dialog without installing must leave the active runtime shown,
    // not an uninstalled row the user only inspected.
    setModelSupport(status?.active ?? "normal");
  };

  const confirmInstall = () => {
    if (!selectedProfile) return;
    confirmAccepted.current = true;
    setConfirmOpen(false);
    void runInstall(selectedProfile.id);
  };

  return (
    <>
      <SettingsRow
        label={t("settings.resources.llamaBackend.modelSupport.label")}
        description={t(
          "settings.resources.llamaBackend.modelSupport.description",
        )}
        hint={t("settings.resources.llamaBackend.modelSupport.hint")}
      >
        <Select
          value={modelSupport}
          disabled={!status || runtimeLocked || running}
          onValueChange={(next) => {
            if (isLlamaModelSupport(next)) selectArchitecture(next);
          }}
        >
          <SelectTrigger
            aria-label={t("settings.resources.llamaBackend.modelSupport.label")}
            aria-describedby={statusId}
            className="w-72"
            size="sm"
            data-testid="llama-model-support-select"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODEL_SUPPORT_OPTIONS.map((option) => {
              const profile = status?.profiles.find(
                (entry) => entry.id === option.value,
              );
              const unavailable =
                status != null &&
                option.value !== "normal" &&
                profile != null &&
                !profile.selectable;
              return (
                <SelectItem
                  key={option.value}
                  value={option.value}
                  disabled={unavailable}
                >
                  {t(option.labelKey)}
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      </SettingsRow>

      {/* Always mounted: it is the picker's aria-describedby target, so the
          reason the picker is disabled is in text beside it, not only in its
          dimming. */}
      <div
        id={statusId}
        className={cn(
          "text-xs text-muted-foreground",
          (statusText || loadError || !status) && "pb-3",
        )}
        data-testid="llama-model-support-hint"
      >
        {loadError ? (
          <span className="text-destructive">{loadError}</span>
        ) : statusText ? (
          <span>{statusText}</span>
        ) : status ? null : (
          <span>{t("common.loading")}</span>
        )}
      </div>

      {status?.active && status.active !== "normal" ? (
        <p className="pb-3 text-xs text-muted-foreground">
          {t(
            "settings.resources.llamaBackend.modelSupport.status.customPathIgnored",
          )}
        </p>
      ) : null}

      {/* Progress and the final result, with a live region that exists before
          its text changes so the change is announced. */}
      <div
        className={cn((running && job) || result ? "pb-3" : null)}
        data-testid="llama-model-support-progress"
      >
        {running && job ? (
          <Progress
            value={Math.round((job.progress ?? 0) * 100)}
            aria-label={t(
              "settings.resources.llamaBackend.modelSupport.status.installing",
              { profile: profileLabel(modelSupport) },
            )}
            className="h-1.5 w-full rounded-full bg-muted"
          />
        ) : null}
        <output aria-live="polite" className="block">
          {running && job ? (
            <span className="mt-2 block text-xs text-muted-foreground">
              {job.message ||
                t(
                  "settings.resources.llamaBackend.modelSupport.status.installing",
                  { profile: profileLabel(modelSupport) },
                )}
            </span>
          ) : null}
          {result ? (
            <span
              className={cn(
                "mt-2 block text-xs",
                result.tone === "error"
                  ? "text-destructive"
                  : "text-muted-foreground",
              )}
            >
              {result.text}
              {result.detail ? (
                <span className="text-muted-foreground"> {result.detail}</span>
              ) : null}
            </span>
          ) : null}
        </output>
      </div>

      {installInFlight || canInstall || canActivate ? (
        <div className="flex items-center gap-2 pb-3">
          {installInFlight ? (
            // Disabled while it runs, with the phase in its label: the row is
            // never left holding an install it does not name.
            <Button
              size="sm"
              disabled={true}
              aria-busy={true}
              data-testid="llama-model-support-install"
            >
              {t(
                "settings.resources.llamaBackend.modelSupport.status.installing",
                { profile: profileLabel(modelSupport) },
              )}
            </Button>
          ) : canInstall ? (
            <Button
              size="sm"
              onClick={openConfirm}
              data-testid="llama-model-support-install"
            >
              {installActionLabel}
            </Button>
          ) : null}
          {canActivate ? (
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                if (selectedProfile) void activateNow(selectedProfile.id);
              }}
              data-testid="llama-model-support-use"
            >
              {t("settings.resources.llamaBackend.modelSupport.status.use")}
            </Button>
          ) : null}
        </div>
      ) : null}

      <Dialog open={confirmOpen} onOpenChange={closeConfirm}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{confirmTitle}</DialogTitle>
            <DialogDescription>{confirmBody}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => closeConfirm(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              disabled={running}
              onClick={confirmInstall}
              data-testid="llama-model-support-install-confirm"
            >
              {installActionLabel}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function LlamaCppPathRow({ onChanged }: { onChanged: () => void }) {
  const t = useT();
  const [settings, setSettings] = useState<LlamaCppPathSettings | null>(null);
  const [draftPath, setDraftPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void loadLlamaCppPathSettings()
      .then((next) => {
        if (!active) return;
        setSettings(next);
        setDraftPath(next.path ?? "");
        setError(null);
      })
      .catch((reason) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      active = false;
    };
  }, []);

  const save = async (path: string | null) => {
    setSaving(true);
    setError(null);
    try {
      const next = await updateLlamaCppPathSettings(path);
      setSettings(next);
      setDraftPath(next.path ?? "");
      setBrowserOpen(false);
      toast.success(t("settings.resources.llamaBackend.customPath.saved"));
      onChanged();
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
      toast.error(t("settings.resources.llamaBackend.customPath.saveError"), {
        description: message,
      });
    } finally {
      setSaving(false);
    }
  };

  const pathDirty = Boolean(
    settings?.editable && draftPath.trim() !== (settings.path ?? ""),
  );
  const detail = settings
    ? settings.source === "environment"
      ? t("settings.resources.llamaBackend.customPath.environmentManaged", {
          variable: settings.environmentVariable ?? "UNSLOTH_LLAMA_CPP_PATH",
        })
      : settings.available
        ? settings.reloadRequired
          ? t("settings.resources.llamaBackend.customPath.reloadRequired")
          : settings.source === "studio"
            ? t("settings.resources.llamaBackend.customPath.active")
            : t("settings.resources.llamaBackend.customPath.bundled")
        : t("settings.resources.llamaBackend.customPath.missingBinary")
    : null;

  return (
    <>
      <SettingsRow
        label={t("settings.resources.llamaBackend.customPath.label")}
        description={t(
          "settings.resources.llamaBackend.customPath.description",
        )}
        hint={t("settings.resources.llamaBackend.customPath.hint")}
        className="max-[840px]:flex-col max-[840px]:items-stretch max-[840px]:gap-2"
        below={
          detail || settings?.source === "studio" ? (
            <div className="w-[calc(392px*var(--ui-space-scale,1))] min-w-0 max-[840px]:w-full">
              <div className="flex min-w-0 items-center justify-between gap-2 pl-3.5 pr-1 text-xs text-muted-foreground">
                {detail ? (
                  <span title={detail} className="min-w-0 truncate">
                    {detail}
                  </span>
                ) : null}
                {settings?.source === "studio" ? (
                  <Button
                    variant="link"
                    size="xs"
                    className="h-auto shrink-0 px-0 text-xs"
                    disabled={saving}
                    onClick={() => void save(null)}
                    data-testid="llama-cpp-path-reset"
                  >
                    {t("settings.resources.llamaBackend.customPath.useBundled")}
                  </Button>
                ) : null}
              </div>
            </div>
          ) : null
        }
      >
        <div className="grid w-[calc(392px*var(--ui-space-scale,1))] min-w-0 grid-cols-[minmax(0,1fr)_auto_auto] gap-x-2 max-[840px]:w-full">
          <Input
            readOnly={!settings?.editable}
            aria-label={t("settings.resources.llamaBackend.customPath.label")}
            value={draftPath}
            placeholder={
              settings
                ? t("settings.resources.llamaBackend.customPath.automatic")
                : t("common.loading")
            }
            onChange={(event) => setDraftPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && pathDirty && !saving) {
                void save(draftPath.trim() || null);
              }
            }}
            title={settings?.resolvedBinary ?? settings?.path ?? undefined}
            className="h-8 min-w-0 font-mono text-xs"
            data-testid="llama-cpp-path-input"
          />
          <Button
            variant="outline"
            size="sm"
            className="h-8"
            disabled={!pathDirty || saving}
            onClick={() => void save(draftPath.trim() || null)}
            data-testid="llama-cpp-path-save"
          >
            {saving ? t("common.saving") : t("common.save")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8"
            disabled={!settings?.editable || saving}
            onClick={() => setBrowserOpen(true)}
            data-testid="llama-cpp-path-change"
          >
            {saving
              ? t("settings.resources.llamaBackend.customPath.saving")
              : t("settings.resources.llamaBackend.customPath.change")}
          </Button>
        </div>
      </SettingsRow>

      {error ? (
        <p
          className="pb-3 text-xs text-destructive"
          data-testid="llama-cpp-path-error"
        >
          {error}
        </p>
      ) : null}

      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(path) => void save(path)}
        initialPath={settings?.path ?? undefined}
        title={t("settings.resources.llamaBackend.customPath.chooseTitle")}
        description={t(
          "settings.resources.llamaBackend.customPath.description",
        )}
        confirmLabel={t(
          "settings.resources.llamaBackend.customPath.chooseAction",
        )}
        showModelHints={false}
      />
    </>
  );
}

export function LlamaBackendSection() {
  const t = useT();
  const { status, selected, setSelected, running, apply, refresh, loadError } =
    useLlamaBackendSwitch();

  const optionLabel = (option: LlamaBackendOption) => {
    if (option.backend !== "auto") {
      return backendDisplayName(option.backend, t);
    }
    // "Automatic" alone hides the decision; naming what it picks is the
    // difference between an informed choice and a shot in the dark.
    return option.resolvedBackend
      ? t("settings.resources.llamaBackend.autoWith", {
          backend: backendDisplayName(option.resolvedBackend, t),
        })
      : backendDisplayName("auto", t);
  };

  const job = status?.job;
  const envLocked = status?.envBackend != null;
  // Null means the marker holds a choice a newer Unsloth wrote. Show it as
  // unknown rather than as Automatic, and let it be replaced deliberately.
  const unknownRecorded = status?.backendRequest === null;
  const value = selected ?? status?.backendRequest ?? "unknown";
  const options = visibleLlamaBackendOptions(
    status,
    selected ?? status?.backendRequest ?? null,
  );
  const pending = options.find((option) => option.backend === value);
  const dirty = !running && llamaBackendSelectionNeedsApply(status, selected);
  const unsupportedKey =
    status && !status.supported && !running
      ? (UNSUPPORTED_REASONS[status.reason ?? ""] ??
        "settings.resources.llamaBackend.unsupported.unresolved")
      : null;

  return (
    <SettingsSection title={t("settings.resources.llamaBackend.title")}>
      <SettingsRow
        label={t("settings.resources.llamaBackend.label")}
        description={
          status?.backend
            ? t("settings.resources.llamaBackend.runningOn", {
                backend: backendDisplayName(status.backend, t),
              })
            : t("settings.resources.llamaBackend.description")
        }
        hint={t("settings.resources.llamaBackend.hint")}
      >
        <div className="flex items-center gap-2">
          <Select
            value={value}
            disabled={!status?.supported || envLocked || running}
            onValueChange={(next) => {
              if (isLlamaBackend(next)) {
                setSelected(next);
              }
            }}
          >
            <SelectTrigger
              aria-label={t("settings.resources.llamaBackend.label")}
              className="w-52"
              size="sm"
              data-testid="llama-backend-select"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {unknownRecorded ? (
                <SelectItem value="unknown">
                  {t("settings.resources.environment.unknown")}
                </SelectItem>
              ) : null}
              {options.map((option) => (
                <SelectItem key={option.backend} value={option.backend}>
                  {optionLabel(option)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            size="sm"
            onClick={apply}
            // The same hard blockers as the Select above, not just dirtiness. An
            // environment pin with an automatic choice that has since drifted leaves
            // selectionApplied false, so the row is dirty while the Select is disabled:
            // Apply would then be the only live control, and the server rightly refuses
            // it with environment_override. Do not offer an action that cannot succeed.
            disabled={!dirty || !status?.supported || envLocked}
            data-testid="llama-backend-apply"
          >
            {running
              ? t("settings.resources.llamaBackend.applying")
              : t("settings.resources.llamaBackend.apply")}
          </Button>
        </div>
      </SettingsRow>

      <ModelSupportRow
        onChanged={() => {
          invalidateLlamaFlagCatalog();
          void refresh(true);
        }}
      />

      <LlamaCppPathRow
        onChanged={() => {
          invalidateLlamaFlagCatalog();
          void refresh(true);
        }}
      />

      {running ? (
        <div className="pb-3" data-testid="llama-backend-progress">
          <Progress
            value={Math.round((job?.progress ?? 0) * 100)}
            aria-label={t("settings.resources.llamaBackend.applying")}
            className="h-1.5 w-full rounded-full bg-muted"
          />
          <p className="mt-2 text-xs text-muted-foreground">
            {job?.message || t("settings.resources.llamaBackend.applying")}
          </p>
        </div>
      ) : null}

      {dirty ? (
        // The two things a switch costs, said before it is paid: a download, and
        // the loaded model going away with the server it runs in.
        <p className="pb-3 text-xs text-muted-foreground">
          {pending?.downloadSizeBytes
            ? t("settings.resources.llamaBackend.applyHintWithSize", {
                size: formatBytes(pending.downloadSizeBytes),
              })
            : t("settings.resources.llamaBackend.applyHint")}
        </p>
      ) : null}

      {envLocked ? (
        <p className="pb-3 text-xs text-status-warning">
          {t("settings.resources.llamaBackend.envLocked", {
            backend: backendDisplayName(status?.envBackend, t),
          })}
        </p>
      ) : null}

      {unsupportedKey ? (
        <p className="pb-3 text-xs text-muted-foreground">
          {t(unsupportedKey)}
        </p>
      ) : null}

      {loadError ? (
        <p className="pb-3 text-xs text-destructive">{loadError}</p>
      ) : null}
    </SettingsSection>
  );
}
