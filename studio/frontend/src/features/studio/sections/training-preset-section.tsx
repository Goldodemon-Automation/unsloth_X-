// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { ComputeCapabilityCard } from "@/features/compute-targets/compute-capability-card";
import { ComputeTargetPicker } from "@/features/compute-targets/compute-target-picker";
import { ComputeTargetsNotice } from "@/features/compute-targets/compute-targets-notice";
import type {
  ComputeTarget,
  ComputeTargetRecommendation,
} from "@/features/compute-targets/types";
import { useComputeTargets } from "@/features/compute-targets/use-compute-targets";
import type { NpuAssistRequested } from "@/features/npu/npu-training-assist";
import { NpuTrainingCard } from "@/features/npu/npu-training-card";
import { useNpuTrainingAssist } from "@/features/npu/use-npu-training-assist";
import { useTrainingConfigStore } from "@/features/training";
import {
  type TrainingFitReport,
  fetchTrainingFit,
  recommendationDiff,
  withTargetRecommendation,
} from "@/features/training/api/fit-api";
import {
  BUILTIN_PRESETS,
  type StoredTrainingPreset,
  applyPreset,
  capturePresetValues,
  loadStoredPresets,
  writeStoredPresets,
} from "@/features/training/lib/training-presets";
import { useInferenceGpuInfo } from "@/hooks/use-gpu-info";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  DashboardSpeed01Icon,
  Delete02Icon,
  FloppyDiskIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";

function verdictTone(verdict: TrainingFitReport["verdict"]): string {
  if (verdict === "fits") {
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  }
  if (verdict === "tight") {
    return "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300";
  }
  if (verdict === "over") {
    return "border-destructive/30 bg-destructive/10 text-destructive";
  }
  return "border-border bg-muted/40 text-muted-foreground";
}

export function TrainingPresetSection(): ReactElement {
  const t = useT();
  const gpu = useInferenceGpuInfo();
  const selectedModel = useTrainingConfigStore((state) => state.selectedModel);
  const trainingMethod = useTrainingConfigStore(
    (state) => state.trainingMethod,
  );
  const contextLength = useTrainingConfigStore((state) => state.contextLength);
  const batchSize = useTrainingConfigStore((state) => state.batchSize);
  const gradientAccumulation = useTrainingConfigStore(
    (state) => state.gradientAccumulation,
  );
  const loraRank = useTrainingConfigStore((state) => state.loraRank);
  const gradientCheckpointing = useTrainingConfigStore(
    (state) => state.gradientCheckpointing,
  );

  // The device the run will use. The hook caches, so mounting the wizard again does not re-probe the
  // hardware; `choose` stores the choice server-side and updates the cache, so the capacity check
  // below sizes for the same device the run will use.
  const {
    data: computeTargets,
    choose: chooseComputeTarget,
    loading: computeTargetsLoading,
    error: computeTargetsError,
    retry: retryComputeTargets,
  } = useComputeTargets();

  // The NPU is never the primary device, so what it can do here is decided once and offered beside
  // the picker rather than hidden in settings: it changes how the run is arranged.
  // The card offers "assist" and "off" today; the request type also carries "full" because the route
  // explains that refusal, and a future control (or a caller above this section) can ask for it.
  const [npuRequested, setNpuRequested] = useState<NpuAssistRequested>("auto");
  const npuAssist = useNpuTrainingAssist({
    requested: npuRequested,
    active: Boolean(computeTargets?.npu),
  });

  const [custom, setCustom] = useState<StoredTrainingPreset[]>([]);
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState("");
  const [fit, setFit] = useState<TrainingFitReport | null>(null);
  // The device recommendation the displayed report was sized with, if any. The wizard's
  // own values are never rewritten -- this only describes how one request was sized.
  const [sizedBy, setSizedBy] = useState<ComputeTargetRecommendation | null>(
    null,
  );
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setCustom(loadStoredPresets());
  }, []);

  const saveCurrent = () => {
    const cleaned = name.trim();
    if (!cleaned) return;
    const preset: StoredTrainingPreset = {
      id: crypto.randomUUID(),
      name: cleaned.slice(0, 60),
      description: "",
      values: capturePresetValues(),
      createdAt: new Date().toISOString(),
    };
    const next = [preset, ...custom].slice(0, 50);
    setCustom(next);
    writeStoredPresets(next);
    setName("");
    setSaving(false);
    toast.success(t("studio.presets.saved"), { description: preset.name });
  };

  const remove = (id: string) => {
    const next = custom.filter((preset) => preset.id !== id);
    setCustom(next);
    writeStoredPresets(next);
  };

  // The device the run will actually use, with the sizing the backend recommends for it.
  const activeTarget = computeTargets?.targets.find(
    (entry) => entry.id === computeTargets.resolved,
  );

  // Where the device's recommendation disagrees with the wizard's fields. The check
  // is sized by it exactly when there is something to change; the wizard's values
  // themselves are never rewritten.
  const diffFor = (target: ComputeTarget | null | undefined) =>
    target
      ? recommendationDiff(
          { batchSize, gradientAccumulation, gradientCheckpointing },
          target,
        )
      : [];

  /** The fit request the wizard's fields imply, for the device named `computeTarget`. */
  const fitRequest = (model: string, computeTarget: string | null) => ({
    model,
    method: trainingMethod,
    loadIn4Bit: trainingMethod === "qlora",
    maxSeqLength: contextLength,
    batchSize,
    gradientAccumulation,
    loraRank,
    gradientCheckpointing,
    gpuMemoryGb: gpu.available ? gpu.memoryTotalGb : null,
    // Sent so the estimate describes the device the run will actually use: an integrated GPU
    // plans against shared memory, a CPU run against RAM.
    computeTarget,
  });

  const use = (
    preset: StoredTrainingPreset | (typeof BUILTIN_PRESETS)[number],
  ) => {
    applyPreset(preset);
    setFit(null);
    setSizedBy(null);
    toast.success(t("studio.presets.applied"), { description: preset.name });
  };

  /**
   * The capacity check. By default it sends exactly the wizard's fields; when the
   * selected device's own recommendation disagrees with them, the request is sized
   * with that recommendation instead and the muted line names it. The store's values
   * are never rewritten -- only this one request is sized differently.
   *
   * `repicked` is the device just chosen in the picker: the hook snapshot still names
   * the previous device until the next render, so a re-run right after a switch passes
   * the fresh choice in here.
   */
  const check = async (repicked?: ComputeTarget) => {
    if (!selectedModel) {
      setError(t("studio.presets.needModel"));
      return;
    }
    const target = repicked ?? activeTarget;
    const differs = diffFor(target);
    setChecking(true);
    setError(null);
    try {
      const computeTarget = repicked
        ? repicked.id
        : (computeTargets?.requested ?? computeTargets?.resolved ?? null);
      const sized = withTargetRecommendation(
        fitRequest(selectedModel, computeTarget),
        target,
        // The device's sizing is requested only where it disagrees with the wizard's
        // fields; otherwise the check keeps sending exactly what the wizard shows.
        { apply: differs.length > 0 },
      );
      const report = await fetchTrainingFit(sized);
      setFit(report);
      setSizedBy(
        differs.length > 0 && target?.recommended ? target.recommended : null,
      );
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : t("studio.presets.checkFailed"),
      );
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-ui-11 font-medium text-foreground">
            {t("studio.presets.builtin")}
          </span>
          <span className="text-ui-10 text-muted-foreground/70">
            {t("studio.presets.builtinHint")}
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {BUILTIN_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              title={preset.description}
              onClick={() => use(preset)}
              className="rounded-full border border-border/70 bg-background px-2.5 py-1 text-ui-11 text-foreground transition-colors hover:bg-muted/60"
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-border/60 pt-3">
        <span className="text-ui-11 font-medium text-foreground">
          {t("studio.presets.mine")}
        </span>
        {custom.length === 0 ? (
          <span className="text-ui-10 text-muted-foreground/70">
            {t("studio.presets.none")}
          </span>
        ) : (
          custom.map((preset) => (
            <span
              key={preset.id}
              className="flex items-center gap-1 rounded-full border border-border/70 bg-background pl-2.5 pr-1"
            >
              <button
                type="button"
                onClick={() => use(preset)}
                className="py-1 text-ui-11 text-foreground"
              >
                {preset.name}
              </button>
              <Button
                size="icon-xs"
                variant="ghost"
                aria-label={t("studio.presets.delete")}
                onClick={() => remove(preset.id)}
              >
                <HugeiconsIcon icon={Delete02Icon} className="size-3" />
              </Button>
            </span>
          ))
        )}
        {saving ? (
          <span className="flex items-center gap-1">
            <input
              // biome-ignore lint/a11y/noAutofocus: the row opens to name the preset.
              autoFocus={true}
              value={name}
              onChange={(event) => setName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") saveCurrent();
                if (event.key === "Escape") setSaving(false);
              }}
              placeholder={t("studio.presets.namePlaceholder")}
              className="h-7 w-44 rounded-lg border border-border bg-background px-2 text-ui-11 outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
            <Button
              size="sm"
              variant="outline"
              onClick={saveCurrent}
              disabled={!name.trim()}
            >
              {t("studio.presets.save")}
            </Button>
          </span>
        ) : (
          <Button size="sm" variant="outline" onClick={() => setSaving(true)}>
            <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
            {t("studio.presets.saveCurrent")}
          </Button>
        )}
      </div>

      {/* The card stays on screen even when the probe failed: a missing surface would leave
          the user with no explanation and no way to recover short of a reload. */}
      <div className="rounded-lg border border-border/70 bg-background/40 p-3">
        {/* Not going through i18n yet: the locale overlays gain this surface in the next
            translation pass, and a half-translated picker reads worse than an English one. */}
        <span className="text-ui-11 font-medium text-foreground">Run on</span>
        <p className="mt-1 text-ui-10 leading-relaxed text-muted-foreground/70">
          The device this run trains on. The capacity check below sizes the run
          for it: an integrated GPU plans against the memory it shares with the
          system, and a CPU run against RAM.
        </p>
        {/* With no list yet this line explains the empty card; with data on screen it renders
            nothing, and the picker carries the stale-list line instead. */}
        <ComputeTargetsNotice
          className="mt-2"
          error={computeTargetsError}
          loading={computeTargetsLoading}
          hasData={computeTargets !== null}
          onRetry={retryComputeTargets}
        />
        {computeTargets ? (
          <>
            <ComputeTargetPicker
              className="mt-2"
              targets={computeTargets.targets}
              npu={computeTargets.npu ?? null}
              value={computeTargets.resolved}
              // A failed refresh keeps the last known list visible but unpickable, so the
              // user cannot switch the run from an answer the backend no longer stands by.
              error={computeTargetsError}
              disabled={computeTargetsLoading}
              onChange={(id) => {
                const picked = computeTargets.targets.find(
                  (entry) => entry.id === id,
                );
                void chooseComputeTarget(id).then((chosen) => {
                  // The report on screen was sized for the device being left. When the
                  // device just picked recommends a different sizing than the wizard's
                  // fields, re-run the check with that recommendation applied and show
                  // the muted line; otherwise clear the stale report, as before.
                  const differs = chosen ? diffFor(picked) : [];
                  if (fit && picked && differs.length > 0) {
                    void check(picked);
                  } else {
                    setFit(null);
                    setSizedBy(null);
                  }
                });
              }}
            />
            {computeTargets.budget?.notes ? (
              <p className="mt-2 text-ui-10 leading-relaxed text-muted-foreground/70">
                {computeTargets.budget.notes}
              </p>
            ) : null}
            {/* The picker answers "where will this run"; this answers "what can this machine do",
                which is the question behind a greyed-out device or a missing 4-bit option. */}
            <details className="mt-2 text-ui-10 text-muted-foreground/80">
              <summary className="cursor-pointer">
                What can this machine do?
              </summary>
              <div className="mt-2">
                <ComputeCapabilityCard data={computeTargets} />
              </div>
            </details>
            {computeTargets.npu ? (
              <div className="mt-3">
                <NpuTrainingCard
                  assist={npuAssist.assist}
                  error={npuAssist.error}
                  loading={npuAssist.loading}
                  onRequest={(next) => {
                    setNpuRequested(next);
                    npuAssist.refresh();
                  }}
                />
              </div>
            ) : null}
          </>
        ) : null}
      </div>

      <div className="rounded-lg border border-border/70 bg-background/40 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-1.5 text-ui-11 font-medium text-foreground">
            <HugeiconsIcon icon={DashboardSpeed01Icon} className="size-3.5" />
            {t("studio.presets.fitTitle")}
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void check()}
            disabled={checking}
          >
            {checking ? (
              <Spinner />
            ) : (
              <HugeiconsIcon icon={DashboardSpeed01Icon} className="size-3.5" />
            )}
            {checking
              ? t("studio.presets.checking")
              : t("studio.presets.check")}
          </Button>
          {fit ? (
            <Badge
              variant="outline"
              className={cn("ml-auto capitalize", verdictTone(fit.verdict))}
            >
              {fit.verdict}
            </Badge>
          ) : null}
        </div>
        <p className="mt-1 text-ui-10 leading-relaxed text-muted-foreground/70">
          {t("studio.presets.fitHint")}
        </p>
        {/* Not going through i18n yet, like the device card above: the locale overlays
            gain this line in the next translation pass. */}
        {sizedBy ? (
          <p className="mt-1 text-ui-10 leading-relaxed text-muted-foreground">
            Sized for this device: batch {sizedBy.batch_size}, accumulation{" "}
            {sizedBy.grad_accum}, {sizedBy.dtype}.
          </p>
        ) : null}

        {error ? (
          <p className="mt-2 text-ui-11 text-destructive">{error}</p>
        ) : null}

        {fit ? (
          <div className="mt-3 flex flex-col gap-2">
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-ui-11">
              <span className="font-mono text-foreground">
                {fit.required_gb.toFixed(1)} GB
              </span>
              <span className="text-ui-10 text-muted-foreground">
                {t("studio.presets.required")}
              </span>
              {fit.budget_gb ? (
                <span className="text-ui-10 text-muted-foreground">
                  {t("studio.presets.budget", {
                    gb: fit.budget_gb.toFixed(1),
                  })}
                </span>
              ) : null}
              {fit.headroom_gb !== null ? (
                <span className="text-ui-10 text-muted-foreground">
                  {t("studio.presets.headroom", {
                    gb: fit.headroom_gb.toFixed(1),
                  })}
                </span>
              ) : null}
              <span className="text-ui-10 text-muted-foreground/70">
                {fit.params_label} · {fit.trainable_params.toLocaleString()}{" "}
                {t("studio.presets.trainable")}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className={cn(
                  "h-full rounded-full",
                  fit.verdict === "fits"
                    ? "bg-emerald-500"
                    : fit.verdict === "tight"
                      ? "bg-amber-500"
                      : "bg-destructive",
                )}
                style={{
                  width: `${Math.min(
                    100,
                    fit.budget_gb
                      ? (fit.required_gb / fit.budget_gb) * 100
                      : 100,
                  )}%`,
                }}
              />
            </div>
            {fit.suggestions.length > 0 ? (
              <ul className="flex flex-col gap-1 rounded-lg border border-border/60 bg-muted/20 p-2.5 text-ui-10 text-muted-foreground">
                {fit.suggestions.map((suggestion) => (
                  <li key={suggestion} className="flex gap-1.5">
                    {fit.verdict === "fits" ? (
                      <HugeiconsIcon
                        icon={Tick02Icon}
                        className="mt-0.5 size-3 shrink-0"
                      />
                    ) : (
                      <HugeiconsIcon
                        icon={Alert02Icon}
                        className="mt-0.5 size-3 shrink-0"
                      />
                    )}
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            ) : null}
            {fit.target_notes && fit.target_notes.length > 0 ? (
              <ul className="flex flex-col gap-1">
                {fit.target_notes.map((note) => (
                  <li
                    key={note}
                    className="text-ui-10 leading-relaxed text-muted-foreground/70"
                  >
                    {note}
                  </li>
                ))}
              </ul>
            ) : null}
            <details className="text-ui-10 text-muted-foreground/80">
              <summary className="cursor-pointer">
                {t("studio.presets.assumptions")}
              </summary>
              <ul className="mt-1 flex list-disc flex-col gap-0.5 pl-4">
                {fit.assumptions.map((assumption) => (
                  <li key={assumption}>{assumption}</li>
                ))}
              </ul>
            </details>
          </div>
        ) : null}
      </div>
    </div>
  );
}
