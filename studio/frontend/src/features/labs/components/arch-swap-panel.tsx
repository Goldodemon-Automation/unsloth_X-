// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { fetchJob } from "@/features/advanced/api/advanced-api";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  ArrowRight01Icon,
  PlayIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import {
  type ArchSwapPlan,
  type ArchSwapResult,
  type ArchSwapSpec,
  planArchSwap,
  runArchSwap,
} from "../api";
import { HubPickerField, type PickerMode } from "./hub-picker-field";
import { LabsPanel } from "./labs-panel";

const INPUT_CLASS =
  "h-8 min-w-0 rounded-lg border border-border bg-background px-2 text-sm text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring";

/**
 * Move weights from one architecture into another by role.
 *
 * The plan is the point of this panel: it lists every target tensor with the
 * source tensors it is built from and what was lost (a fused QKV split, a
 * transposed Conv1D, a zero-padded vocabulary, a repeated block). A swap that
 * hides those would produce a checkpoint that loads and predicts nothing.
 */
export function ArchSwapPanel(): ReactElement {
  const t = useT();
  const adoptTrainingModel = useTrainingConfigStore(
    (state) => state.selectTrainingModel,
  );
  const [source, setSource] = useState("");
  const [sourceMode, setSourceMode] = useState<PickerMode>("local");
  const [target, setTarget] = useState("");
  const [targetMode, setTargetMode] = useState<PickerMode>("local");
  const [extraLayers, setExtraLayers] = useState<"repeat" | "zeros">("repeat");
  const [plan, setPlan] = useState<ArchSwapPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [result, setResult] = useState<ArchSwapResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const buildSpec = (): ArchSwapSpec => ({
    source: source.trim(),
    target: target.trim(),
    extra_layers: extraLayers,
  });

  const makePlan = async () => {
    const spec = buildSpec();
    if (!spec.source || !spec.target) {
      setError(t("labs.swap.needBoth"));
      return;
    }
    setPlanning(true);
    setError(null);
    setResult(null);
    try {
      setPlan(await planArchSwap(spec));
    } catch (failure) {
      setPlan(null);
      setError(
        failure instanceof Error ? failure.message : t("labs.swap.planFailed"),
      );
    } finally {
      setPlanning(false);
    }
  };

  const start = async () => {
    if (!plan?.runnable) return;
    setRunning(true);
    setError(null);
    try {
      const job = await runArchSwap(buildSpec());
      for (;;) {
        const current = await fetchJob(job.job_id);
        setPhase(
          String(
            current.progress?.message ??
              current.progress?.stage ??
              t("labs.swap.running"),
          ),
        );
        if (current.status === "completed") {
          setResult((current.result ?? {}) as unknown as ArchSwapResult);
          toast.success(t("labs.swap.completed"));
          break;
        }
        if (current.status === "failed") {
          throw new Error(current.error || t("labs.swap.failed"));
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    } catch (failure) {
      const message =
        failure instanceof Error ? failure.message : t("labs.swap.failed");
      setError(message);
      toast.error(t("labs.swap.failed"), { description: message });
    } finally {
      setRunning(false);
      setPhase(null);
    }
  };

  const useCheckpoint = () => {
    if (!result?.checkpoint) return;
    adoptTrainingModel(result.checkpoint, null, {
      localPath: result.checkpoint,
    });
    toast.success(t("labs.swap.adopted"));
  };

  return (
    <LabsPanel
      title={t("labs.swap.title")}
      description={t("labs.swap.description")}
    >
      <div className="flex flex-col gap-3">
        <div className="grid gap-3 @2xl/train-card:grid-cols-3">
          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.swap.source")}
            </span>
            <HubPickerField
              kind="model"
              value={source}
              mode={sourceMode}
              onModeChange={setSourceMode}
              onChange={(value) => {
                setSource(value);
                setPlan(null);
              }}
              placeholder={t("labs.swap.sourcePlaceholder")}
            />
          </div>
          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.swap.target")}
            </span>
            <HubPickerField
              kind="model"
              value={target}
              mode={targetMode}
              onModeChange={setTargetMode}
              onChange={(value) => {
                setTarget(value);
                setPlan(null);
              }}
              placeholder={t("labs.swap.targetPlaceholder")}
            />
          </div>
          <label className="flex min-w-0 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.swap.extraLayers")}
            </span>
            <select
              className={INPUT_CLASS}
              value={extraLayers}
              onChange={(event) => {
                setExtraLayers(event.target.value as "repeat" | "zeros");
                setPlan(null);
              }}
            >
              <option value="repeat">{t("labs.swap.repeat")}</option>
              <option value="zeros">{t("labs.swap.zeros")}</option>
            </select>
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => void makePlan()}
            disabled={planning || running}
          >
            {planning ? <Spinner /> : null}
            <HugeiconsIcon icon={ArrowRight01Icon} className="size-3.5" />
            {planning ? t("labs.swap.planning") : t("labs.swap.plan")}
          </Button>
          {plan?.runnable ? (
            <Button size="sm" onClick={() => void start()} disabled={running}>
              {running ? <Spinner /> : null}
              <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
              {running ? t("labs.swap.running") : t("labs.swap.run")}
            </Button>
          ) : null}
          {result ? (
            <Button size="sm" variant="outline" onClick={useCheckpoint}>
              <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
              {t("labs.swap.use")}
            </Button>
          ) : null}
          {plan || result || error ? (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setPlan(null);
                setResult(null);
                setError(null);
              }}
            >
              {t("labs.swap.clear")}
            </Button>
          ) : null}
        </div>

        {phase ? (
          <p className="text-ui-11 text-muted-foreground">{phase}</p>
        ) : null}

        {plan ? (
          <div className="flex flex-col gap-3">
            <div className="grid gap-2 text-ui-11 sm:grid-cols-3">
              <div className="rounded-lg border border-border/50 px-2.5 py-2">
                <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                  {t("labs.swap.layout")}
                </div>
                <div className="font-medium text-foreground">
                  {plan.source_family} → {plan.target_family}
                </div>
              </div>
              <div className="rounded-lg border border-border/50 px-2.5 py-2">
                <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                  {t("labs.swap.blocks")}
                </div>
                <div className="font-medium text-foreground">
                  {t("labs.swap.blockCounts", {
                    source: String(plan.source_layers),
                    target: String(plan.target_layers),
                  })}
                </div>
              </div>
              <div className="rounded-lg border border-border/50 px-2.5 py-2">
                <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                  {t("labs.swap.actions")}
                </div>
                <div className="font-medium text-foreground">
                  {t("labs.swap.actionCounts", {
                    copied: String(plan.counts.copied ?? 0),
                    built: String(plan.counts.built ?? 0),
                    unmapped: String(plan.counts.unmapped ?? 0),
                  })}
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[420px] border-collapse text-left text-ui-10p5">
                <thead>
                  <tr className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                    <th className="py-1 pr-3 font-normal">
                      {t("labs.swap.role")}
                    </th>
                    <th className="py-1 pr-3 font-normal">
                      {t("labs.swap.action")}
                    </th>
                    <th className="py-1 font-normal">
                      {t("labs.swap.tensors")}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {plan.roles.map((row) => (
                    <tr key={row.role} className="border-t border-border/40">
                      <td className="py-1 pr-3 text-foreground">{row.label}</td>
                      <td className="py-1 pr-3 font-mono text-muted-foreground">
                        {row.action}
                      </td>
                      <td className="py-1 text-muted-foreground">
                        {row.tensors}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {plan.blocked_reason ? (
              <p className="flex items-start gap-1.5 text-ui-10p5 leading-relaxed text-destructive">
                <HugeiconsIcon
                  icon={Alert02Icon}
                  className="mt-0.5 size-3 shrink-0"
                />
                {plan.blocked_reason}
              </p>
            ) : null}
            {[...plan.warnings, ...plan.notes].map((note) => (
              <p
                key={note}
                className={cn(
                  "text-ui-10p5 leading-relaxed",
                  plan.warnings.includes(note)
                    ? "text-status-warning"
                    : "text-muted-foreground/80",
                )}
              >
                {note}
              </p>
            ))}
            {plan.mapping_truncated ? (
              <p className="text-ui-10 text-muted-foreground/70">
                {t("labs.swap.truncated")}
              </p>
            ) : null}
          </div>
        ) : null}

        {result ? (
          <p className="text-ui-10p5 leading-relaxed text-muted-foreground">
            {t("labs.swap.result", {
              path: result.checkpoint,
              tensors: String(result.tensors),
              zeroed: String(result.zeroed_tensors),
            })}
          </p>
        ) : null}

        {error ? (
          <p className="text-ui-10p5 leading-relaxed text-destructive">
            {error}
          </p>
        ) : null}
      </div>
    </LabsPanel>
  );
}
