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
  PlayIcon,
  Scissor01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, type ReactNode, useState } from "react";
import {
  type SurgeryPlan,
  type SurgeryResult,
  type SurgerySpec,
  planSurgery,
  runSurgery,
} from "../api";
import { HubPickerField, type PickerMode } from "./hub-picker-field";
import { LabsPanel } from "./labs-panel";

const INPUT_CLASS =
  "h-8 min-w-0 rounded-lg border border-border bg-background px-2 text-sm text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring";

const FFN_CHOICES = [
  { value: "1", label: "100%" },
  { value: "0.9", label: "90%" },
  { value: "0.8", label: "80%" },
  { value: "0.75", label: "75%" },
  { value: "0.5", label: "50%" },
];

const EXPERT_CHOICES = [2, 3, 4, 6, 8];

function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}): ReactElement {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-ui-10 text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

/**
 * Prune a dense checkpoint, then upcycle what is left into routed experts.
 *
 * Every knob here is a real choice the backend honours or refuses: the plan
 * below the form is priced from the checkpoint's headers before a weight is
 * read, so a request that would produce an unloadable checkpoint is blocked
 * with the reason instead of dying halfway through a write.
 */
export function SurgeryPanel(): ReactElement {
  const t = useT();
  const adoptTrainingModel = useTrainingConfigStore(
    (state) => state.selectTrainingModel,
  );
  const [model, setModel] = useState("");
  const [modelMode, setModelMode] = useState<PickerMode>("local");
  const [dropLayers, setDropLayers] = useState("0");
  const [layerStrategy, setLayerStrategy] = useState<"last" | "uniform">(
    "last",
  );
  const [dropHeads, setDropHeads] = useState("0");
  const [ffnKeep, setFfnKeep] = useState("1");
  const [upcycle, setUpcycle] = useState(false);
  const [experts, setExperts] = useState(4);
  const [topK, setTopK] = useState(1);
  const [plan, setPlan] = useState<SurgeryPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [result, setResult] = useState<SurgeryResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const numberOrZero = (value: string): number => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : 0;
  };

  const buildSpec = (): SurgerySpec => ({
    model: model.trim(),
    drop_layers: numberOrZero(dropLayers),
    layer_strategy: layerStrategy,
    drop_heads: numberOrZero(dropHeads),
    ffn_keep: Number(ffnKeep),
    experts: upcycle ? experts : 0,
    top_k: upcycle ? Math.min(topK, experts) : 0,
  });

  const clear = () => {
    setPlan(null);
    setResult(null);
    setError(null);
  };

  const makePlan = async () => {
    const spec = buildSpec();
    if (!spec.model) {
      setError(t("labs.surgery.needModel"));
      return;
    }
    setPlanning(true);
    setError(null);
    setResult(null);
    try {
      setPlan(await planSurgery(spec));
    } catch (failure) {
      setPlan(null);
      setError(
        failure instanceof Error
          ? failure.message
          : t("labs.surgery.planFailed"),
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
      const job = await runSurgery(buildSpec());
      for (;;) {
        const current = await fetchJob(job.job_id);
        setPhase(
          String(
            current.progress?.message ??
              current.progress?.stage ??
              t("labs.surgery.running"),
          ),
        );
        if (current.status === "completed") {
          setResult((current.result ?? {}) as unknown as SurgeryResult);
          toast.success(t("labs.surgery.completed"));
          break;
        }
        if (current.status === "failed") {
          throw new Error(current.error || t("labs.surgery.failed"));
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    } catch (failure) {
      const message =
        failure instanceof Error ? failure.message : t("labs.surgery.failed");
      setError(message);
      toast.error(t("labs.surgery.failed"), { description: message });
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
    toast.success(t("labs.surgery.adopted"));
  };

  return (
    <LabsPanel
      title={t("labs.surgery.title")}
      description={t("labs.surgery.description")}
    >
      <div className="flex flex-col gap-3">
        <div className="flex min-w-0 flex-col gap-1.5">
          <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
            {t("labs.surgery.model")}
          </span>
          <HubPickerField
            kind="model"
            value={model}
            mode={modelMode}
            onModeChange={setModelMode}
            onChange={(value) => {
              setModel(value);
              clear();
            }}
            placeholder={t("labs.surgery.modelPlaceholder")}
          />
          <span className="text-ui-10 text-muted-foreground/70">
            {t("labs.surgery.modelHint")}
          </span>
        </div>

        <div className="rounded-xl border border-border/60 bg-muted/20 p-3">
          <h3 className="text-ui-12 font-semibold text-foreground">
            {t("labs.surgery.pruneTitle")}
          </h3>
          <p className="mt-1 max-w-3xl text-ui-10p5 leading-relaxed text-muted-foreground">
            {t("labs.surgery.pruneHint")}
          </p>
          <div className="mt-3 grid gap-3 @xl/train-card:grid-cols-4">
            <Field label={t("labs.surgery.dropLayers")}>
              <input
                type="number"
                min="0"
                className={INPUT_CLASS}
                value={dropLayers}
                onChange={(event) => {
                  setDropLayers(event.target.value);
                  clear();
                }}
              />
            </Field>
            <Field label={t("labs.surgery.layerStrategy")}>
              <select
                className={INPUT_CLASS}
                value={layerStrategy}
                onChange={(event) => {
                  setLayerStrategy(event.target.value as "last" | "uniform");
                  clear();
                }}
              >
                <option value="last">{t("labs.surgery.strategyLast")}</option>
                <option value="uniform">
                  {t("labs.surgery.strategyUniform")}
                </option>
              </select>
            </Field>
            <Field label={t("labs.surgery.dropHeads")}>
              <input
                type="number"
                min="0"
                className={INPUT_CLASS}
                value={dropHeads}
                onChange={(event) => {
                  setDropHeads(event.target.value);
                  clear();
                }}
              />
            </Field>
            <Field label={t("labs.surgery.ffnKeep")}>
              <select
                className={INPUT_CLASS}
                value={ffnKeep}
                onChange={(event) => {
                  setFfnKeep(event.target.value);
                  clear();
                }}
              >
                {FFN_CHOICES.map((choice) => (
                  <option key={choice.value} value={choice.value}>
                    {choice.label}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <p className="mt-2 text-ui-10 text-muted-foreground/70">
            {t("labs.surgery.headsHint")}
          </p>
        </div>

        <div className="rounded-xl border border-border/60 bg-muted/20 p-3">
          <label className="flex items-start gap-2">
            <input
              type="checkbox"
              className="mt-0.5 size-3.5"
              checked={upcycle}
              onChange={(event) => {
                setUpcycle(event.target.checked);
                clear();
              }}
            />
            <span className="flex min-w-0 flex-col gap-1">
              <span className="text-ui-12 font-semibold text-foreground">
                {t("labs.surgery.upcycle")}
              </span>
              <span className="text-ui-10p5 leading-relaxed text-muted-foreground">
                {t("labs.surgery.upcycleHint")}
              </span>
            </span>
          </label>
          {upcycle ? (
            <div className="mt-3 grid gap-3 @xl/train-card:grid-cols-2">
              <Field label={t("labs.surgery.experts")}>
                <select
                  className={INPUT_CLASS}
                  value={String(experts)}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    setExperts(next);
                    setTopK((current) => Math.min(current, next));
                    clear();
                  }}
                >
                  {EXPERT_CHOICES.map((count) => (
                    <option key={count} value={String(count)}>
                      {count}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={t("labs.surgery.topK")}>
                <select
                  className={INPUT_CLASS}
                  value={String(topK)}
                  onChange={(event) => {
                    setTopK(Number(event.target.value));
                    clear();
                  }}
                >
                  {Array.from({ length: experts }, (_, index) => index + 1).map(
                    (count) => (
                      <option key={count} value={String(count)}>
                        {count}
                      </option>
                    ),
                  )}
                </select>
              </Field>
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => void makePlan()}
            disabled={planning || running}
          >
            {planning ? <Spinner /> : null}
            <HugeiconsIcon icon={Scissor01Icon} className="size-3.5" />
            {planning ? t("labs.surgery.planning") : t("labs.surgery.plan")}
          </Button>
          {plan?.runnable ? (
            <Button size="sm" onClick={() => void start()} disabled={running}>
              {running ? <Spinner /> : null}
              <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
              {running ? t("labs.surgery.running") : t("labs.surgery.run")}
            </Button>
          ) : null}
          {result ? (
            <Button size="sm" variant="outline" onClick={useCheckpoint}>
              <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
              {t("labs.surgery.use")}
            </Button>
          ) : null}
          {plan || result || error ? (
            <Button size="sm" variant="ghost" onClick={clear}>
              {t("labs.surgery.clear")}
            </Button>
          ) : null}
        </div>

        {phase ? (
          <p className="text-ui-11 text-muted-foreground">{phase}</p>
        ) : null}

        {plan ? (
          <div className="flex flex-col gap-2">
            <div className="grid gap-2 text-ui-11 sm:grid-cols-3">
              <div className="rounded-lg border border-border/50 px-2.5 py-2">
                <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                  {t("labs.surgery.before")}
                </div>
                <div className="font-medium text-foreground">
                  {plan.base.params_label}
                </div>
                <div className="text-ui-10 text-muted-foreground">
                  {t("labs.surgery.layerCount", {
                    count: String(plan.base.layers),
                  })}
                </div>
              </div>
              <div className="rounded-lg border border-border/50 px-2.5 py-2">
                <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                  {t("labs.surgery.after")}
                </div>
                <div className="font-medium text-foreground">
                  {plan.projection.params_label}
                </div>
                <div className="text-ui-10 text-muted-foreground">
                  {t("labs.surgery.ratio", {
                    ratio: plan.projection.ratio.toFixed(2),
                  })}
                </div>
              </div>
              <div className="rounded-lg border border-border/50 px-2.5 py-2">
                <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                  {t("labs.surgery.disk")}
                </div>
                <div className="font-medium text-foreground">
                  {plan.projection.bytes_label}
                </div>
                <div className="text-ui-10 text-muted-foreground">
                  {plan.moe
                    ? t("labs.surgery.moeTarget", {
                        architecture: plan.moe.architecture,
                      })
                    : plan.architecture}
                </div>
              </div>
            </div>

            <p className="text-ui-10p5 leading-relaxed text-muted-foreground">
              {t("labs.surgery.shapeSummary", {
                layers: String(plan.prune.layers.keep),
                heads: String(plan.prune.heads.keep),
                kv: String(plan.prune.heads.kv_keep),
                ffn: String(plan.prune.ffn.keep),
              })}
            </p>
            {plan.prune.heads.removed > plan.prune.heads.requested ? (
              <p className="text-ui-10p5 leading-relaxed text-status-warning">
                {t("labs.surgery.headsRounded", {
                  requested: String(plan.prune.heads.requested),
                  removed: String(plan.prune.heads.removed),
                  group: String(plan.prune.heads.group_size),
                })}
              </p>
            ) : null}

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
          </div>
        ) : null}

        {result ? (
          <p className="text-ui-10p5 leading-relaxed text-muted-foreground">
            {t("labs.surgery.result", {
              path: result.checkpoint,
              tensors: String(result.written_tensors),
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
