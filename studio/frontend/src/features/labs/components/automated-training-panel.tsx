// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { fetchJob } from "@/features/advanced/api/advanced-api";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  ArrowDown01Icon,
  Cancel01Icon,
  Delete02Icon,
  PlayIcon,
  PlusSignIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  type AutomatedPreset,
  type AutomatedTrainingPlan,
  type AutomatedTrainingSpec,
  type LabsJob,
  cancelAutomatedTraining,
  listAutomatedTrainingPresets,
  planAutomatedTraining,
  runAutomatedTraining,
} from "../api";
import {
  AUTOMATED_CATEGORIES,
  AUTOMATED_METHODS,
  type AutomatedCategory,
  type AutomatedMethod,
  clampContinueRuns,
  clampDatasetsPerRound,
  clampRounds,
  clampRunsPerRound,
  clampSeed,
  clampStepsPerRun,
  defaultAutomatedTrainingSpec,
  isAutomatedCategory,
  isAutomatedJobCancellable,
  isAutomatedMethod,
  summarizeAutomatedProgress,
  validateAutomatedTrainingSpec,
} from "../lib/automated-training-spec";
import { HubPickerField, type PickerMode } from "./hub-picker-field";
import { LabsPanel } from "./labs-panel";

const INPUT_CLASS =
  "h-9 min-w-0 rounded-full border border-border bg-background px-3 text-sm text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring";

const CATEGORY_LABEL: Record<AutomatedCategory, TranslationKey> = {
  code: "labs.automated.categoryCode",
  syntax: "labs.automated.categorySyntax",
  reasoning: "labs.automated.categoryReasoning",
  general: "labs.automated.categoryGeneral",
};

const METHOD_LABEL: Record<AutomatedMethod, TranslationKey> = {
  cmt: "labs.automated.methodCmt",
  cpt: "labs.automated.methodCpt",
  fft: "labs.automated.methodFft",
};

const DEFAULT_SPEC = defaultAutomatedTrainingSpec();

/** Failures past this many are collapsed behind an "and N more" toggle. */
const FAILURE_PREVIEW = 5;

interface DatasetRow {
  id: string;
  path: string;
  mode: PickerMode;
  name: string;
  category: AutomatedCategory;
  evalPercent: string;
}

interface AutomatedTrainingBest {
  output_dir?: string;
  score?: number | null;
  method?: string;
  round?: number;
}

interface AutomatedTrainingResult extends Record<string, unknown> {
  output_dir?: string;
  best?: AutomatedTrainingBest | null;
  train_runs?: number;
  benchmark_runs?: number;
  failed_runs?: number;
  warnings?: string[];
  rounds?: unknown[];
  cancelled?: boolean;
}

function newRow(): DatasetRow {
  return {
    id: crypto.randomUUID(),
    path: "",
    mode: "hub",
    name: "",
    category: "general",
    evalPercent: "10",
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForAutomatedTraining(
  job: LabsJob,
  onProgress: (progress: Record<string, unknown> | null) => void,
  onStatus?: (status: string) => void,
): Promise<AutomatedTrainingResult> {
  for (let attempt = 0; attempt < 21_600; attempt += 1) {
    const current = await fetchJob(job.job_id);
    // Reported before the status branches so the panel knows a cancel request
    // no longer has anything to stop as soon as the job goes terminal.
    onStatus?.(current.status);
    const progress = current.progress;
    onProgress(
      progress
        ? { ...progress }
        : current.status === "running"
          ? { stage: "queued" }
          : null,
    );
    if (current.status === "completed") {
      return (current.result ?? {}) as AutomatedTrainingResult;
    }
    if (current.status === "failed") {
      throw new Error(current.error || "The automated training job failed.");
    }
    await sleep(2000);
  }
  throw new Error("The automated training job did not finish in twelve hours.");
}

function scoreLabel(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(3);
}

export function AutomatedTrainingPanel(): ReactElement {
  const t = useT();
  const [source, setSource] = useState("");
  const [rows, setRows] = useState<DatasetRow[]>([newRow(), newRow()]);
  const [methods, setMethods] = useState<AutomatedMethod[]>(
    DEFAULT_SPEC.methods ?? ["cmt", "cpt"],
  );
  const [datasetsPerRound, setDatasetsPerRound] = useState(
    String(DEFAULT_SPEC.datasets_per_round),
  );
  const [runsPerRound, setRunsPerRound] = useState(
    String(DEFAULT_SPEC.runs_per_round),
  );
  const [rounds, setRounds] = useState(String(DEFAULT_SPEC.rounds));
  const [continueRuns, setContinueRuns] = useState(
    String(DEFAULT_SPEC.continue_runs),
  );
  const [seed, setSeed] = useState(String(DEFAULT_SPEC.seed));
  const [stepsPerRun, setStepsPerRun] = useState(
    String(DEFAULT_SPEC.steps_per_run),
  );
  const [presets, setPresets] = useState<AutomatedPreset[]>([]);
  const [plan, setPlan] = useState<AutomatedTrainingPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Record<string, unknown> | null>(
    null,
  );
  const [result, setResult] = useState<AutomatedTrainingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [failuresExpanded, setFailuresExpanded] = useState(false);
  // The job the poll is following, so Cancel can name it. Null until the run
  // route answers, and again once the job is finished.
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  // Set by a cancel click and cleared only when the job is gone, so the button
  // stays spent instead of inviting a second request the route would 404.
  const [cancelling, setCancelling] = useState(false);
  // A failed cancel request must not look like a failed run, and must not stop
  // the poll, so it gets its own line beside the button.
  const [cancelError, setCancelError] = useState<string | null>(null);
  // A ref, not the state flag: two clicks in one tick both see running === false.
  const inFlight = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    listAutomatedTrainingPresets(controller.signal)
      .then((body) => setPresets(body.presets))
      .catch(() => {
        // Presets are a convenience; a plan also carries the same list.
      });
    return () => controller.abort();
  }, []);

  const updateRow = (id: string, patch: Partial<DatasetRow>) => {
    setRows((current) =>
      current.map((row) => (row.id === id ? { ...row, ...patch } : row)),
    );
    setPlan(null);
  };

  const toggleMethod = (method: AutomatedMethod) => {
    setMethods((current) =>
      current.includes(method)
        ? current.filter((item) => item !== method)
        : [...current, method],
    );
    setPlan(null);
  };

  const buildSpec = useCallback((): AutomatedTrainingSpec => {
    const datasets = rows
      .filter((row) => row.path.trim() !== "")
      .map((row) => ({
        ...(row.mode === "hub"
          ? { path: "", repo_id: row.path.trim() }
          : { path: row.path.trim() }),
        name: row.name.trim() || undefined,
        category: row.category,
        eval_fraction: Math.max(
          0,
          Math.min(0.5, (Number(row.evalPercent) || 0) / 100),
        ),
      }));
    return {
      source: source.trim(),
      datasets,
      methods: [...methods],
      datasets_per_round: clampDatasetsPerRound(Number(datasetsPerRound)),
      runs_per_round: clampRunsPerRound(Number(runsPerRound)),
      rounds: clampRounds(Number(rounds)),
      continue_runs: clampContinueRuns(Number(continueRuns)),
      seed: clampSeed(Number(seed)),
      steps_per_run: clampStepsPerRun(Number(stepsPerRun)),
    };
  }, [
    continueRuns,
    datasetsPerRound,
    methods,
    rounds,
    rows,
    runsPerRound,
    seed,
    source,
    stepsPerRun,
  ]);

  const applyPreset = (preset: AutomatedPreset) => {
    const spec = preset.spec;
    if (typeof spec.source === "string") setSource(spec.source);
    if (Array.isArray(spec.methods)) {
      setMethods(spec.methods.filter(isAutomatedMethod));
    }
    if (spec.datasets_per_round !== undefined) {
      setDatasetsPerRound(String(clampDatasetsPerRound(spec.datasets_per_round)));
    }
    if (spec.runs_per_round !== undefined) {
      setRunsPerRound(String(clampRunsPerRound(spec.runs_per_round)));
    }
    if (spec.rounds !== undefined) setRounds(String(clampRounds(spec.rounds)));
    if (spec.continue_runs !== undefined) {
      setContinueRuns(String(clampContinueRuns(spec.continue_runs)));
    }
    if (spec.seed !== undefined) setSeed(String(clampSeed(spec.seed)));
    if (spec.steps_per_run !== undefined) {
      setStepsPerRun(String(clampStepsPerRun(spec.steps_per_run)));
    }
    if (Array.isArray(spec.datasets) && spec.datasets.length > 0) {
      setRows(
        spec.datasets.map((dataset) => ({
          id: crypto.randomUUID(),
          path: dataset.path?.trim() || dataset.repo_id?.trim() || "",
          mode: dataset.repo_id && !dataset.path ? "hub" : "local",
          name: dataset.name ?? "",
          category: isAutomatedCategory(dataset.category)
            ? dataset.category
            : "general",
          evalPercent: String(
            Math.round((dataset.eval_fraction ?? 0.1) * 100),
          ),
        })),
      );
    }
    setPlan(null);
  };

  const makePlan = async () => {
    setPlanning(true);
    setError(null);
    setPlan(null);
    try {
      const spec = buildSpec();
      if (!spec.source) throw new Error(t("labs.automated.needSource"));
      if (spec.datasets.length < 2) {
        throw new Error(t("labs.automated.needDatasets"));
      }
      const problems = validateAutomatedTrainingSpec(spec);
      if (problems.length > 0) throw new Error(problems[0]);
      setPlan(await planAutomatedTraining(spec));
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : t("labs.automated.planFailed"),
      );
    } finally {
      setPlanning(false);
    }
  };

  const start = async () => {
    if (inFlight.current || !plan?.runnable) return;
    inFlight.current = true;
    setRunning(true);
    setError(null);
    setResult(null);
    setProgress(null);
    setFailuresExpanded(false);
    setActiveJobId(null);
    setJobStatus(null);
    setCancelling(false);
    setCancelError(null);
    try {
      const spec = buildSpec();
      const job = await runAutomatedTraining(spec);
      setActiveJobId(job.job_id);
      setJobStatus(job.status);
      setProgress({ stage: "queued", message: t("labs.automated.starting") });
      const finished = await waitForAutomatedTraining(
        job,
        setProgress,
        setJobStatus,
      );
      setProgress(null);
      setResult(finished);
      // A cancelled run ends with status "completed" and result.cancelled true;
      // a green success toast for it would misreport what happened.
      if (finished?.cancelled) {
        toast.message("Automated training cancelled");
      } else {
        toast.success(t("labs.automated.completed"));
      }
    } catch (failure) {
      const message =
        failure instanceof Error ? failure.message : t("labs.automated.failed");
      setProgress(null);
      setError(message);
      toast.error(t("labs.automated.failed"), { description: message });
    } finally {
      inFlight.current = false;
      setRunning(false);
      setActiveJobId(null);
      setJobStatus(null);
      setCancelling(false);
    }
  };

  const cancel = async () => {
    const jobId = activeJobId;
    if (!jobId || cancelling) return;
    setCancelling(true);
    setCancelError(null);
    try {
      await cancelAutomatedTraining(jobId);
    } catch (failure) {
      // The job is still running and the poll is untouched. Hand the button
      // back with the reason, so a cancel the route refused is not silent.
      setCancelling(false);
      setCancelError(
        failure instanceof Error
          ? failure.message
          : "The cancel request failed.",
      );
    }
  };

  const cancellable = isAutomatedJobCancellable({
    status: jobStatus ?? undefined,
  });
  const presetList = presets.length > 0 ? presets : (plan?.presets ?? []);
  const summary = summarizeAutomatedProgress(progress, result);
  const visibleFailures = failuresExpanded
    ? summary.failures
    : summary.failures.slice(0, FAILURE_PREVIEW);
  const showSummary = running || progress !== null || result !== null;

  return (
    <LabsPanel
      title={t("labs.automated.title")}
      description={t("labs.automated.description")}
    >
      <div className="grid gap-3 @2xl/train-card:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <label className="flex min-w-0 flex-col gap-1.5">
          <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
            {t("labs.automated.source")}
          </span>
          <input
            className={INPUT_CLASS}
            value={source}
            onChange={(event) => {
              setSource(event.target.value);
              setPlan(null);
            }}
            placeholder={t("labs.automated.sourcePlaceholder")}
          />
          <span className="text-ui-10 text-muted-foreground/70">
            {t("labs.automated.sourceHint")}
          </span>
        </label>

        <div className="flex min-w-0 flex-col gap-1.5">
          <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
            {t("labs.automated.methods")}
          </span>
          <div className="flex flex-wrap gap-1.5">
            {AUTOMATED_METHODS.map((method) => {
              const on = methods.includes(method);
              return (
                <button
                  key={method}
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggleMethod(method)}
                  className={cn(
                    "rounded-full border px-3 py-1 text-ui-11 transition-colors",
                    on
                      ? "border-border bg-muted text-foreground"
                      : "border-border/60 text-muted-foreground hover:text-foreground",
                  )}
                >
                  {t(METHOD_LABEL[method])}
                </button>
              );
            })}
          </div>
          <span className="text-ui-10 text-muted-foreground/70">
            {t("labs.automated.methodsHint")}
          </span>
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
            {t("labs.automated.datasets")}
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setRows((current) => [...current, newRow()])}
            disabled={rows.length >= 8}
          >
            <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
            {t("labs.automated.addDataset")}
          </Button>
        </div>
        <div className="flex flex-col gap-2">
          {rows.map((row) => (
            <div
              key={row.id}
              className="grid gap-2 rounded-lg border border-border/70 bg-background/40 p-2.5 @2xl/train-card:grid-cols-[minmax(0,2fr)_minmax(0,1fr)_8rem_5rem_2rem]"
            >
              <HubPickerField
                kind="dataset"
                value={row.path}
                mode={row.mode}
                onModeChange={(mode) => updateRow(row.id, { mode })}
                onChange={(value) => updateRow(row.id, { path: value })}
                placeholder={t("labs.automated.datasetPath")}
                extensions={["jsonl", "json", "parquet", "csv", "txt"]}
              />
              <input
                className={INPUT_CLASS}
                value={row.name}
                onChange={(event) =>
                  updateRow(row.id, { name: event.target.value })
                }
                placeholder={t("labs.automated.datasetName")}
              />
              <select
                className={INPUT_CLASS}
                value={row.category}
                onChange={(event) =>
                  updateRow(row.id, {
                    category: event.target.value as AutomatedCategory,
                  })
                }
                aria-label={t("labs.automated.category")}
              >
                {AUTOMATED_CATEGORIES.map((category) => (
                  <option key={category} value={category}>
                    {t(CATEGORY_LABEL[category])}
                  </option>
                ))}
              </select>
              <label className="flex items-center gap-1">
                <input
                  className={cn(INPUT_CLASS, "w-full")}
                  inputMode="decimal"
                  value={row.evalPercent}
                  onChange={(event) =>
                    updateRow(row.id, { evalPercent: event.target.value })
                  }
                  aria-label={t("labs.automated.evalPercent")}
                />
                <span className="text-ui-10 text-muted-foreground">%</span>
              </label>
              <Button
                size="icon-sm"
                variant="ghost"
                className="justify-self-end"
                aria-label={t("labs.automated.removeDataset")}
                disabled={rows.length <= 2}
                onClick={() =>
                  setRows((current) =>
                    current.filter((item) => item.id !== row.id),
                  )
                }
              >
                <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
              </Button>
            </div>
          ))}
        </div>
        <p className="text-ui-10 text-muted-foreground/70">
          {t("labs.automated.datasetHint")}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3 @3xl/train-card:grid-cols-6">
        {(
          [
            [
              t("labs.automated.datasetsPerRound"),
              datasetsPerRound,
              setDatasetsPerRound,
              clampDatasetsPerRound,
            ],
            [
              t("labs.automated.runsPerRound"),
              runsPerRound,
              setRunsPerRound,
              clampRunsPerRound,
            ],
            [t("labs.automated.rounds"), rounds, setRounds, clampRounds],
            [
              t("labs.automated.continueRuns"),
              continueRuns,
              setContinueRuns,
              clampContinueRuns,
            ],
            [t("labs.automated.seed"), seed, setSeed, clampSeed],
            [
              t("labs.automated.stepsPerRun"),
              stepsPerRun,
              setStepsPerRun,
              clampStepsPerRun,
            ],
          ] as const
        ).map(([label, value, setter, clamp]) => (
          <label key={label} className="flex min-w-0 flex-col gap-1">
            <span className="text-ui-10 text-muted-foreground">{label}</span>
            <input
              className={INPUT_CLASS}
              inputMode="decimal"
              value={value}
              onChange={(event) => {
                setter(event.target.value);
                setPlan(null);
              }}
              onBlur={() => {
                const parsed = Number(value);
                const next = clamp(Number.isFinite(parsed) ? parsed : NaN);
                if (String(next) !== value) setter(String(next));
              }}
            />
          </label>
        ))}
      </div>

      {presetList.length > 0 ? (
        <div className="flex flex-col gap-2">
          <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
            {t("labs.automated.presets")}
          </span>
          <div className="flex flex-wrap gap-1.5">
            {presetList.map((preset) => (
              <Button
                key={preset.id}
                size="sm"
                variant="outline"
                title={preset.description}
                onClick={() => applyPreset(preset)}
              >
                {preset.label}
              </Button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => void makePlan()}
          disabled={planning || running}
        >
          {planning ? (
            <Spinner />
          ) : (
            <HugeiconsIcon icon={ArrowDown01Icon} className="size-3.5" />
          )}
          {planning ? t("labs.automated.planning") : t("labs.automated.plan")}
        </Button>
        <Button
          size="sm"
          onClick={() => void start()}
          disabled={!plan?.runnable || running || planning}
        >
          {running ? (
            <Spinner />
          ) : (
            <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
          )}
          {running ? t("labs.automated.running") : t("labs.automated.run")}
        </Button>
        {running ? (
          <Button
            size="sm"
            variant="outline"
            onClick={() => void cancel()}
            disabled={cancelling || !cancellable}
            title="Stop this automated run. Runs already finished stay as they are."
          >
            <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
            {cancelling ? "Cancelling…" : "Cancel"}
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
            {t("labs.automated.reset")}
          </Button>
        ) : null}
        <span className="text-ui-10 text-muted-foreground/70">
          {t("labs.automated.oneJob")}
        </span>
      </div>

      {cancelError ? (
        <p className="whitespace-pre-wrap text-ui-11 text-destructive">
          {cancelError}
        </p>
      ) : null}

      {showSummary ? (
        <div className="flex flex-col gap-2 rounded-lg border border-border/70 bg-muted/30 px-3 py-2">
          <div className="flex items-center gap-2 text-ui-11">
            {running ? <Spinner /> : null}
            <span className="font-medium text-foreground">
              {summary.headline}
            </span>
            {summary.total !== null && summary.total > 0 ? (
              <span className="ml-auto font-mono text-ui-10 text-muted-foreground">
                {summary.done}/{summary.total}
              </span>
            ) : null}
          </div>
          {summary.detail ? (
            <p className="text-ui-11 text-muted-foreground">{summary.detail}</p>
          ) : null}
          {summary.total !== null && summary.total > 0 ? (
            // biome-ignore lint/a11y/useFocusableInteractive: a read-only progress indicator must not add a keyboard stop
            <div
              className="h-1.5 w-full overflow-hidden rounded-full bg-border/60"
              role="progressbar"
              aria-label={summary.headline}
              aria-valuemin={0}
              aria-valuemax={summary.total}
              aria-valuenow={summary.done}
            >
              <div
                className="h-full rounded-full bg-emerald-500 transition-[width] duration-300"
                style={{
                  width: `${Math.min(100, (summary.done / summary.total) * 100)}%`,
                }}
              />
            </div>
          ) : null}
          {visibleFailures.length > 0 ? (
            <ul className="flex flex-col gap-1 text-ui-10 text-destructive">
              {visibleFailures.map((failure) => (
                <li key={failure} className="flex gap-1.5">
                  <HugeiconsIcon
                    icon={Alert02Icon}
                    className="mt-0.5 size-3 shrink-0"
                  />
                  <span className="break-words">{failure}</span>
                </li>
              ))}
            </ul>
          ) : null}
          {summary.failures.length > FAILURE_PREVIEW ? (
            <button
              type="button"
              className="self-start text-ui-10 text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
              onClick={() => setFailuresExpanded((current) => !current)}
              aria-expanded={failuresExpanded}
            >
              {failuresExpanded
                ? "Show fewer"
                : `and ${summary.failures.length - FAILURE_PREVIEW} more`}
            </button>
          ) : null}
        </div>
      ) : null}
      {error ? (
        <p className="whitespace-pre-wrap text-ui-11 text-destructive">
          {error}
        </p>
      ) : null}

      {plan ? (
        <div className="flex flex-col gap-3 rounded-xl border border-border/70 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-ui-13 font-semibold text-foreground">
              {t("labs.automated.planTitle")}
            </h3>
            <Badge
              variant="outline"
              className={cn(
                plan.runnable
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                  : "border-destructive/30 bg-destructive/10 text-destructive",
              )}
            >
              {plan.runnable
                ? t("labs.automated.runnable")
                : t("labs.automated.blocked")}
            </Badge>
          </div>

          {plan.blocked_reason ? (
            <p className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-ui-11 text-destructive">
              {plan.blocked_reason}
            </p>
          ) : null}

          <div className="grid gap-2 sm:grid-cols-3">
            {[
              [
                t("labs.automated.metricRuns"),
                plan.projection.total_train_runs.toLocaleString(),
              ],
              [
                t("labs.automated.metricDatasets"),
                String(plan.projection.datasets_per_round),
              ],
              [
                t("labs.automated.metricRounds"),
                String(plan.projection.rounds),
              ],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-lg border border-border/70 bg-background/40 px-3 py-2"
              >
                <div className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
                  {label}
                </div>
                <div className="mt-0.5 font-mono text-ui-13 font-semibold text-foreground">
                  {value}
                </div>
              </div>
            ))}
          </div>

          {plan.warnings.length > 0 ? (
            <ul className="flex flex-col gap-1 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-ui-10 text-amber-700 dark:text-amber-300">
              {plan.warnings.map((warning) => (
                <li key={warning} className="flex gap-1.5">
                  <HugeiconsIcon
                    icon={Alert02Icon}
                    className="mt-0.5 size-3 shrink-0"
                  />
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          ) : null}

          <div className="flex flex-col gap-2">
            {plan.rounds.map((round) => (
              <div
                key={round.index}
                className="overflow-hidden rounded-lg border border-border/70"
              >
                <div className="flex flex-wrap items-center gap-2 bg-muted/30 px-3 py-2">
                  <span className="text-ui-11 font-medium text-foreground">
                    {t("labs.automated.round")} {round.index}
                  </span>
                  <span className="text-ui-10 text-muted-foreground">
                    {round.datasets.join(", ")}
                  </span>
                </div>
                <table className="w-full text-ui-11">
                  <thead>
                    <tr className="text-left text-ui-10 uppercase tracking-[0.04em] text-muted-foreground">
                      <th className="px-3 py-1.5 font-medium">
                        {t("labs.automated.runIndex")}
                      </th>
                      <th className="px-3 py-1.5 font-medium">
                        {t("labs.automated.method")}
                      </th>
                      <th className="px-3 py-1.5 font-medium">
                        {t("labs.automated.datasets")}
                      </th>
                      <th className="px-3 py-1.5 text-right font-medium">
                        {t("labs.automated.steps")}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {round.runs.map((run) => (
                      <tr
                        key={`${round.index}-${run.index}`}
                        className="border-t border-border/50"
                      >
                        <td className="px-3 py-1.5 font-mono text-ui-10">
                          {run.index}
                        </td>
                        <td className="px-3 py-1.5 font-mono text-ui-10 uppercase">
                          {run.method}
                        </td>
                        <td className="max-w-[22rem] truncate px-3 py-1.5 font-mono text-ui-10">
                          {run.datasets.join(", ")}
                        </td>
                        <td className="px-3 py-1.5 text-right font-mono">
                          {run.steps.toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {result ? (
        <div className="flex flex-col gap-3 rounded-xl border border-border/70 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <HugeiconsIcon
              icon={Tick02Icon}
              className="size-4 text-emerald-600 dark:text-emerald-400"
            />
            <h3 className="text-ui-13 font-semibold text-foreground">
              {t("labs.automated.resultTitle")}
            </h3>
            <span className="truncate font-mono text-ui-10 text-muted-foreground">
              {result.best?.output_dir ?? result.output_dir ?? ""}
            </span>
            {result.best?.method ? (
              <Badge variant="outline" className="text-ui-9 uppercase">
                {result.best.method}
                {result.best.round !== undefined
                  ? ` · ${t("labs.automated.round")} ${result.best.round}`
                  : ""}
              </Badge>
            ) : null}
          </div>

          <div className="grid gap-2 sm:grid-cols-3">
            {[
              [
                t("labs.automated.bestOutput"),
                result.best?.output_dir ?? result.output_dir ?? "—",
              ],
              [t("labs.automated.bestScore"), scoreLabel(result.best?.score)],
              [
                t("labs.automated.trainRuns"),
                result.train_runs !== undefined
                  ? result.train_runs.toLocaleString()
                  : "—",
              ],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-lg border border-border/70 bg-background/40 px-3 py-2"
              >
                <div className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
                  {label}
                </div>
                <div className="mt-0.5 truncate font-mono text-ui-13 font-semibold text-foreground">
                  {value}
                </div>
              </div>
            ))}
          </div>

          {result.warnings && result.warnings.length > 0 ? (
            <ul className="flex flex-col gap-1 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-ui-10 text-amber-700 dark:text-amber-300">
              {result.warnings.map((warning) => (
                <li key={warning} className="flex gap-1.5">
                  <HugeiconsIcon
                    icon={Alert02Icon}
                    className="mt-0.5 size-3 shrink-0"
                  />
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </LabsPanel>
  );
}
