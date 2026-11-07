// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const BASE = "/api/labs";

export type LabsFeatureStatus = "alpha" | "beta";

export interface LabsFeature {
  id: string;
  label: string;
  description: string;
  status: LabsFeatureStatus;
  studio: boolean;
  enabled: boolean;
}

export interface LabsState {
  enabled: boolean;
  features: LabsFeature[];
  note: string;
}

export interface ArtifactAction {
  id: "continue-training" | "scale" | "merge" | "quantize" | "benchmark";
  enabled: boolean;
  reason: string | null;
}

export interface ArtifactManifest {
  stage?: string | null;
  run_id?: string | null;
  created_at?: string | null;
  studio_version?: string | null;
}

export interface Artifact {
  path: string;
  name: string;
  modified: string;
  model_type: string | null;
  params: number | null;
  weight_bytes: number;
  adapter: boolean;
  quantized: boolean;
  moe: boolean;
  supported: boolean;
  manifest: ArtifactManifest | null;
  manifest_path: string | null;
  actions: ArtifactAction[];
  raw_manifest?: unknown;
  manifest_error?: string;
}

export interface ArtifactList {
  artifacts: Artifact[];
  root: string;
}

export type PipelineKind =
  | "merge"
  | "scale"
  | "train"
  | "quantize"
  | "benchmark";

export interface PipelineStageSpec {
  kind: PipelineKind;
  factor?: number;
  method?: string;
  steps?: number;
  batch_size?: number;
  sequence_length?: number;
  dataset?: string;
  level?: string;
  baseline?: string;
  suites?: string[];
  parents?: string[];
}

export interface PipelineSpec {
  name?: string;
  source?: string;
  stages: PipelineStageSpec[];
}

export interface PipelineStagePlan {
  index: number;
  kind: PipelineKind;
  label: string;
  params_before: number | null;
  params_after: number | null;
  output_bytes: number;
  warnings: string[];
  blocked: boolean;
  reason: string | null;
  details: Record<string, unknown>;
}

export interface PipelinePlan {
  source: string | null;
  stages: PipelineStagePlan[];
  final_params: number | null;
  total_output_bytes: number;
  total_output_label: string;
  peak_stage_bytes: number;
  peak_stage_label: string;
  warnings: string[];
  blocked: boolean;
  blocked_reason: string | null;
  notes: string[];
}

export interface SavedPipeline {
  id: string;
  name: string;
  created_at: string;
  spec: PipelineSpec;
  plan: {
    final_params: number | null;
    total_output_bytes: number;
    blocked: boolean;
    blocked_reason: string | null;
    stage_kinds: PipelineKind[];
  };
}

async function readJson<T>(response: Response, what: string): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new Error(`The ${what} response was not JSON.`);
  }
}

function jsonRequest(body: unknown, method: "POST" | "PUT"): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export async function loadLabsState(signal?: AbortSignal): Promise<LabsState> {
  const response = await authFetch(BASE, { method: "GET", signal });
  return readJson<LabsState>(response, "Labs");
}

export async function saveLabsState(payload: {
  enabled?: boolean;
  features?: Record<string, boolean>;
}): Promise<LabsState> {
  const response = await authFetch(BASE, jsonRequest(payload, "PUT"));
  return readJson<LabsState>(response, "Labs");
}

export async function loadArtifacts(
  limit = 50,
  signal?: AbortSignal,
): Promise<ArtifactList> {
  const response = await authFetch(`${BASE}/artifacts?limit=${limit}`, {
    method: "GET",
    signal,
  });
  return readJson<ArtifactList>(response, "Artifacts");
}

export async function loadArtifactDetail(
  path: string,
  signal?: AbortSignal,
): Promise<Artifact> {
  const query = encodeURIComponent(path);
  const response = await authFetch(`${BASE}/artifacts/detail?path=${query}`, {
    method: "GET",
    signal,
  });
  return readJson<Artifact>(response, "Artifact");
}

export interface ArtifactDiffFact {
  field: string;
  label: string;
  left: unknown;
  right: unknown;
  changed: boolean;
}

export interface ArtifactDiffChange {
  field: string;
  group: string;
  left: unknown;
  right: unknown;
}

export interface ArtifactDiffSide {
  path: string;
  name: string;
  modified: string;
  has_manifest: boolean;
}

export type PinnedStatus =
  | "pinned"
  | "regressed"
  | "improved"
  | "mixed"
  | "flat"
  | "unscored"
  | "missing"
  | "unpinned";

export interface PinnedVerdict {
  status: PinnedStatus;
  pinned_record_id?: string;
  pinned_at?: string | null;
  aggregate_delta?: number | null;
  tolerance?: number | null;
  suites?: Array<{
    suite: string;
    pinned: number;
    candidate: number;
    delta: number;
    tolerance: number;
    verdict: "regressed" | "improved" | "flat";
  }>;
  regressions?: string[];
  improved?: string[];
  flat?: string[];
}

export interface ArtifactDiffBenchmarkSide {
  id: string;
  created_at: string | null;
  suites: string[];
  candidate_scores: Record<string, number>;
  baseline_scores: Record<string, number>;
  verdict: string | null;
  vs_pinned: PinnedVerdict;
}

export interface ArtifactDiff {
  left: ArtifactDiffSide;
  right: ArtifactDiffSide;
  facts: ArtifactDiffFact[];
  changes: ArtifactDiffChange[];
  changed_count: number;
  same_count: number;
  truncated: boolean;
  benchmark: {
    left: ArtifactDiffBenchmarkSide | null;
    right: ArtifactDiffBenchmarkSide | null;
    delta: Array<{
      suite: string;
      left: number | null;
      right: number | null;
      delta: number | null;
    }> | null;
    aggregate_delta: number | null;
  };
  notes: string[];
}

export async function loadArtifactDiff(
  left: string,
  right: string,
  signal?: AbortSignal,
): Promise<ArtifactDiff> {
  const query = new URLSearchParams({ left, right });
  const response = await authFetch(`${BASE}/artifacts/diff?${query}`, {
    method: "GET",
    signal,
  });
  return readJson<ArtifactDiff>(response, "diff");
}

export async function planPipeline(
  spec: PipelineSpec,
  signal?: AbortSignal,
): Promise<PipelinePlan> {
  const response = await authFetch(`${BASE}/pipelines/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<PipelinePlan>(response, "pipeline plan");
}

export async function listPipelines(
  signal?: AbortSignal,
): Promise<SavedPipeline[]> {
  const response = await authFetch(`${BASE}/pipelines`, {
    method: "GET",
    signal,
  });
  const body = await readJson<{ pipelines: SavedPipeline[] }>(
    response,
    "pipelines",
  );
  return body.pipelines;
}

export async function savePipeline(spec: PipelineSpec): Promise<SavedPipeline> {
  const response = await authFetch(
    `${BASE}/pipelines`,
    jsonRequest({ spec }, "POST"),
  );
  const body = await readJson<{ pipeline: SavedPipeline }>(
    response,
    "saved pipeline",
  );
  return body.pipeline;
}

export async function deletePipeline(id: string): Promise<void> {
  const response = await authFetch(
    `${BASE}/pipelines/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
}

export interface LineageNode {
  id: string;
  label: string;
  path: string;
  stage: string;
  created_at: string;
  studio_version: string | null;
  valid: boolean;
  problems: string[];
}

export interface LineageEdge {
  from: string;
  to: string;
  label: string;
}

export interface LineageReport {
  root: string;
  nodes: LineageNode[];
  edges: LineageEdge[];
  roots: string[];
  truncated: boolean;
}

export async function loadLineage(
  signal?: AbortSignal,
): Promise<LineageReport> {
  const response = await authFetch(`${BASE}/lineage`, {
    method: "GET",
    signal,
  });
  return readJson<LineageReport>(response, "lineage");
}

export interface BenchmarkRecord {
  id: string;
  job_id: string;
  artifact: string;
  artifact_name: string;
  baseline: string;
  baseline_name: string;
  suites: string[];
  limit: number | null;
  batch_size: number;
  created_at: string;
  duration_seconds: number | null;
  candidate_scores: Record<string, number>;
  baseline_scores: Record<string, number>;
  pin?: {
    record_id: string;
    pinned_at: string | null;
    is_self: boolean;
  } | null;
  vs_pinned?: PinnedVerdict;
  verdict: string;
  comparison: {
    entries: Array<{
      suite?: string;
      baseline?: number;
      candidate?: number;
      delta?: number;
    }>;
    aggregate_delta: number | null;
    verdict: string | null;
    regressions: string[];
    improved: string[];
    flat: string[];
  };
  errors: string[];
  warnings: string[];
  report_dir: string;
}

export interface BenchmarkRunBody {
  artifact: string;
  baseline: string;
  suites: string[];
  limit?: number | null;
  batch_size?: number;
  timeout_seconds?: number;
}

export interface LabsJob {
  job_id: string;
  kind: string;
  status: "running" | "completed" | "failed";
  output_dir?: string | null;
  progress?: {
    stage?: string;
    done?: number;
    total?: number;
    message?: string;
    updated_at?: number;
  } | null;
  result: Record<string, unknown> | null;
  error: string | null;
}

export async function loadBenchmarks(
  limit = 50,
  artifact?: string,
  signal?: AbortSignal,
): Promise<BenchmarkRecord[]> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (artifact) query.set("artifact", artifact);
  const response = await authFetch(`${BASE}/benchmarks?${query}`, {
    method: "GET",
    signal,
  });
  const body = await readJson<{ records: BenchmarkRecord[] }>(
    response,
    "benchmarks",
  );
  return body.records;
}

export async function runBenchmark(body: BenchmarkRunBody): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/benchmarks/run`, {
    ...jsonRequest(body, "POST"),
  });
  return readJson<LabsJob>(response, "benchmark run");
}

export async function deleteBenchmark(id: string): Promise<void> {
  const response = await authFetch(
    `${BASE}/benchmarks/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
}

export async function pinBenchmark(id: string): Promise<void> {
  const response = await authFetch(
    `${BASE}/benchmarks/${encodeURIComponent(id)}/pin`,
    { method: "POST" },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
}

export async function unpinBenchmark(id: string): Promise<void> {
  const response = await authFetch(
    `${BASE}/benchmarks/${encodeURIComponent(id)}/pin`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
}

export interface HousekeepingManifest {
  stage?: string | null;
  run_id?: string | null;
  created_at?: string | null;
}

export interface HousekeepingEntry {
  path: string;
  name: string;
  root: "outputs" | "exports" | "hf-hub" | "hf-datasets";
  kind: string;
  bytes: number;
  bytes_label: string;
  file_count: number;
  modified: string;
  manifest: HousekeepingManifest | null;
  deletable: boolean;
  protected_reason: string | null;
}

export interface HousekeepingRoot {
  id: string;
  label: string;
  note: string;
  path: string;
  exists: boolean;
  entry_count: number;
  bytes: number;
}

export interface HousekeepingReport {
  roots: HousekeepingRoot[];
  entries: HousekeepingEntry[];
  total_bytes: number;
  total_label: string;
  reclaimable_bytes: number;
  reclaimable_label: string;
  free_bytes: number;
  free_label: string;
  disk_total_bytes: number;
  protected: Array<{ path: string; reason: string }>;
  notes: string[];
}

export interface HousekeepingDeleteResult {
  deleted: Array<{
    path: string;
    bytes: number;
    bytes_label: string;
    file_count: number;
  }>;
  refused: Array<{ path: string; reason: string }>;
  freed_bytes: number;
  freed_label: string;
}

export async function scanHousekeeping(
  signal?: AbortSignal,
): Promise<HousekeepingReport> {
  const response = await authFetch(`${BASE}/housekeeping`, {
    method: "GET",
    signal,
  });
  return readJson<HousekeepingReport>(response, "housekeeping");
}

export async function deleteHousekeeping(
  paths: string[],
): Promise<HousekeepingDeleteResult> {
  const response = await authFetch(`${BASE}/housekeeping/delete`, {
    ...jsonRequest({ paths, confirm: true }, "POST"),
  });
  return readJson<HousekeepingDeleteResult>(response, "delete");
}

export interface JanitorRules {
  enabled: boolean;
  keep_newest: number;
  min_age_days: number;
  roots: string[];
}

export interface JanitorCandidate {
  path: string;
  name: string;
  root: string;
  kind: string;
  bytes: number;
  bytes_label: string;
  modified: string;
}

export interface JanitorPreview {
  rules: JanitorRules;
  candidates: JanitorCandidate[];
  kept: JanitorCandidate[];
  protected: Array<{ path: string; reason: string }>;
  total_bytes: number;
  total_label: string;
  notes: string[];
}

export interface JanitorApplyResult {
  rules: JanitorRules;
  preview: JanitorPreview;
  result: HousekeepingDeleteResult;
}

export async function loadJanitorRules(
  signal?: AbortSignal,
): Promise<JanitorRules> {
  const response = await authFetch(`${BASE}/housekeeping/rules`, {
    method: "GET",
    signal,
  });
  const body = await readJson<{ rules: JanitorRules }>(
    response,
    "janitor rules",
  );
  return body.rules;
}

export async function saveJanitorRules(
  rules: Partial<JanitorRules>,
): Promise<JanitorRules> {
  const response = await authFetch(`${BASE}/housekeeping/rules`, {
    ...jsonRequest(rules, "PUT"),
  });
  const body = await readJson<{ rules: JanitorRules }>(
    response,
    "janitor rules",
  );
  return body.rules;
}

export async function previewJanitorRules(
  rules: Partial<JanitorRules>,
): Promise<JanitorPreview> {
  const response = await authFetch(`${BASE}/housekeeping/rules/preview`, {
    ...jsonRequest(rules, "POST"),
  });
  return readJson<JanitorPreview>(response, "rules preview");
}

export async function applyJanitorRules(
  rules: Partial<JanitorRules>,
): Promise<JanitorApplyResult> {
  const response = await authFetch(`${BASE}/housekeeping/rules/apply`, {
    ...jsonRequest(rules, "POST"),
  });
  return readJson<JanitorApplyResult>(response, "rules apply");
}

export interface CmtDataset {
  path: string;
  repo_id?: string | null;
  /**
   * Whether the plan actually measured this dataset. False for a Hub dataset,
   * whose rows and tokens stay placeholders until the run downloads it. Never
   * infer this from a zero: an empty local file measures as zero rows too.
   */
  measured: boolean;
  name: string;
  text_column: string | null;
  eval_fraction: number;
  bytes: number;
  rows: number;
  eval_rows: number;
  train_rows: number;
  tokens_estimate: number;
  notes: string[];
}

export interface CmtPlan {
  source: string;
  mode: "cpt" | "finetune";
  added_source: "tokens" | "override";
  datasets: CmtDataset[];
  totals: {
    datasets: number;
    /** The figures below are subtotals over the measured datasets only. */
    measured_datasets: number;
    unmeasured_datasets: number;
    rows: number;
    train_rows: number;
    eval_rows: number;
    tokens_estimate: number;
    train_tokens_estimate: number;
    train_tokens_label: string;
  };
  sizing: {
    rule: string;
    added_params: number;
    added_label: string;
    suggested_from_tokens: number;
    suggested_from_rows: number;
    min_added_params: number;
    max_added_params: number;
    note: string;
  };
  expansion: {
    base_params: number;
    base_label: string;
    target_params: number;
    target_label: string;
    added_params: number;
    added_label: string;
    growth_factor: number;
    params_per_layer: number;
    new_layers: number;
    floor_bypassed: boolean;
    growth_floor: number | null;
    supported: boolean;
    notes: string[];
  };
  training: {
    mode: string;
    method: string;
    max_steps: number;
    suggested_steps: number;
    batch_size: number;
    gradient_accumulation: number;
    max_seq_length: number;
    learning_rate: number;
    eval_every: number;
    patience: number;
    seed: number;
    train_tokens_per_step: number;
    note: string;
  };
  dependencies: {
    available: boolean;
    missing: string[];
    install_hint: string;
  };
  warnings: string[];
  blocked: boolean;
  blocked_reason: string | null;
  runnable: boolean;
  notes: string[];
}

export interface CmtDatasetSpec {
  /** A local file. Empty when the entry names a Hub dataset instead. */
  path: string;
  repo_id?: string;
  subset?: string | null;
  split?: string | null;
  name?: string;
  text_column?: string | null;
  eval_fraction?: number;
}

export interface CmtSpec {
  source: string;
  datasets: CmtDatasetSpec[];
  mode: "cpt" | "finetune";
  added_params?: number | null;
  max_steps?: number;
  batch_size?: number;
  gradient_accumulation?: number;
  max_seq_length?: number;
  learning_rate?: number;
  eval_every?: number;
  patience?: number;
  seed?: number;
}

export async function planCmt(
  spec: CmtSpec,
  signal?: AbortSignal,
): Promise<CmtPlan> {
  const response = await authFetch(`${BASE}/cmt/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<CmtPlan>(response, "CMT plan");
}

export async function runCmt(spec: CmtSpec): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/cmt/run`, {
    ...jsonRequest({ spec }, "POST"),
  });
  return readJson<LabsJob>(response, "CMT run");
}

export interface AutoScaleSpec {
  /** The checkpoint to grow: a local folder or a Hub id. */
  model: string;
  /** Local dataset files, or Hub ids the picker committed, the growth is sized from. */
  datasets?: Array<{ path?: string; repo_id?: string }>;
  method?: "depth" | "width";
  insert_strategy?: string;
  /** Blank lets the sizing rule decide; otherwise 1M-700M. */
  added_params?: number | null;
  rows?: number;
  tokens?: number;
}

export interface AutoScalePlan {
  model: string;
  method: string;
  sizing: {
    rule: string;
    rows: number;
    tokens: number;
    added_params: number;
    added_label: string;
    source: string;
    note: string;
  };
  datasets: Array<{
    path?: string;
    repo_id?: string;
    name: string;
    rows: number;
    bytes: number;
    tokens_estimate: number;
  }>;
  expansion: {
    base_params: number;
    base_label: string;
    added_params: number;
    added_label: string;
    target_params: number;
    target_label: string;
    growth_factor: number;
    params_per_layer: number;
    new_layers: number;
    floor_bypassed: boolean;
  };
  warnings: string[];
  blocked: boolean;
  blocked_reason: string | null;
  runnable: boolean;
  notes: string[];
}

export interface AutoScaleResult {
  output_dir: string;
  checkpoint: string;
  base_params: number;
  added_params: number;
  final_params: number;
  growth_factor: number;
  new_layers: number;
  warnings?: string[];
}

export async function planAutoScale(
  spec: AutoScaleSpec,
  signal?: AbortSignal,
): Promise<AutoScalePlan> {
  const response = await authFetch(`${BASE}/auto-scale/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<AutoScalePlan>(response, "scale plan");
}

export async function runAutoScale(spec: AutoScaleSpec): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/auto-scale/run`, {
    ...jsonRequest({ spec }, "POST"),
  });
  return readJson<LabsJob>(response, "scale run");
}

export interface SurgerySpec {
  /** A local checkpoint folder: surgery rewrites weights on disk. */
  model: string;
  drop_layers?: number;
  layer_strategy?: "last" | "uniform";
  /** Query heads to remove. Whole KV groups go, so the real count can be higher. */
  drop_heads?: number;
  ffn_keep?: number;
  experts?: number;
  top_k?: number;
}

export interface SurgeryPlan {
  model: string;
  architecture: string;
  moe_architecture: string | null;
  runnable: boolean;
  blocked_reason: string | null;
  base: {
    layers: number;
    layer_indices: number[];
    hidden_size: number;
    num_heads: number;
    num_kv_heads: number;
    head_dim: number;
    ffn_width: number;
    params: number;
    params_label: string;
    bytes: number;
    bytes_label: string;
  };
  prune: {
    layers: {
      drop: number;
      keep: number;
      strategy: string;
      dropped_indices: number[];
    };
    heads: {
      requested: number;
      removed: number;
      keep: number;
      kv_keep: number;
      group_size: number;
    };
    ffn: { keep_ratio: number; keep: number; drop: number };
  };
  moe: {
    experts: number;
    top_k: number;
    architecture: string;
    router: string;
    ffn_width: number;
  } | null;
  projection: {
    params: number;
    params_label: string;
    bytes: number;
    bytes_label: string;
    ratio: number;
  };
  warnings: string[];
  notes: string[];
}

export interface SurgeryResult {
  output_dir: string;
  checkpoint: string;
  manifest_path: string;
  params: number;
  params_label: string;
  weight_bytes: number;
  written_tensors: number;
  architecture: string;
  warnings?: string[];
}

export async function planSurgery(
  spec: SurgerySpec,
  signal?: AbortSignal,
): Promise<SurgeryPlan> {
  const response = await authFetch(`${BASE}/surgery/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<SurgeryPlan>(response, "surgery plan");
}

export async function runSurgery(spec: SurgerySpec): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/surgery/run`, {
    ...jsonRequest({ spec }, "POST"),
  });
  return readJson<LabsJob>(response, "surgery run");
}

export interface ArchSwapSpec {
  source: string;
  target: string;
  extra_layers?: "repeat" | "zeros";
}

export interface ArchSwapMapping {
  target: string;
  role: string;
  layer: number | null;
  from: string[];
  action: string;
  note: string;
}

export interface ArchSwapPlan {
  source: string;
  target: string;
  source_family: string;
  target_family: string;
  source_layers: number;
  target_layers: number;
  extra_layers: string;
  runnable: boolean;
  blocked_reason: string | null;
  mapping: ArchSwapMapping[];
  mapping_truncated: boolean;
  roles: Array<{
    role: string;
    label: string;
    tensors: number;
    action: string;
  }>;
  counts: Record<string, number>;
  warnings: string[];
  notes: string[];
}

export interface ArchSwapResult {
  output_dir: string;
  checkpoint: string;
  manifest_path: string;
  source_family: string;
  target_family: string;
  tensors: number;
  zeroed_tensors: number;
  weight_bytes: number;
  warnings?: string[];
}

export async function planArchSwap(
  spec: ArchSwapSpec,
  signal?: AbortSignal,
): Promise<ArchSwapPlan> {
  const response = await authFetch(`${BASE}/arch-swap/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<ArchSwapPlan>(response, "arch swap plan");
}

export async function runArchSwap(spec: ArchSwapSpec): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/arch-swap/run`, {
    ...jsonRequest({ spec }, "POST"),
  });
  return readJson<LabsJob>(response, "arch swap run");
}

export type AutomatedTrainingCategory =
  | "code"
  | "syntax"
  | "reasoning"
  | "general";

export type AutomatedTrainingMethod = "cmt" | "cpt" | "fft";

export interface AutomatedTrainingDatasetSpec {
  path?: string;
  repo_id?: string;
  name?: string;
  category?: AutomatedTrainingCategory;
  eval_fraction?: number;
}

export interface AutomatedTrainingSpec {
  source: string;
  datasets: AutomatedTrainingDatasetSpec[];
  methods?: AutomatedTrainingMethod[];
  datasets_per_round?: number;
  runs_per_round?: number;
  rounds?: number;
  continue_runs?: number;
  seed?: number;
  steps_per_run?: number;
  preset?: string;
  benchmark?: { suites?: string[]; limit?: number; batch_size?: number };
  training_kwargs?: Record<string, unknown>;
}

export interface AutomatedTrainingRunStep {
  index: number;
  method: AutomatedTrainingMethod;
  datasets: string[];
  steps: number;
}

export interface AutomatedTrainingRound {
  index: number;
  datasets: string[];
  runs: AutomatedTrainingRunStep[];
  continue_runs: number;
}

export interface AutomatedPreset {
  id: string;
  label: string;
  description: string;
  spec: Partial<AutomatedTrainingSpec>;
}

export interface AutomatedTrainingPlan {
  runnable: boolean;
  blocked_reason?: string | null;
  rounds: AutomatedTrainingRound[];
  warnings: string[];
  projection: {
    total_train_runs: number;
    datasets_per_round: number;
    rounds: number;
  };
  presets: AutomatedPreset[];
}

export async function listAutomatedTrainingPresets(
  signal?: AbortSignal,
): Promise<{ presets: AutomatedPreset[] }> {
  const response = await authFetch(`${BASE}/automated/presets`, {
    method: "GET",
    signal,
  });
  return readJson<{ presets: AutomatedPreset[] }>(
    response,
    "automated presets",
  );
}

export async function planAutomatedTraining(
  spec: AutomatedTrainingSpec,
  signal?: AbortSignal,
): Promise<AutomatedTrainingPlan> {
  const response = await authFetch(`${BASE}/automated/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<AutomatedTrainingPlan>(response, "automated plan");
}

export async function runAutomatedTraining(
  spec: AutomatedTrainingSpec,
  outputDir?: string,
): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/automated/run`, {
    ...jsonRequest({ spec, output_dir: outputDir }, "POST"),
  });
  return readJson<LabsJob>(response, "automated run");
}

/**
 * What the cancel route answers: the job id it accepted and nothing else. The
 * job keeps its own status, so the page waits for the poll to see it finish
 * instead of pretending the training already stopped.
 */
export interface AutomatedTrainingCancel {
  job_id: string;
  cancelling: true;
}

/**
 * Ask the engine to stop a running automated job. A job that already finished
 * is unknown to the route and answers 404, so the caller reads the error rather
 * than assume the cancel landed.
 */
export async function cancelAutomatedTraining(
  jobId: string,
  signal?: AbortSignal,
): Promise<AutomatedTrainingCancel> {
  const response = await authFetch(
    `${BASE}/automated/jobs/${encodeURIComponent(jobId)}/cancel`,
    { ...jsonRequest({}, "POST"), signal },
  );
  return readJson<AutomatedTrainingCancel>(response, "automated cancel");
}
