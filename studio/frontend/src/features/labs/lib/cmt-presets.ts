// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import type { CmtSpec } from "../api";

/**
 * The tuning fields a CMT preset can supply. Every preset value stays inside
 * the bounds the backend normaliser enforces, so a preset can never make a
 * request invalid on its own.
 */
export interface CmtPresetValues {
  max_steps: number;
  batch_size: number;
  gradient_accumulation: number;
  eval_every: number;
  patience: number;
  learning_rate: number;
}

/** The panel's own starting point for every preset field. */
export const CMT_PRESET_BASELINE: CmtPresetValues = {
  max_steps: 100,
  batch_size: 1,
  gradient_accumulation: 8,
  eval_every: 20,
  patience: 4,
  learning_rate: 2e-4,
};

export interface CmtPreset {
  id: string;
  label: string;
  description: string;
  values: Partial<CmtPresetValues>;
}

/**
 * A spec plus the preset marker the backend understands. The marker rides in
 * the existing spec object, so the plan and run endpoints stay untouched.
 */
export type CmtSpecWithPreset = CmtSpec & { preset?: string };

/**
 * Four ready-made starting points. A preset only supplies defaults for fields
 * the caller leaves unset; every explicit spec field still wins, matching the
 * backend's `PRESETS` table in core/labs/cmt.py.
 */
export const CMT_PRESETS: CmtPreset[] = [
  {
    id: "quick-check",
    label: "Quick check",
    description:
      "A short smoke run: 40 steps, effective batch 4, checks every 10 steps. Use it to prove the datasets, checkpoint and pipeline work before a full run.",
    values: {
      max_steps: 40,
      batch_size: 1,
      gradient_accumulation: 4,
      eval_every: 10,
      patience: 2,
    },
  },
  {
    id: "balanced",
    label: "Balanced",
    description:
      "The everyday mix: 100 steps, effective batch 8, checks every 20 steps with moderate patience. A sensible default for general text.",
    values: {
      max_steps: 100,
      batch_size: 1,
      gradient_accumulation: 8,
      eval_every: 20,
      patience: 4,
    },
  },
  {
    id: "code-heavy",
    label: "Code heavy",
    description:
      "Tuned for code corpora: a higher learning rate, checks every 15 steps and tighter patience. Supply your own code datasets; CMT never downloads any.",
    values: {
      max_steps: 100,
      batch_size: 1,
      gradient_accumulation: 8,
      learning_rate: 1.5e-4,
      eval_every: 15,
      patience: 3,
    },
  },
  {
    id: "reasoning-heavy",
    label: "Reasoning heavy",
    description:
      "A longer run: 150 steps, a higher learning rate and more patience, for reasoning-style data that needs time to show a held-out gain.",
    values: {
      max_steps: 150,
      batch_size: 1,
      gradient_accumulation: 8,
      learning_rate: 2e-4,
      eval_every: 20,
      patience: 5,
    },
  },
];

/**
 * The dataset count window the backend enforces in core/labs/cmt.py. A spec
 * outside it is rejected before any planning work starts, so the panel blocks
 * the actions instead of letting the request fail.
 */
export const CMT_MIN_DATASETS = 2;
export const CMT_MAX_DATASETS = 8;

/**
 * A sentence naming the dataset bound a count breaks, or null when the count is
 * inside the window. A blank row is not a dataset, so callers pass the number
 * of rows that carry a path.
 */
export function cmtDatasetWarning(count: number): string | null {
  if (!Number.isFinite(count)) {
    return `CMT needs at least ${CMT_MIN_DATASETS} datasets.`;
  }
  if (count < CMT_MIN_DATASETS) {
    const verb = count === 1 ? "is" : "are";
    return `CMT needs at least ${CMT_MIN_DATASETS} datasets; ${count} ${verb} filled in.`;
  }
  if (count > CMT_MAX_DATASETS) {
    return `CMT takes at most ${CMT_MAX_DATASETS} datasets; ${count} are filled in.`;
  }
  return null;
}

/** The dataset categories a preset can point at, as the panel names them. */
export type CmtCategory = "code" | "syntax" | "reasoning" | "general";

/**
 * The category a preset's numbers are tuned for, as a hint only: the panel
 * never writes it anywhere. Presets that suit any text return null, so the
 * panel shows no hint rather than a misleading one.
 */
export function recommendedCategory(
  presetId: string | null | undefined,
): CmtCategory | null {
  switch (presetId) {
    case "code-heavy":
      return "code";
    case "reasoning-heavy":
      return "reasoning";
    default:
      return null;
  }
}

/** Look up a preset by id; unknown, empty or missing ids return null. */
export function cmtPresetById(id: string | null | undefined): CmtPreset | null {
  if (!id) return null;
  return CMT_PRESETS.find((preset) => preset.id === id) ?? null;
}

/** Fill the fields a preset omits from the panel baseline. */
export function cmtPresetValues(preset: CmtPreset): CmtPresetValues {
  return { ...CMT_PRESET_BASELINE, ...preset.values };
}

/**
 * Merge a preset into a spec: the preset supplies defaults for fields the spec
 * leaves undefined, while explicit spec fields win. An unknown id returns the
 * spec unchanged, with no marker attached.
 */
export function applyCmtPreset(
  spec: CmtSpec,
  presetId: string | null | undefined,
): CmtSpecWithPreset {
  const preset = cmtPresetById(presetId);
  if (!preset) return spec;
  const merged: CmtSpecWithPreset = { ...spec };
  const tuning = merged as Partial<CmtPresetValues>;
  for (const key of Object.keys(preset.values) as Array<
    keyof CmtPresetValues
  >) {
    const value = preset.values[key];
    // Explicit spec fields win, so only fill what the caller left unset.
    if (value !== undefined && tuning[key] === undefined) {
      tuning[key] = value;
    }
  }
  merged.preset = preset.id;
  return merged;
}

/**
 * The preset whose effective values match the running values exactly, or null
 * when the user has edited their way to a custom set. Fields a preset omits are
 * compared against the baseline, so editing any tuning field drops the marker.
 */
export function cmtPresetForValues(values: CmtPresetValues): CmtPreset | null {
  for (const preset of CMT_PRESETS) {
    const full = cmtPresetValues(preset);
    const matches = (Object.keys(full) as Array<keyof CmtPresetValues>).every(
      (key) => values[key] === full[key],
    );
    if (matches) return preset;
  }
  return null;
}
