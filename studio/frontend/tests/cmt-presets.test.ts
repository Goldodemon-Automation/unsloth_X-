// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import assert from "node:assert/strict";
import test from "node:test";

import type { CmtSpec } from "../src/features/labs/api";

const {
  CMT_PRESETS,
  CMT_PRESET_BASELINE,
  CMT_MAX_DATASETS,
  CMT_MIN_DATASETS,
  applyCmtPreset,
  cmtDatasetWarning,
  cmtPresetById,
  cmtPresetForValues,
  cmtPresetValues,
  recommendedCategory,
} = await import("../src/features/labs/lib/cmt-presets.ts");

const PRESET_FIELDS = new Set([
  "max_steps",
  "batch_size",
  "gradient_accumulation",
  "eval_every",
  "patience",
  "learning_rate",
]);

function baseSpec(): CmtSpec {
  return {
    source: "org/model",
    mode: "cpt",
    datasets: [
      {
        path: "",
        repo_id: "org/one",
        name: "one",
        text_column: "text",
        eval_fraction: 0.1,
      },
      {
        path: "/data/two.jsonl",
        name: "two",
        text_column: null,
        eval_fraction: 0.1,
      },
    ],
  };
}

test("presets expose the four documented ids, labels and value shapes", () => {
  assert.deepEqual(
    CMT_PRESETS.map((preset: { id: string }) => preset.id),
    ["quick-check", "balanced", "code-heavy", "reasoning-heavy"],
  );
  for (const preset of CMT_PRESETS) {
    assert.equal(typeof preset.label, "string");
    assert.ok(preset.label.length > 0);
    assert.equal(typeof preset.description, "string");
    assert.ok(preset.description.length > 0);
    for (const [key, value] of Object.entries(preset.values)) {
      assert.ok(PRESET_FIELDS.has(key), `unexpected preset field ${key}`);
      assert.equal(typeof value, "number");
      assert.ok(Number.isFinite(value));
    }
  }
});

test("preset values match the backend's suggested numbers", () => {
  assert.equal(cmtPresetById("quick-check")?.values.max_steps, 40);
  assert.equal(cmtPresetById("quick-check")?.values.eval_every, 10);
  assert.equal(cmtPresetById("quick-check")?.values.patience, 2);
  assert.equal(cmtPresetById("balanced")?.values.max_steps, 100);
  assert.equal(cmtPresetById("balanced")?.values.eval_every, 20);
  assert.equal(cmtPresetById("balanced")?.values.patience, 4);
  assert.equal(cmtPresetById("code-heavy")?.values.learning_rate, 1.5e-4);
  assert.equal(cmtPresetById("code-heavy")?.values.eval_every, 15);
  assert.equal(cmtPresetById("code-heavy")?.values.patience, 3);
  assert.equal(cmtPresetById("reasoning-heavy")?.values.max_steps, 150);
  assert.equal(cmtPresetById("reasoning-heavy")?.values.eval_every, 20);
  assert.equal(cmtPresetById("reasoning-heavy")?.values.patience, 5);
});

test("cmtPresetValues fills omitted fields from the baseline", () => {
  const quick = cmtPresetById("quick-check");
  assert.ok(quick);
  assert.deepEqual(cmtPresetValues(quick), {
    ...CMT_PRESET_BASELINE,
    max_steps: 40,
    gradient_accumulation: 4,
    eval_every: 10,
    patience: 2,
  });
  assert.equal(cmtPresetValues(quick).learning_rate, 2e-4);
});

test("applyCmtPreset merges preset values as defaults", () => {
  const merged = applyCmtPreset(baseSpec(), "quick-check");
  assert.equal(merged.max_steps, 40);
  assert.equal(merged.batch_size, 1);
  assert.equal(merged.gradient_accumulation, 4);
  assert.equal(merged.eval_every, 10);
  assert.equal(merged.patience, 2);
  assert.equal(merged.preset, "quick-check");
});

test("explicit spec fields win over the preset", () => {
  const spec: CmtSpec = { ...baseSpec(), max_steps: 999, patience: 9 };
  const merged = applyCmtPreset(spec, "balanced");
  assert.equal(merged.max_steps, 999);
  assert.equal(merged.patience, 9);
  assert.equal(merged.eval_every, 20);
  assert.equal(merged.gradient_accumulation, 8);
  assert.equal(merged.preset, "balanced");
});

test("an unknown preset leaves the spec untouched and unmarked", () => {
  const spec = baseSpec();
  const merged = applyCmtPreset(spec, "not-a-preset");
  assert.deepEqual(merged, spec);
  assert.equal("preset" in merged, false);
  assert.equal(applyCmtPreset(spec, undefined), spec);
  assert.equal(applyCmtPreset(spec, null), spec);
  assert.equal(applyCmtPreset(spec, ""), spec);
});

test("cmtPresetById returns the entry for known ids and null otherwise", () => {
  assert.equal(cmtPresetById("code-heavy")?.id, "code-heavy");
  assert.equal(cmtPresetById("code-heavy")?.label, "Code heavy");
  assert.equal(cmtPresetById("bogus"), null);
  assert.equal(cmtPresetById(undefined), null);
  assert.equal(cmtPresetById(null), null);
  assert.equal(cmtPresetById(""), null);
});

test("a merged spec keeps source, mode and dataset entries untouched", () => {
  const spec = baseSpec();
  const merged = applyCmtPreset(spec, "reasoning-heavy");
  assert.equal(merged.source, spec.source);
  assert.equal(merged.mode, spec.mode);
  assert.deepEqual(merged.datasets, spec.datasets);
  assert.equal(merged.datasets, spec.datasets);
});

test("cmtPresetForValues finds the matching preset and rejects custom values", () => {
  const balanced = cmtPresetById("balanced");
  assert.ok(balanced);
  assert.equal(cmtPresetForValues(cmtPresetValues(balanced))?.id, "balanced");
  const quick = cmtPresetById("quick-check");
  assert.ok(quick);
  assert.equal(cmtPresetForValues(cmtPresetValues(quick))?.id, "quick-check");
  assert.equal(
    cmtPresetForValues({ ...CMT_PRESET_BASELINE, max_steps: 7 }),
    null,
  );
});

test("the dataset window matches the backend's 2-8 bound", () => {
  assert.equal(CMT_MIN_DATASETS, 2);
  assert.equal(CMT_MAX_DATASETS, 8);
});

test("cmtDatasetWarning is silent for every count inside the window", () => {
  assert.equal(cmtDatasetWarning(2), null);
  assert.equal(cmtDatasetWarning(8), null);
  for (let count = CMT_MIN_DATASETS; count <= CMT_MAX_DATASETS; count += 1) {
    assert.equal(cmtDatasetWarning(count), null, `count ${count}`);
  }
});

test("cmtDatasetWarning names the minimum when too few datasets are filled", () => {
  const one = cmtDatasetWarning(1);
  assert.equal(typeof one, "string");
  assert.ok(one?.includes("at least 2"), one ?? "");
  assert.ok(one?.includes("1 "), one ?? "");
  const none = cmtDatasetWarning(0);
  assert.ok(none?.includes("at least 2"), none ?? "");
  assert.ok(none?.includes("0 "), none ?? "");
  // A NaN count is not inside the window either.
  assert.ok(cmtDatasetWarning(Number.NaN)?.includes("at least 2"));
});

test("cmtDatasetWarning names the maximum when too many datasets are filled", () => {
  const nine = cmtDatasetWarning(9);
  assert.equal(typeof nine, "string");
  assert.ok(nine?.includes("at most 8"), nine ?? "");
  assert.ok(nine?.includes("9 "), nine ?? "");
  assert.equal(cmtDatasetWarning(9), cmtDatasetWarning(9));
});

test("recommendedCategory maps the four presets to their data kind", () => {
  assert.equal(recommendedCategory("quick-check"), null);
  assert.equal(recommendedCategory("balanced"), null);
  assert.equal(recommendedCategory("code-heavy"), "code");
  assert.equal(recommendedCategory("reasoning-heavy"), "reasoning");
});

test("recommendedCategory ignores unknown, empty and missing preset ids", () => {
  assert.equal(recommendedCategory("bogus"), null);
  assert.equal(recommendedCategory(""), null);
  assert.equal(recommendedCategory(undefined), null);
  assert.equal(recommendedCategory(null), null);
  for (const preset of CMT_PRESETS) {
    const category = recommendedCategory(preset.id);
    assert.ok(
      category === null ||
        ["code", "syntax", "reasoning", "general"].includes(category),
      `unexpected category ${String(category)} for ${preset.id}`,
    );
  }
});
