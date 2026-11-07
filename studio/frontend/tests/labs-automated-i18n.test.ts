// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Labs automated training ships its own copy surface: the tab label and the
// whole `labs.automated.*` subtree. A key that only exists in en.ts falls back
// to English silently, so this test pins the nested shape of that subtree in
// every overlay instead of trusting the runtime fallback.

import assert from "node:assert/strict";
import test from "node:test";

import { ar } from "../src/i18n/locales/ar.ts";
import { de } from "../src/i18n/locales/de.ts";
import { en } from "../src/i18n/locales/en.ts";
import { es } from "../src/i18n/locales/es.ts";
import { fr } from "../src/i18n/locales/fr.ts";
import { he } from "../src/i18n/locales/he.ts";
import { hi } from "../src/i18n/locales/hi.ts";
import { it } from "../src/i18n/locales/it.ts";
import { ja } from "../src/i18n/locales/ja.ts";
import { ko } from "../src/i18n/locales/ko.ts";
import { ptBR } from "../src/i18n/locales/pt-br.ts";
import { ru } from "../src/i18n/locales/ru.ts";
import { sv } from "../src/i18n/locales/sv.ts";
import { zhCN } from "../src/i18n/locales/zh-CN.ts";

type Tree = { readonly [key: string]: string | Tree };

const overlays: Record<string, Tree> = {
  ar: ar as unknown as Tree,
  de: de as unknown as Tree,
  es: es as unknown as Tree,
  fr: fr as unknown as Tree,
  he: he as unknown as Tree,
  hi: hi as unknown as Tree,
  it: it as unknown as Tree,
  ja: ja as unknown as Tree,
  ko: ko as unknown as Tree,
  "pt-BR": ptBR as unknown as Tree,
  ru: ru as unknown as Tree,
  sv: sv as unknown as Tree,
  "zh-CN": zhCN as unknown as Tree,
};

const enTree = en as unknown as Tree;
const enAutomated = enTree.labs as Tree;

/** Every leaf path under `labs.automated`, so shape is compared, not just counts. */
function leafPaths(node: Tree, prefix: string): string[] {
  const out: string[] = [];
  for (const [key, value] of Object.entries(node)) {
    const path = `${prefix}.${key}`;
    if (typeof value === "string") {
      out.push(path);
    } else {
      out.push(...leafPaths(value, path));
    }
  }
  return out.sort();
}

function read(node: Tree, path: string): unknown {
  let cursor: unknown = node;
  for (const part of path.split(".")) {
    if (typeof cursor !== "object" || cursor === null) {
      return undefined;
    }
    cursor = (cursor as Tree)[part];
  }
  return cursor;
}

const enLeaves = leafPaths(
  (enAutomated.automated as Tree | undefined) ?? {},
  "labs.automated",
);

/** The controls a user actually sees on the automated-training page. */
const representativeKeys = [
  "labs.automated.title",
  "labs.automated.description",
  "labs.automated.source",
  "labs.automated.methods",
  "labs.automated.datasets",
  "labs.automated.datasetsPerRound",
  "labs.automated.rounds",
  "labs.automated.plan",
  "labs.automated.run",
  "labs.automated.planTitle",
  "labs.automated.metricRuns",
  "labs.automated.trainRuns",
];

function placeholders(value: string): string[] {
  return [...value.matchAll(/\{([a-zA-Z0-9_]+)\}/g)].map((m) => m[1]).sort();
}

test("en.ts still defines the automated tab and its full subtree", () => {
  assert.equal(typeof read(enTree, "labs.tab.automated"), "string");
  assert.ok(
    enLeaves.length >= 50,
    `expected the full subtree, got ${enLeaves.length}`,
  );
  for (const key of representativeKeys) {
    assert.ok(enLeaves.includes(key), `${key} is missing from en.ts`);
  }
});

test("every locale labels the automated-training tab", () => {
  for (const [locale, tree] of Object.entries(overlays)) {
    const label = read(tree, "labs.tab.automated");
    assert.equal(typeof label, "string", `${locale}: labs.tab.automated`);
    assert.ok(
      (label as string).length > 0,
      `${locale}: labs.tab.automated is empty`,
    );
  }
});

test("every locale carries the whole labs.automated subtree", () => {
  for (const [locale, tree] of Object.entries(overlays)) {
    const automated = read(tree, "labs.automated");
    assert.equal(typeof automated, "object", `${locale}: labs.automated`);
    assert.deepEqual(
      leafPaths(automated as Tree, "labs.automated"),
      enLeaves,
      `${locale}: labs.automated key set`,
    );
  }
});

test("overlay values are non-empty strings that keep en placeholders", () => {
  for (const [locale, tree] of Object.entries(overlays)) {
    for (const key of enLeaves) {
      const overlayValue = read(tree, key);
      const enValue = read(enTree, key);
      assert.equal(typeof overlayValue, "string", `${locale}: ${key}`);
      assert.ok(
        (overlayValue as string).length > 0,
        `${locale}: ${key} is empty`,
      );
      assert.deepEqual(
        placeholders(overlayValue as string),
        placeholders(enValue as string),
        `${locale}: ${key} placeholders`,
      );
    }
  }
});

test("every locale exposes the representative automated controls", () => {
  for (const [locale, tree] of Object.entries(overlays)) {
    for (const key of representativeKeys) {
      const value = read(tree, key);
      assert.equal(typeof value, "string", `${locale}: ${key}`);
      assert.ok((value as string).length > 0, `${locale}: ${key} is empty`);
    }
  }
});

// The rest of the Labs surface: the sibling tab labels, the top-level Labs
// strings and the handoff banner. They fall back to English silently too, so
// the same shape check applies.
const enLabs = enTree.labs as Tree;
const enTabLeaves = leafPaths(enLabs.tab as Tree, "labs.tab");
const enHandoffLeaves = leafPaths(enLabs.handoff as Tree, "labs.handoff");
const enTopLevelStrings = Object.entries(enLabs)
  .filter(([, value]) => typeof value === "string")
  .map(([key]) => `labs.${key}`)
  .sort();

test("every locale labels every Labs tab that en.ts declares", () => {
  assert.deepEqual(
    enTabLeaves,
    [
      "labs.tab.artifacts",
      "labs.tab.automated",
      "labs.tab.benchmarks",
      "labs.tab.cleanup",
      "labs.tab.cmt",
      "labs.tab.lineage",
      "labs.tab.pipelines",
      "labs.tab.surgery",
      "labs.tab.swap",
    ],
    "en.ts Labs tab set",
  );
  for (const [locale, tree] of Object.entries(overlays)) {
    const tab = read(tree, "labs.tab");
    assert.equal(typeof tab, "object", `${locale}: labs.tab`);
    assert.deepEqual(
      leafPaths(tab as Tree, "labs.tab"),
      enTabLeaves,
      `${locale}: labs.tab key set`,
    );
    for (const key of enTabLeaves) {
      const value = read(tree, key);
      assert.equal(typeof value, "string", `${locale}: ${key}`);
      assert.ok((value as string).length > 0, `${locale}: ${key} is empty`);
    }
  }
});

test("every locale carries the top-level Labs strings", () => {
  assert.ok(
    enTopLevelStrings.length >= 14,
    `expected the full Labs header copy, got ${enTopLevelStrings.length}`,
  );
  for (const key of [
    "labs.title",
    "labs.description",
    "labs.experimental",
    "labs.offTitle",
    "labs.offDescription",
    "labs.openSettings",
    "labs.reload",
    "labs.featureOff",
    "labs.masterOff",
    "labs.unknownFeature",
    "labs.unknownFeatureDescription",
    "labs.refresh",
    "labs.loadFailed",
  ]) {
    assert.ok(enTopLevelStrings.includes(key), `${key} is missing from en.ts`);
  }
  for (const [locale, tree] of Object.entries(overlays)) {
    for (const key of enTopLevelStrings) {
      const value = read(tree, key);
      assert.equal(typeof value, "string", `${locale}: ${key}`);
      assert.ok((value as string).length > 0, `${locale}: ${key} is empty`);
      assert.deepEqual(
        placeholders(value as string),
        placeholders(read(enTree, key) as string),
        `${locale}: ${key} placeholders`,
      );
    }
  }
});

test("every locale carries the labs.handoff subtree", () => {
  assert.deepEqual(
    enHandoffLeaves,
    ["labs.handoff.continue", "labs.handoff.scale"],
    "en.ts labs.handoff set",
  );
  for (const [locale, tree] of Object.entries(overlays)) {
    const handoff = read(tree, "labs.handoff");
    assert.equal(typeof handoff, "object", `${locale}: labs.handoff`);
    assert.deepEqual(
      leafPaths(handoff as Tree, "labs.handoff"),
      enHandoffLeaves,
      `${locale}: labs.handoff key set`,
    );
    for (const key of enHandoffLeaves) {
      const value = read(tree, key);
      assert.equal(typeof value, "string", `${locale}: ${key}`);
      assert.ok((value as string).length > 0, `${locale}: ${key} is empty`);
    }
  }
});
