// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings grew one section at a time, and the three largest of them were written before the
// device screens got their token pass. The reference is `compute-targets/compute-target-picker.tsx`:
// its rows border with `border-border`, ring focus with `focus-visible:ring-2` over the `ring`
// token, and say "secondary" with `text-muted-foreground`; and `components/ui/button.tsx`, whose
// variants colour a destructive action with the `destructive` token rather than a red step.
// These three sections now say the same thing the same way.
//
// Source-text only, on purpose. Nothing here reads the JSX tree: a previous attempt at that
// misread correct code twice. Each rule is a count or a search over the file's own characters, so
// it cannot care what order the classes were written in, only which tokens the file spells.
// Comments are blanked first, because a comment is allowed to name the very class it forbids, and
// a text match over the whole file would count the class it explains as a class the file uses.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const VOICE = "features/settings/tabs/voice-tab.tsx";
const AGENTS = "features/settings/tabs/agents-tab.tsx";
const DATA = "features/settings/tabs/data-tab.tsx";

/** The device rows' ring width, and the token it rings with: the screen owns both together. */
const RING_WIDTH = "focus-visible:ring-2";
const RING_TOKEN = "focus-visible:ring-ring";
/** One grey for secondary text, on every one of the three sections. */
const MUTED_TEXT = "text-muted-foreground";
/** The one token that colours an action which removes something. */
const DESTRUCTIVE_TEXT = "text-destructive";

/** Tailwind's colour names: a colour spelled out is a theme assumed. */
const PALETTE_COLOUR =
  "white|black|slate|gray|grey|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose";
/** Utilities that name a colour rather than a role, and so can carry a palette step. */
const COLOURED_UTILITY =
  "bg|text|border|ring|fill|stroke|from|to|via|decoration|outline|shadow|placeholder|accent|caret|divide";
/**
 * A palette class, wherever it sits: any run of variants (`dark:`, `hover:`) in front of a
 * colouring utility, a colour name, an optional step and an optional alpha. Bounded on both
 * sides by characters that cannot occur inside a class token, so `text-status-success` and
 * `bg-muted/40` never read as colour names.
 */
const PALETTE_CLASS = new RegExp(
  String.raw`(?:^|[^\w./[-])((?:[\w-]+:)*(?:${COLOURED_UTILITY})-(?:${PALETTE_COLOUR})(?:-\d{1,3})?(?:\/\d{1,3}|#[0-9a-fA-F]{2})?)(?=$|[^\w./-])`,
  "g",
);
/** The `border-border` class, alone or at an alpha, as one whole class token. */
const BORDER_TOKEN =
  /(?:^|[\s"'`])border-border(?:\/\d{1,3})?(?=$|[\s"'`])/g;

/**
 * `source` with every comment blanked and its length unchanged, so offsets still line up.
 *
 * A `//` only opens a comment when nothing but whitespace precedes it on its line, which keeps a
 * protocol in a string literal intact. Quotes are not tracked: a section's class lists are the
 * only thing this reads, and an apostrophe in prose is exactly what must not shift the scan.
 */
function withoutComments(source: string): string {
  const out = [...source];
  const blank = (from: number, to: number): void => {
    for (let index = from; index < to; index += 1) {
      if (out[index] !== "\n") out[index] = " ";
    }
  };
  let index = 0;
  while (index < source.length) {
    if (source.startsWith("/*", index)) {
      const end = source.indexOf("*/", index + 2);
      const to = end === -1 ? source.length : end + 2;
      blank(index, to);
      index = to;
      continue;
    }
    if (source.startsWith("//", index)) {
      const lead = index === 0 ? "\n" : source[index - 1];
      if (lead === " " || lead === "\t" || lead === "\n" || lead === "\r") {
        const end = source.indexOf("\n", index);
        const to = end === -1 ? source.length : end;
        blank(index, to);
        index = to;
        continue;
      }
    }
    index += 1;
  }
  return out.join("");
}

/** Every class list the file spells: the contents of each `"…"` and ` ` `…` ` ` ` string. */
function classLists(code: string): string[] {
  const found: string[] = [];
  let index = 0;
  while (index < code.length) {
    const quote = code[index];
    if (quote !== '"' && quote !== "`") {
      index += 1;
      continue;
    }
    let end = index + 1;
    while (end < code.length && code[end] !== quote) {
      // An interpolation is part of the class list this reads, so a `{` is not a close.
      if (code[end] === "\\") end += 1;
      end += 1;
    }
    found.push(code.slice(index + 1, end));
    index = end + 1;
  }
  return found;
}

function countOf(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

const files: readonly [string, string][] = [VOICE, AGENTS, DATA].map((path) => [
  path,
  withoutComments(readSrc(path)),
]);

test("no class in these three sections names a colour instead of a role", () => {
  for (const [label, code] of files) {
    const offenders = [...code.matchAll(PALETTE_CLASS)].map((match) => match[1]);
    assert.deepEqual(
      offenders,
      [],
      `${label} still carries a palette class: ${offenders.join(", ")}. A step assumes a light theme; ` +
        `the role tokens are bg-status-success, text-status-warning, text-destructive and the border/muted tokens.`,
    );
  }
});

test("each section borders with the shared border token", () => {
  for (const [label, code] of files) {
    const borders = [...code.matchAll(BORDER_TOKEN)];
    assert.ok(
      borders.length > 0,
      `${label} draws no edge with border-border: the device rows' border is that token, at most faded with an alpha`,
    );
  }
});

test("a focus ring is spelled once per control: its width beside its token", () => {
  // The width and the token arrive as a pair, on the device rows and in every section here. A
  // count that differs is a control ringed with a width nobody chose, or a colour of its own.
  for (const [label, code] of files) {
    const widths = countOf(code, RING_WIDTH);
    const tokens = countOf(code, RING_TOKEN);
    assert.ok(widths > 0, `${label} rings nothing at all`);
    assert.equal(
      widths,
      tokens,
      `${label}: ${widths} occurrences of ${RING_WIDTH} against ${tokens} of ${RING_TOKEN}`,
    );
  }
});

test("a ringed control hides the outline it would show beside its ring", () => {
  for (const [label, code] of files) {
    for (const list of classLists(code)) {
      if (!list.includes(RING_WIDTH)) continue;
      assert.ok(
        list.includes("outline-none"),
        `${label}: a class list rings ${RING_WIDTH} but leaves the browser outline standing`,
      );
    }
  }
});

test("secondary text is one grey on all three sections", () => {
  for (const [label, code] of files) {
    assert.ok(
      countOf(code, MUTED_TEXT) > 0,
      `${label} has no secondary line in the muted foreground token`,
    );
    // A second spelling of the same grey would read as a hierarchy the screen never drew.
    for (const offender of [
      "text-muted/",
      "text-sidebar-foreground",
      "text-panel-surface-fg-muted",
    ]) {
      assert.ok(
        !code.includes(offender),
        `${label} mutes secondary text with ${offender} as well as ${MUTED_TEXT}`,
      );
    }
  }
});

test("a destructive action is destructive, not red", () => {
  const total = files.reduce(
    (seen, [, code]) => seen + countOf(code, DESTRUCTIVE_TEXT),
    0,
  );
  assert.ok(
    total > 0,
    `none of the three sections colours a destructive action with ${DESTRUCTIVE_TEXT}; ` +
      `button.tsx does it with that token, and the picker's blocked rows with text-muted-foreground`,
  );
});
