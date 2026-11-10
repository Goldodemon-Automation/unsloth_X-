// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings grew one section at a time, and its three largest sections were written before the
// device screens got their token pass. The reference is `compute-targets/compute-target-picker.tsx`:
// its rows border with `border-border`, ring focus with `focus-visible:ring-2` over the `ring`
// token, and say "secondary" with `text-muted-foreground`; and `components/ui/button.tsx`, which
// colours a destructive action with the `destructive` token rather than with a step of red.
// These three sections now say the same thing the same way.
//
// Source-text only, on purpose: nothing here reads the JSX tree, so no rule can misread a class
// list for a control it is not on. Each rule is a count or a search over the file's own characters,
// which is why none of them cares what order the classes were written in, only which tokens the
// file spells. Comments are blanked before the scan, because a comment is allowed to name the very
// class it explains, and a match over the whole file would count the class it forbids as a class
// the file uses.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const VOICE = "features/settings/tabs/voice-tab.tsx";
const AGENTS = "features/settings/tabs/agents-tab.tsx";
const DATA = "features/settings/tabs/data-tab.tsx";

/** The device rows' ring width, and the token it rings with: a screen owns both together. */
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
 * A palette class wherever it sits: any run of variants (`dark:`, `hover:`) in front of a
 * colouring utility, a colour name, an optional step, and an optional alpha. Bounded on both
 * sides by characters that cannot occur inside a class token, so `bg-status-success`, `bg-muted/40`
 * and `text-destructive-foreground` never read as a colour spelled out.
 */
const PALETTE_CLASS = new RegExp(
  String.raw`(?:^|[^\w./[-])((?:[\w-]+:)*(?:${COLOURED_UTILITY})-(?:${PALETTE_COLOUR})(?:-\d{1,3})?(?:\/\d{1,3}|#[0-9a-fA-F]{2})?)(?=$|[^\w./-])`,
  "g",
);
/** The shared border class, whole or faded by an alpha, as one class token. */
const BORDER_TOKEN = /(?:^|[\s"'`])border-border(?:\/\d{1,3})?(?=$|[\s"'`])/;

/** Any text utility that mutes, an alpha included: each has to be spelled exactly as the token. */
const MUTED_SPELLING = /text-muted[\w-]*(?:\/\d{1,3})?/g;

const PALETTE_FIX =
  "A step assumes one theme; the roles here are the status, border, muted and destructive tokens.";
const DESTRUCTIVE_REASON =
  "button.tsx colours one with the destructive token, and the picker's blocked rows with the muted one.";

/**
 * `source` with every comment blanked and its length unchanged, so offsets still line up.
 *
 * A `//` opens a comment only when nothing but whitespace precedes it on its line, which keeps a
 * protocol inside a string literal intact. Quotes are not tracked: a section's class lists are all
 * this reads, and an apostrophe in prose is exactly what must not shift the scan.
 */
function withoutComments(source: string): string {
  const out = [...source];
  const blank = (from: number, to: number): void => {
    for (let index = from; index < to; index += 1) {
      if (out[index] !== "\n") {
        out[index] = " ";
      }
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
    const lead = index === 0 ? "\n" : source[index - 1];
    const atLineStart =
      lead === " " || lead === "\t" || lead === "\n" || lead === "\r";
    if (atLineStart && source.startsWith("//", index)) {
      const end = source.indexOf("\n", index);
      const to = end === -1 ? source.length : end;
      blank(index, to);
      index = to;
      continue;
    }
    index += 1;
  }
  return out.join("");
}

/** Every class list the file spells: the inside of each `"…"` and of each backtick string. */
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
      // An interpolation is part of the class list, so a `{` is not a close, and an escaped
      // quote is not one either.
      if (code[end] === "\\") {
        end += 1;
      }
      end += 1;
    }
    found.push(code.slice(index + 1, end));
    index = end + 1;
  }
  return found;
}

/** How many times the fixed string `needle` occurs in `haystack`. */
function countOf(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

const files: readonly [string, string][] = [VOICE, AGENTS, DATA].map((path) => [
  path,
  withoutComments(readSrc(path)),
]);

test("no class in these three sections names a colour instead of a role", () => {
  for (const [label, code] of files) {
    const offenders = [...code.matchAll(PALETTE_CLASS)].map(
      (match) => match[1],
    );
    assert.deepEqual(
      offenders,
      [],
      `${label} still carries ${offenders.join(", ")}. ${PALETTE_FIX}`,
    );
  }
});

test("each section borders with the shared border token", () => {
  for (const [label, code] of files) {
    // The device rows' edge is this token, at most faded by an alpha. A control with no edge at
    // all still counts: the rule is that a border drawn here is drawn with the token.
    assert.ok(
      BORDER_TOKEN.test(code),
      `${label} draws an edge with something other than border-border`,
    );
  }
});

test("a focus ring is spelled once per control: its width beside its token", () => {
  // Width and token arrive as a pair, on the device rows and in every section here. Counts that
  // differ mean a control ringed with a width nobody chose, or a ring in a colour of its own.
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
      if (!list.includes(RING_WIDTH)) {
        continue;
      }
      assert.ok(
        list.includes("outline-none"),
        `${label}: a class list rings ${RING_WIDTH} and leaves the browser outline standing`,
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
    // A second spelling of the same grey — the token on its own, or the token at an alpha — reads
    // as a hierarchy the screen never drew.
    const spellings = new Set(
      [...code.matchAll(MUTED_SPELLING)].map((match) => match[0]),
    );
    assert.deepEqual(
      [...spellings],
      [MUTED_TEXT],
      `${label} mutes secondary text with more than one class: ${[...spellings].join(", ")}`,
    );
  }
});

test("a destructive action is destructive, not red", () => {
  const total = files.reduce(
    (seen, [, code]) => seen + countOf(code, DESTRUCTIVE_TEXT),
    0,
  );
  assert.ok(
    total > 0,
    `none of the three sections colours a destructive action with ${DESTRUCTIVE_TEXT}. ${DESTRUCTIVE_REASON}`,
  );
});
