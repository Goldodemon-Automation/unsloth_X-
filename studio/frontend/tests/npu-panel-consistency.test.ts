// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The NPU panel is two files written a apart from each other: `npu-setup-notice.tsx` (the
// older one, on the Hub NPU tab and the picker's NPU group) and `npu-training-card.tsx` (the
// newer one, on the training presets). They render on the same screens, so a reader sees both
// cards at once and any difference in their surface, their buttons or their live-region roles
// looks like a bug. Source-level checks, class-set based: nothing here cares what order the
// classes are written in, only that the two files say the same thing.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const NOTICE = "features/npu/npu-setup-notice.tsx";
const CARD = "features/npu/npu-training-card.tsx";

/** Tailwind utilities that make a surface, as opposed to laying content out inside it. */
const SURFACE = /^(?:rounded|border|bg)(?:[-/]|$)/;
/** The utility prefixes a raw colour can hide behind, `dark:bg-white` is still a colour. */
const PALETTE =
  /^(?:white|black|slate|gray|grey|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)(?:-\d{1,3})?(?:\/\d{1,3})?$/;

interface Attr {
  /** Whether the value came from `={...}` (an expression) rather than `="..."` (a literal). */
  expr: boolean;
  value: string;
}

interface Tag {
  name: string;
  attrs: Map<string, Attr>;
  /** Every class the tag can produce: literals, strings inside `cn(...)`, named constants. */
  classes: Set<string>;
  start: number;
}

function quoteAt(source: string, index: number): boolean {
  const ch = source[index];
  return ch === '"' || ch === "'" || ch === "`";
}

/**
 * `source` with every comment blanked, same length, so offsets still line up.
 * Comments are documentation and one of them will name the classes it forbids (`bg-white` is
 * the example the panel avoids); reading them as code would fail the file for explaining it.
 */
function withoutComments(source: string): string {
  const out = [...source];
  let index = 0;
  const blank = (from: number, to: number) => {
    for (let i = from; i < to; i += 1) out[i] = out[i] === "\n" ? "\n" : " ";
  };
  while (index < source.length) {
    if (source.startsWith("//", index)) {
      const end = source.indexOf("\n", index);
      const to = end === -1 ? source.length : end;
      blank(index, to);
      index = to;
      continue;
    }
    if (source.startsWith("/*", index)) {
      const end = source.indexOf("*/", index + 2);
      const to = end === -1 ? source.length : end + 2;
      blank(index, to);
      index = to;
      continue;
    }
    if (quoteAt(source, index)) {
      index = closeString(source, index) + 1;
      continue;
    }
    index += 1;
  }
  return out.join("");
}

/** Every string literal in the code (comments blanked first), class or not. */
function stringLiterals(source: string): string[] {
  const found: string[] = [];
  let index = 0;
  while (index < source.length) {
    if (quoteAt(source, index)) {
      const end = closeString(source, index);
      found.push(source.slice(index + 1, end));
      index = end + 1;
      continue;
    }
    index += 1;
  }
  return found;
}

/** Index of the closing quote for the string literal starting at `open`. */
function closeString(source: string, open: number): number {
  const quote = source[open];
  for (let index = open + 1; index < source.length; index += 1) {
    const ch = source[index];
    if (ch === "\\") {
      index += 1;
      continue;
    }
    if (ch === quote) return index;
  }
  return source.length;
}

/** Index of `}` closing the `{` at `open`, counting nested braces and skipping strings. */
function braceEnd(source: string, open: number): number {
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    const ch = source[index];
    if (quoteAt(source, index)) {
      index = closeString(source, index);
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return source.length;
}

/** The opening tag's own `>`: the first one outside a string or a `{...}` expression. */
function tagEnd(source: string, from: number): number {
  let depth = 0;
  for (let index = from; index < source.length; index += 1) {
    const ch = source[index];
    if (quoteAt(source, index)) {
      index = closeString(source, index);
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") depth -= 1;
    else if (ch === ">" && depth <= 0) return index;
  }
  return source.length;
}

function parseAttributes(text: string): Map<string, Attr> {
  const attrs = new Map<string, Attr>();
  let index = 0;
  while (index < text.length) {
    const ch = text[index];
    if (/\s/.test(ch) || ch === "/") {
      index += 1;
      continue;
    }
    const name = /^[A-Za-z_$][\w$:.**-]*/.exec(text.slice(index))?.[0];
    if (!name) {
      index += 1;
      continue;
    }
    index += name.length;
    if (text[index] !== "=") {
      attrs.set(name, { expr: false, value: "" });
      continue;
    }
    index += 1;
    const lead = text[index];
    if (lead === '"' || lead === "'") {
      const end = closeString(text, index);
      attrs.set(name, { expr: false, value: text.slice(index + 1, end) });
      index = end + 1;
      continue;
    }
    if (lead === "{") {
      const end = braceEnd(text, index);
      attrs.set(name, { expr: true, value: text.slice(index + 1, end) });
      index = end + 1;
      continue;
    }
    index += 1;
  }
  return attrs;
}

/** `const CARD_CLASS = "…"`: the shared literals a file names, so a tag can reference them. */
function constants(source: string): Map<string, string> {
  const consts = new Map<string, string>();
  for (const match of source.matchAll(/^const ([A-Z][A-Z0-9_]*) = (.*);$/gm)) {
    const value = match[2].trim();
    if (value.startsWith('"') && value.endsWith('"') && value.length > 1) {
      consts.set(match[1], value.slice(1, -1));
    }
  }
  return consts;
}

function classSetOf(attr: Attr | undefined, consts: Map<string, string>): Set<string> {
  const classes = new Set<string>();
  if (!attr) return classes;
  const chunks: string[] = [];
  if (attr.expr) {
    for (const match of attr.value.matchAll(/"([^"]*)"|'([^']*)'|`([^`$]*)`/g)) {
      chunks.push(match[1] ?? match[2] ?? match[3] ?? "");
    }
    for (const match of attr.value.matchAll(/\b([A-Z][A-Z0-9_]*)\b/g)) {
      const value = consts.get(match[1]);
      if (value) chunks.push(value);
    }
  } else {
    chunks.push(attr.value);
  }
  for (const chunk of chunks) {
    for (const token of chunk.split(/\s+/)) {
      if (token) classes.add(token);
    }
  }
  return classes;
}

/** Every opening tag in a component file, in source order. Comments do not count. */
function tags(source: string): Tag[] {
  const code = withoutComments(source);
  const consts = constants(code);
  const found: Tag[] = [];
  let index = 0;
  while (index < code.length) {
    if (quoteAt(code, index)) {
      index = closeString(code, index) + 1;
      continue;
    }
    const match =
      code[index] === "<" ? /^<([A-Za-z][\w.]*)/.exec(code.slice(index)) : null;
    if (!match) {
      index += 1;
      continue;
    }
    const after = index + match[0].length;
    const end = tagEnd(code, after);
    const attrs = parseAttributes(code.slice(after, end));
    found.push({
      name: match[1],
      attrs,
      classes: classSetOf(attrs.get("className"), consts),
      start: index,
    });
    // Step inside the tag: an element in an attribute expression is a tag of its own.
    index = after;
  }
  return found;
}

function base(token: string): string {
  const parts = token.split(":");
  return parts[parts.length - 1] ?? "";
}

function variantPrefixes(token: string): string[] {
  return token.split(":").slice(0, -1);
}

/** The one tag that is this file's card: the muted surface block. */
function cardTag(source: string, label: string): Tag {
  const cards = tags(source).filter((tag) => tag.classes.has("bg-muted/50"));
  assert.equal(
    cards.length,
    1,
    `${label}: expected exactly one bg-muted/50 card, found ${cards.length}`,
  );
  return cards[0];
}

function surfaceOf(tag: Tag): string[] {
  return [...tag.classes].filter((cls) => SURFACE.test(cls)).sort();
}

function buttons(source: string): Tag[] {
  return tags(source).filter((tag) => tag.name === "Button");
}

/** Variants a `variant=` prop can settle on: its literals, or absent meaning `default`. */
function variantsOf(tag: Tag): string[] {
  const attr = tag.attrs.get("variant");
  if (!attr) return ["default"];
  if (!attr.expr) return [attr.value];
  const literals = [...attr.value.matchAll(/"([^"]*)"/g)].map((match) => match[1]);
  return literals.length > 0 ? literals : ["default"];
}

/** The opening tag nearest before `text`, i.e. the element that renders that line. */
function tagRendering(source: string, text: string): Tag {
  const at = source.indexOf(text);
  assert.notEqual(at, -1, `nothing renders ${text}`);
  const before = tags(source).filter((tag) => tag.start < at);
  const last = before[before.length - 1];
  assert.ok(last, `no element before ${text}`);
  return last;
}

const notice = readSrc(NOTICE);
const card = readSrc(CARD);

test("both files describe the same card class literal", () => {
  const noticeConst = constants(notice).get("CARD_CLASS");
  const cardConst = constants(card).get("CARD_CLASS");
  assert.ok(noticeConst, `${NOTICE} has no CARD_CLASS literal`);
  assert.ok(cardConst, `${CARD} has no CARD_CLASS literal`);
  // Tolerant of order: the same classes in a different sequence still passes.
  assert.deepEqual(
    noticeConst.split(/\s+/).sort(),
    cardConst.split(/\s+/).sort(),
    "the notice and the card renamed their card surface apart from each other",
  );
});

test("the two cards have one border/background treatment", () => {
  const noticeSurface = surfaceOf(cardTag(notice, NOTICE));
  const cardSurface = surfaceOf(cardTag(card, CARD));
  assert.deepEqual(
    noticeSurface,
    cardSurface,
    "the notice's card and the training card no longer share a surface",
  );
  // Named so the pass cannot be undone by swapping a token for a colour: the card is a
  // rounded, token-bordered, token-filled surface in both themes.
  for (const cls of ["rounded-lg", "border", "border-border/60", "bg-muted/50"]) {
    assert.ok(
      noticeSurface.includes(cls) && cardSurface.includes(cls),
      `missing card surface class: ${cls}`,
    );
  }
});

test("no class assumes a light theme", () => {
  for (const [label, source] of [
    [NOTICE, notice],
    [CARD, card],
  ] as const) {
    // The blunt check first: these strings have no business in either file's classes.
    for (const literal of stringLiterals(withoutComments(source))) {
      for (const raw of ["bg-white", "text-black", "text-red-", "bg-red-"]) {
        assert.ok(
          !literal.includes(raw),
          `${label} still carries the raw palette class ${raw}`,
        );
      }
    }
    // Then the wider family: a palette colour behind any variant prefix or opacity.
    for (const tag of tags(source)) {
      for (const cls of tag.classes) {
        const colour = base(cls).replace(/^[a-z]+-/, "");
        assert.ok(
          !PALETTE.test(colour),
          `${label}: <${tag.name}> carries the palette colour class ${cls}`,
        );
      }
    }
  }
});

test("every button carries a focus-visible ring", () => {
  const panels: string[][] = [];
  for (const [label, source] of [
    [NOTICE, notice],
    [CARD, card],
  ] as const) {
    const found = buttons(source);
    assert.ok(found.length > 0, `${label} renders no button`);
    const used: string[][] = [];
    for (const tag of found) {
      const rings = [...tag.classes].filter((cls) =>
        base(cls).startsWith("ring-"),
      );
      // Order- and prefix-tolerant: any position in the class string, and either spelling
      // of the colour (`ring-ring` set once, or `focus-visible:ring-ring` on focus).
      assert.ok(
        rings.some(
          (cls) => base(cls) === "ring-2" && variantPrefixes(cls).includes("focus-visible"),
        ),
        `${label}: a button has no focus-visible ring width`,
      );
      assert.ok(
        rings.some((cls) => base(cls) === "ring-ring"),
        `${label}: a button's ring is not the ring token`,
      );
      used.push(rings.sort());
    }
    panels.push(used.flat());
  }
  // One ring for the panel, so a focus outline cannot differ between the two cards.
  const ringSet = (entries: string[]) => [...new Set(entries)].sort();
  const [noticeRings, cardRings] = panels;
  assert.deepEqual(
    ringSet(noticeRings),
    ringSet(cardRings),
    "the notice and the card ring their buttons differently",
  );
});

test("both files size their primary action the same way", () => {
  for (const [label, source] of [
    [NOTICE, notice],
    [CARD, card],
  ] as const) {
    const found = buttons(source);
    assert.ok(found.length > 0, `${label} renders no button`);
    let primaries = 0;
    for (const tag of found) {
      const size = tag.attrs.get("size")?.value;
      assert.equal(size, "sm", `${label}: a button is not the panel's sm size`);
      const variants = variantsOf(tag);
      for (const variant of variants) {
        assert.ok(
          variant === "default" || variant === "outline",
          `${label}: unexpected button variant ${variant}`,
        );
      }
      if (variants.includes("default")) primaries += 1;
    }
    assert.ok(primaries > 0, `${label}: no button offers the primary variant`);
  }
});

test("secondary lines use one muted class", () => {
  for (const [label, source] of [
    [NOTICE, notice],
    [CARD, card],
  ] as const) {
    const muted = new Set<string>();
    for (const tag of tags(source)) {
      for (const cls of tag.classes) {
        if (base(cls).startsWith("text-muted")) muted.add(base(cls));
      }
    }
    assert.deepEqual(
      [...muted],
      ["text-muted-foreground"],
      `${label}: secondary lines are muted by more than one class`,
    );
  }
});

test("status text is announced the same way in both files", () => {
  for (const [label, source] of [
    [NOTICE, notice],
    [CARD, card],
  ] as const) {
    // A failure reads as an alert, wherever it is written.
    let alerts = 0;
    for (const tag of tags(source)) {
      const destructive = [...tag.classes].some((cls) => base(cls).startsWith("text-destructive"));
      if (!destructive) continue;
      alerts += 1;
      assert.equal(
        tag.attrs.get("role")?.value,
        "alert",
        `${label}: a text-destructive <${tag.name}> is not role="alert"`,
      );
    }
    assert.ok(alerts > 0, `${label} reports no failure at all`);
    // And the container says so while work is in flight.
    const busy = tags(source).filter((tag) => tag.attrs.has("aria-busy"));
    assert.equal(busy.length, 1, `${label}: the panel's busy state is not on one container`);
  }

  // A loading line is a live region, not a static paragraph.
  const loading = tagRendering(card, "Checking your NPU");
  assert.equal(
    loading.attrs.get("role")?.value,
    "status",
    'the card\'s loading line is not role="status"',
  );
  const noticeLoading = tagRendering(notice, "Enabling downloads");
  assert.ok(
    noticeLoading.classes.has("text-muted-foreground"),
    "the notice's in-flight line is not a secondary line",
  );
});
