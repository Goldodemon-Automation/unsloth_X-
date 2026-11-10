// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The NPU panel is two files written apart from each other: `npu-setup-notice.tsx` (the older
// one, on the Hub's NPU tab and the picker's NPU group) and `npu-training-card.tsx` (the newer
// one, on the training presets). They render on the same screens, so a reader sees both cards at
// once, and any difference in their surface, their buttons or their live-region roles reads as a
// bug in the panel.
//
// Source-level and class-set based: nothing here cares what order the classes are written in,
// only that the two files say the same thing. Comments are stripped before reading, because a
// comment is allowed to name the very class it explains.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const NOTICE = "features/npu/npu-setup-notice.tsx";
const CARD = "features/npu/npu-training-card.tsx";

/** The raw palette a card must not assume: it pins the surface to one theme. */
const RAW_PALETTE = ["bg-white", "text-black", "text-red-", "bg-red-"];
/** Tailwind colours, with any numeric step or alpha, behind any variant prefix. */
const PALETTE_COLOUR =
  /^(?:white|black|slate|gray|grey|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)(?:-\d{1,3})?(?:\/\d{1,3})?$/;

const TAG_NAME = /^<([A-Za-z][\w.]*)/;
const ATTRIBUTE_NAME = /^[A-Za-z_$][\w$:.**-]*/;
const CONST_LITERAL = /^const ([A-Z][A-Z0-9_]*) = (.*);$/gm;
const CLASS_TOKENS = /"([^"]*)"|'([^']*)'|`([^`$]*)`/g;
const CONST_REFERENCE = /\b([A-Z][A-Z0-9_]*)\b/g;
const STRING_LITERAL = /"([^"]*)"/g;
const CLASS_SEPARATOR = /\s+/;
const ANY_WHITESPACE = /\s/;
const UTILITY_PREFIX = /^[a-z]+-/;
/** The utilities that make a surface, as opposed to laying content out inside it. */
const SURFACE = /^(?:rounded|border|bg)(?:[-/]|$)/;

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

function isQuote(source: string, index: number): boolean {
  const ch = source[index];
  return ch === '"' || ch === "'" || ch === "`";
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
    if (ch === quote) {
      return index;
    }
  }
  return source.length;
}

/**
 * `source` with every comment blanked and the same length, so offsets still line up. A comment
 * naming `bg-white` to say why it is wrong is documentation, not a class in the panel.
 */
function withoutComments(source: string): string {
  const out = [...source];
  const blank = (from: number, to: number): void => {
    for (let index = from; index < to; index += 1) {
      out[index] = out[index] === "\n" ? "\n" : " ";
    }
  };
  let index = 0;
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
    if (isQuote(source, index)) {
      index = closeString(source, index) + 1;
      continue;
    }
    index += 1;
  }
  return out.join("");
}

/** Every string literal in code, comments already blanked. */
function stringLiterals(code: string): string[] {
  const found: string[] = [];
  let index = 0;
  while (index < code.length) {
    if (isQuote(code, index)) {
      const end = closeString(code, index);
      found.push(code.slice(index + 1, end));
      index = end + 1;
      continue;
    }
    index += 1;
  }
  return found;
}

/** Index of `}` closing the `{` at `open`, counting nested braces and skipping strings. */
function braceEnd(source: string, open: number): number {
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    const ch = source[index];
    if (isQuote(source, index)) {
      index = closeString(source, index);
      continue;
    }
    if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        return index;
      }
    }
  }
  return source.length;
}

/** The opening tag's own `>`: the first one outside a string or a `{...}` expression. */
function tagEnd(source: string, from: number): number {
  let depth = 0;
  for (let index = from; index < source.length; index += 1) {
    const ch = source[index];
    if (isQuote(source, index)) {
      index = closeString(source, index);
      continue;
    }
    if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
    } else if (ch === ">" && depth <= 0) {
      return index;
    }
  }
  return source.length;
}

function parseAttributes(text: string): Map<string, Attr> {
  const attrs = new Map<string, Attr>();
  let index = 0;
  while (index < text.length) {
    const ch = text[index];
    if (ANY_WHITESPACE.test(ch) || ch === "/") {
      index += 1;
      continue;
    }
    const name = ATTRIBUTE_NAME.exec(text.slice(index))?.[0];
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

/** `const CARD_CLASS = "…"`: the literals a file names, so a tag can be written with them. */
function constants(code: string): Map<string, string> {
  const consts = new Map<string, string>();
  for (const match of code.matchAll(CONST_LITERAL)) {
    const value = match[2].trim();
    if (value.length > 1 && value.startsWith('"') && value.endsWith('"')) {
      consts.set(match[1], value.slice(1, -1));
    }
  }
  return consts;
}

/** What a `className={...}` expression can produce: its own literals and the constants it names. */
function chunksOfExpression(
  expression: string,
  consts: Map<string, string>,
): string[] {
  const chunks: string[] = [];
  for (const match of expression.matchAll(CLASS_TOKENS)) {
    chunks.push(match[1] ?? match[2] ?? match[3] ?? "");
  }
  for (const match of expression.matchAll(CONST_REFERENCE)) {
    const value = consts.get(match[1]);
    if (value !== undefined) {
      chunks.push(value);
    }
  }
  return chunks;
}

function classSetOf(
  attr: Attr | undefined,
  consts: Map<string, string>,
): Set<string> {
  const classes = new Set<string>();
  if (attr === undefined) {
    return classes;
  }
  const chunks = attr.expr
    ? chunksOfExpression(attr.value, consts)
    : [attr.value];
  for (const chunk of chunks) {
    for (const token of chunk.split(CLASS_SEPARATOR)) {
      if (token !== "") {
        classes.add(token);
      }
    }
  }
  return classes;
}

/** Every opening tag in a component file, in source order, comments ignored. */
function tags(source: string): Tag[] {
  const code = withoutComments(source);
  const consts = constants(code);
  const found: Tag[] = [];
  let index = 0;
  while (index < code.length) {
    if (isQuote(code, index)) {
      index = closeString(code, index) + 1;
      continue;
    }
    const match = code[index] === "<" ? TAG_NAME.exec(code.slice(index)) : null;
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

/** `bg:foo`, `focus-visible:ring-2`, `text-red-500/40` read as their utility and their variants. */
function baseUtility(token: string): string {
  const parts = token.split(":");
  return parts[parts.length - 1] ?? "";
}

function variantPrefixes(token: string): string[] {
  return token.split(":").slice(0, -1);
}

function isPaletteClass(token: string): boolean {
  const colour = baseUtility(token).replace(UTILITY_PREFIX, "");
  return PALETTE_COLOUR.test(colour);
}

/** The one tag that is this file's card: the muted surface block. */
function cardTag(source: string, label: string): Tag {
  const cards = tags(source).filter((tag) => tag.classes.has("bg-muted/50"));
  if (cards.length !== 1) {
    throw new Error(
      `${label}: expected exactly one bg-muted/50 card, found ${cards.length}`,
    );
  }
  return cards[0];
}

/** Rounded, border and background classes only: the card's treatment, not its layout. */
function surfaceOf(tag: Tag): string[] {
  return [...tag.classes].filter((cls) => SURFACE.test(cls)).sort();
}

function buttons(source: string): Tag[] {
  return tags(source).filter((tag) => tag.name === "Button");
}

/** Variants a `variant=` prop can settle on: its literals, or absent meaning `default`. */
function variantsOf(tag: Tag): string[] {
  const attr = tag.attrs.get("variant");
  if (attr === undefined) {
    return ["default"];
  }
  if (!attr.expr) {
    return [attr.value];
  }
  const literals = [...attr.value.matchAll(STRING_LITERAL)].map(
    (match) => match[1],
  );
  return literals.length > 0 ? literals : ["default"];
}

/** The opening tag nearest before `text`, i.e. the element that renders that line. */
function tagRendering(source: string, text: string): Tag {
  const code = withoutComments(source);
  const at = code.indexOf(text);
  if (at === -1) {
    throw new Error(`nothing renders ${text}`);
  }
  const before = tags(source).filter((tag) => tag.start < at);
  const last = before[before.length - 1];
  if (!last) {
    throw new Error(`no element before ${text}`);
  }
  return last;
}

const notice = readSrc(NOTICE);
const card = readSrc(CARD);
const panel = [
  [NOTICE, notice],
  [CARD, card],
] as const;

test("both files describe the same card class literal", () => {
  const noticeConst = constants(withoutComments(notice)).get("CARD_CLASS");
  const cardConst = constants(withoutComments(card)).get("CARD_CLASS");
  assert.ok(noticeConst, `${NOTICE} has no CARD_CLASS literal`);
  assert.ok(cardConst, `${CARD} has no CARD_CLASS literal`);
  // Tolerant of order: the same classes written in a different sequence still pass.
  assert.deepEqual(
    noticeConst.split(CLASS_SEPARATOR).sort(),
    cardConst.split(CLASS_SEPARATOR).sort(),
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
  // Named rather than left to the diff, so the pass cannot be undone by trading a token for a
  // colour: the card is a rounded, token-bordered, token-filled surface in both themes.
  for (const cls of [
    "rounded-lg",
    "border",
    "border-border/60",
    "bg-muted/50",
  ]) {
    assert.ok(
      noticeSurface.includes(cls) && cardSurface.includes(cls),
      `missing card surface class: ${cls}`,
    );
  }
});

test("no class assumes a light theme", () => {
  for (const [label, source] of panel) {
    const code = withoutComments(source);
    // The blunt check first: these four have no business in any class the file can produce.
    for (const literal of stringLiterals(code)) {
      for (const raw of RAW_PALETTE) {
        assert.ok(
          !literal.includes(raw),
          `${label} still carries the raw palette class ${raw}`,
        );
      }
    }
    // Then the wider family, so `bg-red-500` or a `dark:bg-white` cannot stand in for the fix.
    for (const tag of tags(source)) {
      for (const cls of tag.classes) {
        assert.ok(
          !isPaletteClass(cls),
          `${label}: <${tag.name}> carries the palette colour class ${cls}`,
        );
      }
    }
  }
});

test("every button carries a focus-visible ring", () => {
  const ringsPerFile: string[][] = [];
  for (const [label, source] of panel) {
    const found = buttons(source);
    assert.ok(found.length > 0, `${label} renders no button`);
    const rings: string[] = [];
    for (const tag of found) {
      const classes = [...tag.classes].filter((cls) =>
        baseUtility(cls).startsWith("ring-"),
      );
      // Order- and prefix-tolerant: wherever they sit in the string, and either spelling of the
      // colour (`ring-ring` set once, or `focus-visible:ring-ring` on focus).
      assert.ok(
        classes.some(
          (cls) =>
            baseUtility(cls) === "ring-2" &&
            variantPrefixes(cls).includes("focus-visible"),
        ),
        `${label}: a button has no focus-visible ring width`,
      );
      assert.ok(
        classes.some((cls) => baseUtility(cls) === "ring-ring"),
        `${label}: a button's ring is not the ring token`,
      );
      rings.push(...classes);
    }
    ringsPerFile.push([...new Set(rings)].sort());
  }
  // One ring for the panel: a focus outline must not read differently on the two cards.
  const [noticeRings, cardRings] = ringsPerFile;
  assert.deepEqual(
    noticeRings,
    cardRings,
    "the notice and the card ring their buttons differently",
  );
});

test("both files size their primary action the same way", () => {
  for (const [label, source] of panel) {
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
      if (variants.includes("default")) {
        primaries += 1;
      }
    }
    assert.ok(
      primaries > 0,
      `${label}: no button offers the primary default variant`,
    );
  }
});

test("secondary lines use one muted class", () => {
  for (const [label, source] of panel) {
    const muted = new Set<string>();
    for (const tag of tags(source)) {
      for (const cls of tag.classes) {
        if (baseUtility(cls).startsWith("text-muted")) {
          muted.add(baseUtility(cls));
        }
      }
    }
    // One class, no `/70` and no `/80`: a panel with two greys reads as two hierarchies it
    // never meant to draw.
    assert.deepEqual(
      [...muted],
      ["text-muted-foreground"],
      `${label}: secondary lines are muted by more than one class`,
    );
  }
});

test("status text is announced the same way in both files", () => {
  for (const [label, source] of panel) {
    // A failure reads as an alert wherever it is written.
    let alerts = 0;
    for (const tag of tags(source)) {
      const destructive = [...tag.classes].some((cls) =>
        baseUtility(cls).startsWith("text-destructive"),
      );
      if (!destructive) {
        continue;
      }
      alerts += 1;
      assert.equal(
        tag.attrs.get("role")?.value,
        "alert",
        `${label}: a text-destructive <${tag.name}> is not role="alert"`,
      );
    }
    assert.ok(alerts > 0, `${label} reports no failure at all`);
    // And the container says so while its own work is in flight.
    const busy = tags(source).filter((tag) => tag.attrs.has("aria-busy"));
    assert.equal(
      busy.length,
      1,
      `${label}: the panel's busy state is not on one container`,
    );
  }

  // A loading line is a live region, not a static paragraph.
  const loading = tagRendering(card, "Checking your NPU");
  assert.equal(
    loading.attrs.get("role")?.value,
    "status",
    "the card's loading line is not role=status",
  );
  // The notice has no loading sentence of its own, only the in-flight card; it must mute that
  // line the way the card mutes its own, and leave the announcement to the spinner it renders.
  const inFlight = tagRendering(notice, "Enabling downloads");
  assert.ok(
    inFlight.classes.has("text-muted-foreground"),
    "the notice's in-flight line is not a secondary line",
  );
});
