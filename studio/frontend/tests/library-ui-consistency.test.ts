// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Library list is the screen the device screens overtook: `library-page.tsx` and
// `components/library-list.tsx` were written before the token pass, so a row that brightened with
// `muted` beside one that brightens with `accent`, and a delete pill coloured by a red step from
// the light theme, all read as two screens pretending to be one.
//
// Source-level and class-set based: nothing here cares what order the classes are written in, only
// that the two files the screen renders from carry the tokens the device rows carry. Comments are
// blanked before reading, because a comment is allowed to name the very class it explains, and a
// text match over the whole file would count the class it forbids as a class the file uses.
//
// The reference is `features/compute-targets/compute-target-picker.tsx`: its idle rows hover with
// `hover:bg-accent/60` and ring with `focus-visible:ring-2` over the `ring` token. That treatment
// is spelled out here rather than read from the picker, so this test cannot break when a different
// screen is next repainted.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PAGE = "features/library/library-page.tsx";
const LIST = "features/library/components/library-list.tsx";

/** The device rows' hover, which a Library row now answers a pointer with. */
const DEVICE_ROW_HOVER = "hover:bg-accent/60";
/** One grey for secondary text, one colour for the action that removes files. */
const MUTED_TEXT = "text-muted-foreground";
const DESTRUCTIVE_TEXT = "text-destructive";
/** The screen's whole focus style: a width, on focus, in the ring token. */
const SHARED_RING = ["focus-visible:ring-2", "focus-visible:ring-ring"];

/** Tailwind colours, with any numeric step or alpha, behind any variant prefix. */
const PALETTE_COLOUR =
  /^(?:white|black|slate|gray|grey|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)(?:-\d{1,3})?(?:\/\d{1,3})?$/;
/** The utilities that name a colour rather than a role, and so can carry a palette step. */
const COLOURED_UTILITY =
  /^(?:bg|text|border|ring|fill|stroke|from|to|decoration|outline|shadow)-/;
const UTILITY_PREFIX = /^[a-z]+-/;
const CLASS_SEPARATOR = /\s+/;
const TAG_NAME = /^<([A-Za-z][\w.]*)/;
const ANY_WHITESPACE = /\s/;
const ATTRIBUTE_NAME = /^[A-Za-z_$][\w$:.**-]*/;
/** A file's own class list: `const ROW_HOVER = "…";`, its value written on one line. */
const CONST_LITERAL = /^const ([A-Z][A-Z0-9_]*) =\s*"([^"]*)";$/gm;
const CONST_REFERENCE = /\b([A-Z][A-Z0-9_]*)\b/g;

/** An attribute: where its value came from, and its text. */
interface Attr {
  /** Whether the value is `={...}` (an expression) rather than `="..."` (a literal). */
  expr: boolean;
  value: string;
}

interface Tag {
  name: string;
  /** Every class the tag can produce: its own literals plus the named constants it uses. */
  classes: Set<string>;
  attrs: Map<string, Attr>;
}

function isQuote(source: string, index: number): boolean {
  const ch = source[index];
  return ch === '"' || ch === "'" || ch === "`";
}

/** Index of the closing quote for the string literal starting at `open`. */
function closeString(source: string, open: number): number {
  const quote = source[open];
  for (let index = open + 1; index < source.length; index += 1) {
    if (source[index] === "\\") {
      index += 1;
      continue;
    }
    if (source[index] === quote) {
      return index;
    }
  }
  return source.length;
}

/**
 * `source` with every comment blanked and the same length, so offsets still line up.
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

/** Index of the `}` closing the `{` at `open`, counting nested braces and skipping strings. */
function braceEnd(source: string, open: number): number {
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (isQuote(source, index)) {
      index = closeString(source, index);
      continue;
    }
    const ch = source[index];
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
    if (isQuote(source, index)) {
      index = closeString(source, index);
      continue;
    }
    const ch = source[index];
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

/** `const ROW_FOCUS = "…"`: the class lists a file names, so a tag can be written with them. */
function constants(code: string): Map<string, string> {
  const consts = new Map<string, string>();
  for (const match of code.matchAll(CONST_LITERAL)) {
    consts.set(match[1], match[2]);
  }
  return consts;
}

/** The `className` a tag carries, read as an attribute rather than searched for in its text. */
function classesOf(
  attr: Attr | undefined,
  consts: Map<string, string>,
): Set<string> {
  if (attr === undefined) {
    return new Set();
  }
  // A literal is the class list; an expression is the strings it holds and the lists it names.
  const chunks = attr.expr ? stringLiterals(attr.value) : [attr.value];
  for (const match of attr.value.matchAll(CONST_REFERENCE)) {
    const value = consts.get(match[1]);
    if (value !== undefined) {
      chunks.push(value);
    }
  }
  const classes = new Set<string>();
  for (const chunk of chunks) {
    for (const token of chunk.split(CLASS_SEPARATOR)) {
      if (token !== "") {
        classes.add(token);
      }
    }
  }
  return classes;
}

/**
 * The tag's attributes, walked rather than searched: a nested element written inside `tile={...}`
 * owns its own `className`, and the outer tag must not be read as carrying it.
 */
function attributesOf(text: string): Map<string, Attr> {
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
    const text = code.slice(after, tagEnd(code, after));
    const attrs = attributesOf(text);
    found.push({
      name: match[1],
      classes: classesOf(attrs.get("className"), consts),
      attrs,
    });
    // Step inside the tag: an element written in an attribute expression is a tag of its own.
    index = after;
  }
  return found;
}

/** `focus-visible:ring-2`, `dark:bg-card`: the utility a variant sits under. */
function baseUtility(token: string): string {
  const parts = token.split(":");
  return parts[parts.length - 1] ?? "";
}

function variantPrefixes(token: string): string[] {
  return token.split(":").slice(0, -1);
}

function colourOf(token: string): string {
  return baseUtility(token).replace(UTILITY_PREFIX, "");
}

/** `text-red-600`, `dark:bg-white`, `border-neutral-300`: a colour named instead of a role. */
function isPaletteClass(token: string): boolean {
  return (
    COLOURED_UTILITY.test(baseUtility(token)) &&
    PALETTE_COLOUR.test(colourOf(token))
  );
}

function hasFocusRing(classes: Set<string>): boolean {
  const tokens = [...classes];
  return (
    tokens.some(
      (cls) =>
        baseUtility(cls) === "ring-2" &&
        variantPrefixes(cls).includes("focus-visible"),
    ) && tokens.some((cls) => baseUtility(cls) === "ring-ring")
  );
}

/** The one tag whose classes include `marker`, for a surface this screen draws once. */
function tagMarked(source: string, marker: string, label: string): Tag {
  const marked = tags(source).filter((tag) => tag.classes.has(marker));
  if (marked.length !== 1) {
    throw new Error(
      `${label}: expected exactly one tag carrying ${marker}, found ${marked.length}`,
    );
  }
  return marked[0];
}

/** Every border the tag draws is drawn with a role, never with a colour step. */
function assertTokenBorders(tag: Tag, what: string): void {
  for (const cls of tag.classes) {
    if (!baseUtility(cls).startsWith("border-")) {
      continue;
    }
    // A border that names a colour rather than a role is what this pass took out.
    if (isPaletteClass(cls)) {
      throw new Error(
        `${what} borders with ${cls}, a colour rather than a role`,
      );
    }
  }
}

/** The classes a surface must carry, and the border rule it must obey. */
function assertSurface(
  source: string,
  marker: string,
  what: string,
  expected: readonly string[],
): void {
  const tag = tagMarked(source, marker, what);
  for (const cls of expected) {
    if (!tag.classes.has(cls)) {
      throw new Error(`${what} is missing ${cls}`);
    }
  }
  assertTokenBorders(tag, what);
}

const page = readSrc(PAGE);
const list = readSrc(LIST);
const files: readonly [string, string][] = [
  [PAGE, page],
  [LIST, list],
];

test("a list row brightens the way a device row does", () => {
  const row = tagMarked(list, "group/library-row", "a list row");
  for (const cls of [
    DEVICE_ROW_HOVER,
    "transition-colors",
    "text-foreground",
  ]) {
    assert.ok(row.classes.has(cls), `the row no longer carries ${cls}`);
  }
  // The `muted` pair it replaced: `bg-muted` reads flat against a dark listing, and a
  // `dark:hover:bg-muted/60` twin is a second hover the token pass was meant to remove.
  for (const cls of row.classes) {
    assert.ok(
      !baseUtility(cls).startsWith("bg-muted"),
      `the row fills with ${cls} again`,
    );
  }
  // A selected row keeps the stronger accent fill and does not dim back to the hover under the
  // pointer, which is why the selected class list carries its own hover.
  const selected = constants(withoutComments(list)).get("ROW_SELECTED");
  assert.ok(
    selected,
    "the row's selected fill is no longer a named class list",
  );
  const selectedClasses = selected.split(CLASS_SEPARATOR);
  assert.ok(
    selectedClasses.includes("bg-accent"),
    "a selected row is not filled with the accent token",
  );
  assert.ok(
    selectedClasses.includes("hover:bg-accent"),
    "a selected row still dims to the hover fill under the pointer",
  );
  for (const cls of selectedClasses) {
    assert.ok(
      row.classes.has(cls),
      `the row never applies the selected class ${cls}`,
    );
  }
});

test("the row tile and the empty state use the shared surface classes", () => {
  // Each surface is found by a class it draws once on its file, then checked token by token.
  assertSurface(
    page,
    "bg-muted/40",
    "library-page.tsx's empty-folder drop zone",
    ["rounded-xl", "border", "border-border", "border-dashed", "bg-muted/40"],
  );
  assertSurface(list, "rounded-[10px]", "library-list.tsx's row tile", [
    "border",
    "border-border/60",
  ]);
  // Secondary lines are the muted text token in one spelling only: two greys would read as a
  // hierarchy the screen never meant to draw.
  for (const [label, source] of files) {
    const muted = new Set<string>();
    for (const tag of tags(source)) {
      for (const cls of tag.classes) {
        if (baseUtility(cls).startsWith("text-muted")) {
          muted.add(baseUtility(cls));
        }
      }
    }
    assert.deepEqual(
      [...muted],
      [MUTED_TEXT],
      `${label}: secondary lines are muted by more than one class`,
    );
  }
});

test("the selection bar's destructive action is destructive, not red", () => {
  const destructive = tags(page).filter((tag) =>
    tag.classes.has(DESTRUCTIVE_TEXT),
  );
  assert.equal(
    destructive.length,
    1,
    "more than one action on the page is destructive",
  );
  const deletePill = destructive[0];
  for (const cls of [
    DESTRUCTIVE_TEXT,
    "border",
    "border-destructive/40",
    "hover:bg-destructive/15",
  ]) {
    assert.ok(
      deletePill.classes.has(cls),
      `the delete pill no longer carries ${cls}`,
    );
  }
  // The pill keeps the bar's own ring: width and token come from the shared class list it uses.
  assert.ok(
    hasFocusRing(deletePill.classes),
    "the delete pill has no focus-visible ring",
  );
});

test("no class assumes a light theme", () => {
  for (const [label, source] of files) {
    const code = withoutComments(source);
    // The blunt pass first: no string the file holds names a palette colour, wherever it lands.
    for (const literal of stringLiterals(code)) {
      for (const token of literal.split(CLASS_SEPARATOR)) {
        assert.ok(
          !isPaletteClass(token),
          `${label} still carries the palette class ${token}`,
        );
      }
    }
    // Then the tags, so a colour reached through a `cn(...)` cannot stand in for the fix.
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

test("every hand-styled control rings its focus with the ring token", () => {
  for (const [label, source] of files) {
    // `<Button>` is excluded on purpose: that component owns its own focus treatment. The
    // controls named here are the ones this screen styles by hand, rows among them.
    const controls = tags(source).filter(
      (tag) =>
        tag.name === "button" ||
        (tag.attrs.has("tabIndex") && tag.classes.size > 0),
    );
    assert.ok(controls.length > 0, `${label} renders no hand-styled control`);
    for (const control of controls) {
      assert.ok(
        control.classes.has("outline-none"),
        `${label}: a focused <${control.name}> shows a browser outline beside its ring`,
      );
      assert.ok(
        hasFocusRing(control.classes),
        `${label}: a <${control.name}> has no focus-visible ring in the ring token`,
      );
    }
  }
});

test("the screen rings focus one way", () => {
  const perFile: string[][] = [];
  for (const [label, source] of files) {
    const rings = new Set<string>();
    for (const tag of tags(source)) {
      for (const cls of tag.classes) {
        if (baseUtility(cls).startsWith("ring-")) {
          rings.add(cls);
        }
      }
    }
    assert.ok(rings.size > 0, `${label} rings nothing at all`);
    // Only the width and the token: an offset or a per-control colour would be a second focus
    // style on one screen.
    assert.deepEqual(
      [...rings].sort(),
      [...SHARED_RING].sort(),
      `${label} rings with more than the screen's two shared classes`,
    );
    perFile.push([...rings].sort());
  }
  assert.deepEqual(
    perFile[0],
    perFile[1],
    "the page and the list ring a focused row differently",
  );
});
