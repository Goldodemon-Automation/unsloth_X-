// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Library screen must use the shared tokens, not palette steps. These are source-text checks on
// purpose: a JSX parser in a test is a second implementation of the thing under test, and the first
// version of this file failed on correct code because of its own parser. Text is enough for "the token
// is present everywhere it should be, and no raw palette class is anywhere".

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..", "src");

const FILES = [
  "features/library/library-page.tsx",
  "features/library/components/library-list.tsx",
] as const;

function source(relative: string): string {
  return readFileSync(resolve(SRC, relative), "utf8");
}

// A raw palette class names a colour instead of a role, so it cannot follow the theme.
const RAW_PALETTE =
  /\b(?:bg|text|border|from|to|ring|fill|stroke)-(?:white|black|slate|gray|grey|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-?\d*\b/g;

test("no Library file hard-codes a palette colour", () => {
  for (const file of FILES) {
    const hits = [...source(file).matchAll(RAW_PALETTE)].map((match) => match[0]);
    assert.deepEqual(hits, [], `${file} names palette colours: ${hits.join(", ")}`);
  }
});

test("the empty state is drawn with the shared surface tokens", () => {
  const text = source(FILES[0]);
  for (const token of ["border-dashed", "border-border", "bg-muted/40", "text-muted-foreground"]) {
    assert.ok(text.includes(token), `library-page.tsx never uses ${token}`);
  }
});

test("the row list draws its divider with the border token", () => {
  assert.ok(
    source(FILES[1]).includes("border-border"),
    "library-list.tsx draws no border with the border token",
  );
});

test("every interactive surface rings its focus with the ring token", () => {
  for (const file of FILES) {
    const text = source(file);
    const rings = text.match(/focus-visible:ring-2/g) ?? [];
    const ringToken = text.match(/focus-visible:ring-ring/g) ?? [];
    assert.ok(rings.length > 0, `${file} has no focus ring at all`);
    assert.equal(
      ringToken.length,
      rings.length,
      `${file} has ${rings.length} rings but only ${ringToken.length} use the ring token`,
    );
    assert.ok(
      text.includes("outline-none"),
      `${file} draws a ring without suppressing the browser outline`,
    );
  }
});

test("the destructive action is destructive, not red", () => {
  const text = FILES.map(source).join("\n");
  assert.ok(text.includes("text-destructive"), "no destructive action uses the destructive token");
});
