// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The canvas/artifact routes answer an `Authorization` header or a short-lived `link=` capability,
// and refuse a session bearer in the query string: `_authenticate_header_or_query` in
// `routes/inference.py` returns 400 for one, because a bearer in a URL outlives the click in shell
// history, in a reverse proxy's access log and in the `Referer` of whatever the page links to.
//
// Source text, not a render: the question is how a URL is spelled, and a JSX tree answers it worse
// than the file does. Nothing here imports `src` -- those modules resolve `@/` aliases that plain
// `node --experimental-strip-types` does not know.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";

const SRC = new URL("../src/", import.meta.url);

const read = (relative: string) => readFileSync(new URL(relative, SRC), "utf8");

/** A query string carrying a session bearer: `?token=` or `&token=`. */
const QUERY_TOKEN = /[?&]token=/;
/** The shape these routes do take, and the word a caller has to say to be read as knowing it. */
const QUERY_LINK = /link=/;
const SCRIPT_FILE = /\.tsx?$/;
const DOWNLOAD_BUILDER = /export function libraryDownloadPath\([\s\S]*?\n}/;
const DOWNLOAD_WRAPPER =
  /export async function libraryDownloadUrl\([\s\S]*?\n}/;
const ANY_TOKEN = /token/i;

const CANVAS_CARD = "components/assistant-ui/sandbox-files-view.tsx";
const LIBRARY_API = "features/library/api.ts";

/** Every `.ts`/`.tsx` under `src`, as paths relative to it. */
function sources(): string[] {
  const found: string[] = [];
  const walk = (relative: string) => {
    for (const entry of readdirSync(new URL(relative, SRC), {
      withFileTypes: true,
    })) {
      const path = `${relative}${entry.name}`;
      if (entry.isDirectory()) {
        walk(`${path}/`);
      } else if (SCRIPT_FILE.test(entry.name)) {
        found.push(path);
      }
    }
  };
  walk("./");
  return found;
}

test("no frontend file builds a URL with a session token in the query string", () => {
  const files = sources();
  // Sanity: the walk really did find the tree, so an empty result means "no offenders".
  assert.ok(files.length > 500, `only ${files.length} files walked`);
  const offenders = files.filter((file) => QUERY_TOKEN.test(read(file)));
  assert.deepEqual(
    offenders,
    [],
    `these build a ?token= URL: ${offenders.join(", ")}`,
  );
});

test("the canvas and library callers name the query the routes take", () => {
  for (const file of [CANVAS_CARD, LIBRARY_API]) {
    assert.match(
      read(file),
      QUERY_LINK,
      `${file} never says what the route accepts`,
    );
  }
});

test("the canvas file card authorizes with a header, not a URL", () => {
  const card = read(CANVAS_CARD);
  // The save goes through the same credentialed fetch the context menu's load uses, so the token is
  // an `Authorization` header the fetch adds and refreshes -- never a value inside the URL.
  assert.match(card, /loadSandboxFile\(sessionId, file\.name\)/);
  // The old shape read the raw bearer out and pasted it into the URL.
  assert.doesNotMatch(card, /getAuthToken/);
  assert.doesNotMatch(card, /downloadUrlStreaming/);
});

test("the library download builder takes a link and never a token", () => {
  const source = read(LIBRARY_API);
  const builder = DOWNLOAD_BUILDER.exec(source)?.[0];
  assert.ok(
    builder,
    "libraryDownloadPath is gone: the download URL has no one builder",
  );
  assert.match(builder, /new URLSearchParams\(\{ id: item\.id \}\)/);
  assert.match(builder, /query\.set\("link", link\)/);
  assert.doesNotMatch(builder, ANY_TOKEN);

  // The async wrapper asks for the URL through the builder and appends nothing of its own.
  const wrapper = DOWNLOAD_WRAPPER.exec(source)?.[0];
  assert.ok(
    wrapper,
    "libraryDownloadUrl is gone: the native save has nothing to call",
  );
  assert.match(wrapper, /libraryDownloadPath\(item\)/);
  assert.doesNotMatch(wrapper, ANY_TOKEN);
});
