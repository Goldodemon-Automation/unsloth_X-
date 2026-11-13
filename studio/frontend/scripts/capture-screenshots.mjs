// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Screenshot harness for the README: drives headless Edge/Chrome over CDP with no dependencies
// (Node's global WebSocket), signs in by writing the app's own localStorage token, and captures
// the real built UI. The in-app preview browser's automation channel was unresponsive on this
// host, so this is the fallback path; it takes the same pictures from the same served app.
//
// Usage: node scripts/capture-screenshots.mjs --base http://127.0.0.1:5001 --token <jwt> \
//          --out docs/screenshots --port 9222

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (value.startsWith("--")) pairs.push([value.slice(2), all[index + 1]]);
    return pairs;
  }, []),
);

const BASE = args.base ?? "http://127.0.0.1:5001";
const TOKEN = args.token;
const OUT = args.out ?? "docs/screenshots";
const PORT = args.port ?? "9222";
const VIEWPORT = { width: 1440, height: 900 };
// --signed-out captures the pre-authentication screen from a clean profile: an authenticated tab
// that navigates to /login is redirected to /chat, so the sign-in screen needs its own browser.
const SIGNED_OUT = process.argv.includes("--signed-out");

if (!TOKEN && !SIGNED_OUT) {
  console.error("--token is required (or pass --signed-out)");
  process.exit(2);
}
mkdirSync(OUT, { recursive: true });

const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = targets.find((target) => target.type === "page");
if (!page) {
  console.error("no page target on the debugging port");
  process.exit(2);
}

const socket = new WebSocket(page.webSocketDebuggerUrl);
let nextId = 1;
const pending = new Map();

socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(JSON.stringify(message.error)));
    else resolve(message.result);
  }
});

await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

function send(method, params = {}) {
  const id = nextId++;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    setTimeout(() => {
      if (pending.delete(id)) reject(new Error(`${method} timed out`));
    }, 60000);
  });
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", {
  ...VIEWPORT,
  deviceScaleFactor: 1,
  mobile: false,
});

async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text ?? "evaluate failed");
  }
  return result.result.value;
}

async function goto(path) {
  await send("Page.navigate", { url: `${BASE}${path}` });
  await sleep(1200);
}

async function shoot(name, { settleMs = 2500, scrollToTop = true } = {}) {
  if (scrollToTop) {
    await evaluate("(() => { document.scrollingElement?.scrollTo(0, 0); return true; })()");
  }
  await sleep(settleMs);
  const { data } = await send("Page.captureScreenshot", { format: "png" });
  const path = join(OUT, `${name}.png`);
  writeFileSync(path, Buffer.from(data, "base64"));
  console.log(`captured ${path}`);
}

if (SIGNED_OUT) {
  await goto("/login");
  await evaluate('(() => { localStorage.setItem("theme", "dark"); return true; })()');
  await send("Page.navigate", { url: `${BASE}/login` });
  await shoot("06-login-dark", { settleMs: 4000 });
  socket.close();
  console.log("done");
  process.exit(0);
}

// Sign in through the app's own storage contract, then let the shell boot from it.
await goto("/login");
await evaluate(
  `(() => {
     localStorage.setItem("unsloth_auth_token", ${JSON.stringify(TOKEN)});
     localStorage.setItem("theme", "dark");
     return localStorage.getItem("unsloth_auth_token") ? "ok" : "failed";
   })()`,
);

const shots = [
  ["01-chat-dark", "/chat", 5000],
  ["02-library-dark", "/library", 4000],
  ["03-hub-dark", "/hub", 4000],
  ["04-studio-dark", "/studio", 4500],
];

for (const [name, path, settle] of shots) {
  await goto(path);
  await shoot(name, { settleMs: settle });
}

// Settings is a dialog over the shell, not a route (`/settings` redirects to `/chat`), so it is
// opened the way a user opens it: the cog in the sidebar footer.
await goto("/chat");
await sleep(2500);
await evaluate(`(() => {
  const button = [...document.querySelectorAll("button[aria-label]")]
    .find((node) => /settings/i.test(node.getAttribute("aria-label")));
  if (!button) return "no settings button";
  button.click();
  return "clicked";
})()`);
await shoot("05-settings-dark", { settleMs: 3000 });

socket.close();
console.log("done");
