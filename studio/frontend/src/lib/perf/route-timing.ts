// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Route-mount timing: one span per navigation, from the render that starts the
 * new location to the effect after its content committed. The first question
 * this answers is how long the Train page takes to mount.
 *
 * Only the public perf API is used, so a disabled session stays inert.
 */

import { type PerfEntry, mark, measure, snapshot } from "./index.ts";

/**
 * Top-level routes worth their own span name. Unknown paths share
 * `route:other` rather than minting a name per URL. `/studio` is the Train
 * page, so it reports under the same `route:train` name as `/train`.
 */
const KNOWN_ROUTES = new Set([
  "chat",
  "train",
  "projects",
  "library",
  "hub",
  "login",
  "signup",
  "change-password",
  "export",
  "merge",
  "scale",
  "labs",
  "api-monitor",
  "data-recipes",
  "images",
  "video",
  "audio",
]);

/** How many slow spans the report names. */
const REPORTED_SPANS = 3;

const TRAILING_SLASHES = /\/+$/;

/** The span name for a pathname: `/` is the index, trailing slashes are noise. */
export function routeSpanName(pathname: string): string {
  const trimmed = pathname.replace(TRAILING_SLASHES, "");
  if (trimmed === "") {
    return "route:index";
  }
  const segment = trimmed.split("/").find(Boolean) ?? "";
  if (segment === "studio") {
    return "route:train";
  }
  return KNOWN_ROUTES.has(segment) ? `route:${segment}` : "route:other";
}

/** Start the span for a navigation. The last start for a name wins. */
export function markRouteStart(pathname: string): void {
  mark(`${routeSpanName(pathname)}:start`);
}

/** Close the span once the new route's content has committed. */
export function markRouteCommit(pathname: string): void {
  const name = routeSpanName(pathname);
  mark(`${name}:end`);
  measure(name, `${name}:start`, `${name}:end`);
}

/** The slowest three recorded route spans, as `"name  12.3ms"` lines. */
export function describeRouteSpans(entries: PerfEntry[]): string[] {
  return entries
    .filter((entry) => entry.name.startsWith("route:"))
    .sort((left, right) => right.durationMs - left.durationMs)
    .slice(0, REPORTED_SPANS)
    .map((entry) => `${entry.name}  ${entry.durationMs.toFixed(1)}ms`);
}

/** Convenience for a report: what has been recorded so far. */
export function routeReportLines(): string[] {
  return describeRouteSpans(snapshot());
}
