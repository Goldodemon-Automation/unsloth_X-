// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Window arithmetic for the training run history grid (features/studio/history-card-grid.tsx), the
// app's heaviest repeated-render surface: while any run trains, a poll replaces the whole runs array
// every five seconds, and every "Load more" grows it by a page, so a long history re-rendered every
// poll was the cost. The shape is the widen-only mount window progressive-mount-controller.ts uses
// for the chat thread, and for the same reason: on a repeated-render surface the cost being cut is
// mounting and re-rendering rows, and the only thing that collects that is not doing it -- hiding a
// row still pays its React render on every poll, so an unmounted row is the only cheap row.
// Pure and dependency-free so tests/render-budget.test.ts can pin the clamping, growth and style
// decisions instead of letting them drift out of the component one edit at a time.

import type { CSSProperties } from "react";

/**
 * Rows to mount before any scrolling: `firstScreen` clamped into `[1, total]`, and `0` for an empty
 * list.
 *
 * The lower clamp is load-bearing. A window of `0` over a non-empty list paints an empty grid over
 * rows that exist, and growth is driven by the frontier of MOUNTED rows, so with nothing mounted
 * there is no frontier to approach and the list stays empty forever. A window above `total` would
 * just be rows that do not exist. `firstScreen` is the caller's claim about how many cards fill the
 * first screen; a viewport narrow or short enough can compute it as `0`, which the clamp turns into
 * "one row" -- the smallest window that paints something and still has a frontier to grow from.
 */
export function initialWindow(total: number, firstScreen: number): number {
  if (total <= 0) return 0;
  return Math.min(total, Math.max(1, firstScreen));
}

/**
 * The window after one growth step, capped at `total`. Widening-only while the list holds still:
 * rows already mounted stay mounted (the "nothing is ever unmounted" rule from
 * progressive-mount-controller.ts), so a poll landing mid-scroll cannot yank a card out from under
 * the reader.
 *
 * `step` is clamped to at least one row. A `0` or negative step would return the current window
 * unchanged, leaving the frontier parked against the last mounted row with no new row ever
 * intersecting it: the user would sit at the end of the visible list with no way to reach the rest.
 *
 * When the list itself shrinks below the window (a delete, a narrowing filter) the cap wins over
 * "never shrinks": those rows are gone either way, and returning `rendered` would ask the caller to
 * render rows the list no longer has.
 */
export function nextWindow(
  rendered: number,
  total: number,
  step: number,
): number {
  if (total <= 0) return 0;
  return Math.min(total, Math.max(0, rendered) + Math.max(1, step));
}

/**
 * Whether row `index` is inside the `rendered`-row window: mounts and renders normally.
 *
 * This is the single definition of "inside the window" that both the grid's mount gate and
 * `contentVisibilityStyle` share, so mounting and styling cannot drift apart -- the test pinning
 * their agreement pins it by construction rather than by two lists of comparisons that happen to
 * match today.
 */
export function shouldRenderRow(index: number, rendered: number): boolean {
  return index >= 0 && index < rendered;
}

/**
 * The card's authored height (`h-[calc(11.5rem*var(--ui-space-scale,1))]` on the grid card), quoted
 * so a skipped row keeps the footprint of a rendered one and the scrollbar and the frontier below it
 * do not jump. The `auto` prefix lets the browser remember the real size once one has been measured,
 * the same spelling code-fence-defer.ts ships for the same job.
 */
const CARD_INTRINSIC_SIZE = "auto 11.5rem";

/**
 * Per-row render style. Rows inside the window are pinned `visible`; rows past it get
 * `content-visibility: auto` plus the intrinsic size above so the browser can skip their layout
 * while they keep their place.
 *
 * In-window rows stay `visible` instead of joining the `auto` band because an on-screen `auto` row
 * still pays the bounce: every flip between skipped and laid out -- each scroll jitter at the row's
 * edge -- re-lays the row out and shifts what the reader is looking at. The window is exactly what
 * the reader has reached, so it is exactly the region that must never bounce. The grid withholds
 * out-of-window rows from the DOM entirely today (cheaper still than `auto`); this helper is what
 * any row past the window must carry wherever such a row is rendered, so a later change that renders
 * the tail cannot silently paint a grid that bounces under the reader.
 */
export function contentVisibilityStyle(
  rendered: number,
  index: number,
): CSSProperties {
  return shouldRenderRow(index, rendered)
    ? { contentVisibility: "visible" }
    : { contentVisibility: "auto", containIntrinsicSize: CARD_INTRINSIC_SIZE };
}

/**
 * React key for a history card: the run's id, verbatim.
 *
 * Never the array index. The poll replaces (and re-sorts) the whole runs array every few seconds,
 * and an index key makes React treat every card as a fresh element at a new position, re-mounting
 * the entire grid on every poll -- the exact cost this window exists to remove. The id is server-side
 * identity, so it follows the run across sorts, page appends and poll replacements, and only a real
 * change to a run can re-render its card. `created_at` is the fallback for legacy rows an import
 * produced without ids, where the creation time is the only identity there is.
 */
export function historyCardKey(run: {
  id: string;
  created_at?: string | null;
}): string {
  return run.id ? run.id : `created:${run.created_at ?? ""}`;
}
