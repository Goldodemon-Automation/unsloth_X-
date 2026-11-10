// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Poll scheduling policy for the training surface.
 *
 * The training page used to poll on a bare `setInterval`: a backend that was
 * down was still hit at the full rate, and the timer had no reason to stop, so
 * a run that had already finished kept the 3s cadence alive for as long as the
 * page stayed open. These two functions hold the decisions the polling hooks
 * need, as pure functions, so the timings can be read (and tested) without a
 * browser and without a running backend.
 */

/** The backoff never grows past this multiple of the base delay. */
export const POLL_BACKOFF_CAP_FACTOR = 4;

/** A hidden tab is polled this many times less often than a visible one. */
export const POLL_HIDDEN_FACTOR = 2;

export interface PollDelayInput {
  /** The delay a healthy poll uses. */
  baseMs: number;
  /**
   * The zero-based retry index: 0 is the first delay after a failure, 1 the
   * second, and so on. Ignored while the poll is healthy.
   */
  attempt: number;
  /** True when the last attempt failed. */
  failed: boolean;
  /** True while the tab is hidden. */
  hidden: boolean;
}

export interface PollStopInput {
  /** True when the thing being polled is done. */
  finished: boolean;
  /** Attempts made since the last success. */
  attempts: number;
  /** Ceiling on `attempts`. `Infinity` never stops on the ceiling. */
  maxAttempts: number;
}

/**
 * The delay before the next poll.
 *
 * A healthy poll waits the base delay. A failed one waits `base * 2^attempt`,
 * capped at `base * 4`, so a backend that stays down is asked less and less
 * often instead of at the full rate. A hidden tab waits longer than a visible
 * one either way: the answer is not on screen, so it is not worth the request.
 */
export function nextPollDelay({
  baseMs,
  attempt,
  failed,
  hidden,
}: PollDelayInput): number {
  const base = Math.max(0, Math.round(baseMs));
  const retry = failed ? Math.max(0, Math.floor(attempt)) : 0;
  const backedOff = base * 2 ** retry;
  const capped = Math.min(backedOff, base * POLL_BACKOFF_CAP_FACTOR);
  const delayed = hidden ? capped * POLL_HIDDEN_FACTOR : capped;
  return Math.max(0, Math.round(delayed));
}

/**
 * Whether the poll loop should stop.
 *
 * `finished` ends the watch when the thing being polled is done (a run that
 * has finished needs no status poll). The ceiling ends it when the loop has
 * spent its budget of attempts without a success, so a dead backend is not
 * asked forever.
 */
export function shouldStop({
  finished,
  attempts,
  maxAttempts,
}: PollStopInput): boolean {
  if (finished) {
    return true;
  }
  if (!Number.isFinite(maxAttempts)) {
    return false;
  }
  return attempts >= maxAttempts;
}
