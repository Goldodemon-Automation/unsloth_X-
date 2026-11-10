// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type NpuAssistRequested,
  type NpuTrainingAssist,
  getNpuTrainingAssist,
  npuAssistKey,
} from "./npu-training-assist";

export interface UseNpuTrainingAssistOptions {
  evalEnabled?: boolean;
  qat?: boolean;
  requested?: NpuAssistRequested;
  /**
   * The device the run would use, or "auto" when none is resolved. The backend's answer describes
   * that device, so it is part of the cache key only -- it is never sent as a request parameter.
   */
  primaryTargetId?: string;
  active?: boolean;
}

export interface UseNpuTrainingAssistResult {
  assist: NpuTrainingAssist | null;
  error: string | null;
  loading: boolean;
  refresh: () => void;
}

interface AssistState {
  key: string;
  assist: NpuTrainingAssist | null;
  error: string | null;
}

interface Inflight {
  promise: Promise<NpuTrainingAssist>;
  controller: AbortController;
  /** How many mounted hooks are waiting on this request. */
  waiters: number;
  settled: boolean;
}

/** One answer per request key: the panel is a GET, so revisiting it is free. */
const cache = new Map<string, NpuTrainingAssist>();
const inflight = new Map<string, Inflight>();

class InflightRequest implements Inflight {
  readonly controller = new AbortController();
  waiters = 1;
  settled = false;
  readonly promise: Promise<NpuTrainingAssist>;

  constructor(key: string, options: UseNpuTrainingAssistOptions) {
    this.promise = getNpuTrainingAssist({
      ...options,
      signal: this.controller.signal,
    })
      .then((assist) => {
        cache.set(key, assist);
        return assist;
      })
      .finally(() => {
        this.settled = true;
        if (inflight.get(key) === this) inflight.delete(key);
      });
    // A rejection nobody is left to hear still must not go unhandled.
    this.promise.catch(() => {});
  }
}

function acquire(key: string, options: UseNpuTrainingAssistOptions): Inflight {
  const existing = inflight.get(key);
  if (existing) {
    existing.waiters += 1;
    return existing;
  }
  const cached = cache.get(key);
  if (cached) {
    // Already answered: resolve on the same microtask as a real request so the
    // hook never depends on where the answer came from.
    return {
      promise: Promise.resolve(cached),
      controller: new AbortController(),
      waiters: 1,
      settled: true,
    };
  }
  const entry = new InflightRequest(key, options);
  inflight.set(key, entry);
  return entry;
}

function release(key: string, entry: Inflight): void {
  entry.waiters -= 1;
  if (entry.waiters > 0) return;
  if (inflight.get(key) === entry) inflight.delete(key);
  if (!entry.settled) entry.controller.abort();
}

function messageFor(cause: unknown): string {
  if (cause instanceof Error && cause.message) return cause.message;
  return "Could not read NPU training assist";
}

/**
 * What the NPU can do for the current run. Every request is cancelled as soon
 * as nothing is waiting on it, because the key changes as options change.
 */
export function useNpuTrainingAssist(
  options: UseNpuTrainingAssistOptions = {},
): UseNpuTrainingAssistResult {
  const {
    evalEnabled = false,
    qat = false,
    requested = "auto",
    primaryTargetId = "auto",
    active = true,
  } = options;
  // The device is in the key, so switching it asks again instead of serving the sentence the
  // backend wrote for the device the run just moved away from. `acquire` below deliberately gets
  // only the request fields: the device must not reach the query string.
  const key = npuAssistKey({ evalEnabled, qat, requested, primaryTargetId });
  const [state, setState] = useState<AssistState | null>(null);
  // A refresh asks for the same key again, so it needs its own reason to run.
  const [reload, setReload] = useState(0);
  const servedRef = useRef(0);

  useEffect(() => {
    if (!active) return;
    const forced = reload !== servedRef.current;
    servedRef.current = reload;
    if (forced) cache.delete(key);
    let live = true;
    const entry = acquire(key, { evalEnabled, qat, requested });
    entry.promise.then(
      (assist) => {
        if (!live) return;
        setState((prev) =>
          prev?.key === key && prev.assist === assist
            ? prev
            : { key, assist, error: null },
        );
      },
      (cause: unknown) => {
        if (!live || entry.controller.signal.aborted) return;
        setState({ key, assist: null, error: messageFor(cause) });
      },
    );
    return () => {
      live = false;
      release(key, entry);
    };
  }, [active, key, evalEnabled, qat, requested, reload]);

  const refresh = useCallback(() => {
    // Clearing here, in an event handler, is what drops a stale error so the
    // panel can show its loading line until the new answer arrives.
    setState(null);
    setReload((count) => count + 1);
  }, []);

  const current: AssistState =
    state?.key === key
      ? state
      : { key, assist: cache.get(key) ?? null, error: null };
  return {
    assist: current.assist,
    error: current.error,
    loading: active && current.assist === null && current.error === null,
    refresh,
  };
}
