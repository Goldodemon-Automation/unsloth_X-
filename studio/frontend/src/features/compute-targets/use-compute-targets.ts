// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import {
  type SetComputeTargetResult,
  getComputeTargets,
  setComputeTarget,
} from "./api";
import { type ComputeTargetsResponse, computeTargetError } from "./types";

export interface ComputeTargetsSnapshot {
  data: ComputeTargetsResponse | null;
  /** A message a UI can show, or null while the last call went through. */
  error: string | null;
  loading: boolean;
}

export interface UseComputeTargetsResult extends ComputeTargetsSnapshot {
  /** Ask again, bypassing the cache; joins a probe that is already in flight. */
  refresh: () => void;
  /**
   * Ask again from scratch: starts a fresh probe even when one is in flight, so an error state
   * cannot stick to the surface. The cached error stays put until that probe settles -- the notice
   * ranks a failure above a retry in flight -- and only a successful read clears it.
   */
  retry: () => void;
  /**
   * Ask the backend to train on `target`; resolves true once the cache holds a fresh payload,
   * budget and NPU capability included.
   */
  choose: (id: string) => Promise<boolean>;
}

// The Train page mounts on every navigation, so the hardware probe is module state:
// one cache, one in-flight promise, and every mounted hook sees the same answer.
let cached: ComputeTargetsResponse | null = null;
let lastError: string | null = null;
let attempted = false;
let pending: Promise<ComputeTargetsResponse | null> | null = null;
// Bumped by every probe, so an older reply that lands after a retry cannot paint over it.
let generation = 0;
const listeners = new Set<(snapshot: ComputeTargetsSnapshot) => void>();

function snapshot(): ComputeTargetsSnapshot {
  // Loading while a probe is in flight, and before any probe has been attempted: the first render
  // lands before the effect can start one, and an inactive hook never starts one at all, so
  // without the second half those paints answer "none" and the card comes up blank.
  return {
    data: cached,
    error: lastError,
    loading: pending !== null || !attempted,
  };
}

/** Push the module state to every mounted hook, so both surfaces agree. */
function publish(): void {
  const next = snapshot();
  for (const listener of listeners) {
    listener(next);
  }
}

/**
 * Probe the backend. A plain call reuses the probe already in flight; `force` starts a new one
 * and makes it the answer that counts, so the probe it bypassed cannot repaint the surface.
 */
function fetchOnce(force = false): Promise<ComputeTargetsResponse | null> {
  if (!force && pending) {
    // The probe is in flight, so a caller that renders from `loading` sees it.
    publish();
    return pending;
  }
  const ticket = ++generation;
  const run = getComputeTargets()
    .then((data) => {
      if (ticket !== generation) {
        return cached;
      }
      cached = data;
      lastError = null;
      return data;
    })
    .catch((error: unknown) => {
      if (ticket !== generation) {
        return null;
      }
      lastError = computeTargetError(error).message;
      return null;
    })
    .finally(() => {
      attempted = true;
      if (ticket === generation) {
        pending = null;
      }
      publish();
    });
  pending = run;
  publish();
  return run;
}

/** The cached answer with `id` reported as the resolved target, if it is one. */
function withResolved(
  data: ComputeTargetsResponse | null,
  id: string,
): ComputeTargetsResponse | null {
  if (!data) return null;
  const target = data.targets.find((entry) => entry.id === id);
  return target ? { ...data, resolved: target.id } : null;
}

/**
 * The cached answer with the backend's reply folded in. The POST reply carries no `budget` or
 * `npu`, so this merge is only the instant repaint; `choose` re-reads the whole payload right
 * after, and that read is what those two fields are rendered from.
 */
function withChosen(
  data: ComputeTargetsResponse,
  result: SetComputeTargetResult,
): ComputeTargetsResponse {
  return {
    ...data,
    targets: data.targets.map((entry) =>
      entry.id === result.target.id ? result.target : entry,
    ),
    resolved: result.resolved,
    // Carry what was asked for, not just what it resolved to. The capacity check sends `requested`
    // so a run keeps meaning "auto", and dropping the field here left it at the page-load value --
    // the estimate below could then be sized for the device the user had just moved away from.
    requested: result.requested ?? data.requested,
  };
}

export function useComputeTargets(active = true): UseComputeTargetsResult {
  const [state, setState] = useState<ComputeTargetsSnapshot>(snapshot);

  useEffect(() => {
    listeners.add(setState);
    // Read again: a probe may have settled between the first render and this effect.
    setState(snapshot());
    return () => {
      listeners.delete(setState);
    };
  }, []);

  useEffect(() => {
    // One probe per session: navigating back must not re-read the hardware.
    if (!active || attempted) return;
    fetchOnce();
  }, [active]);

  const refresh = useCallback(() => {
    fetchOnce();
  }, []);

  const retry = useCallback(() => {
    // `refresh` joins a probe already in flight; retry probes from scratch so a failed read is
    // never the answer the surface settles on. The cached error is kept until that probe settles:
    // the notice ranks a failure above a retry in flight, so "Try again" must keep the message
    // and its button instead of flashing back to "checking". A successful read clears it.
    fetchOnce(true);
  }, []);

  const choose = useCallback(async (id: string) => {
    const previous = cached;
    const optimistic = withResolved(previous, id);
    if (optimistic) {
      cached = optimistic;
      publish();
    }
    try {
      const result = await setComputeTarget(id);
      lastError = null;
      if (cached) {
        cached = withChosen(cached, result);
        publish();
      }
      // The POST reply carries no budget or NPU capability, and both follow the chosen device, so
      // re-read the whole payload instead of inventing the missing fields or leaving the previous
      // device's sentences on screen. Forced: the POST makes any probe already in flight stale.
      await fetchOnce(true);
      return true;
    } catch (error) {
      cached = previous;
      lastError = computeTargetError(error).message;
      publish();
      return false;
    }
  }, []);

  return { ...state, refresh, retry, choose };
}
