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
  /** Ask the backend to train on `target`; resolves true once the cache holds the reply. */
  choose: (id: string) => Promise<boolean>;
}

// The Train page mounts on every navigation, so the hardware probe is module state:
// one cache, one in-flight promise, and every mounted hook sees the same answer.
let cached: ComputeTargetsResponse | null = null;
let lastError: string | null = null;
let attempted = false;
let pending: Promise<ComputeTargetsResponse | null> | null = null;
const listeners = new Set<(snapshot: ComputeTargetsSnapshot) => void>();

function snapshot(): ComputeTargetsSnapshot {
  return { data: cached, error: lastError, loading: pending !== null };
}

/** Push the module state to every mounted hook, so both surfaces agree. */
function publish(): void {
  const next = snapshot();
  for (const listener of listeners) {
    listener(next);
  }
}

function fetchOnce(): Promise<ComputeTargetsResponse | null> {
  pending ??= getComputeTargets()
    .then((data) => {
      cached = data;
      lastError = null;
      return data;
    })
    .catch((error: unknown) => {
      lastError = computeTargetError(error).message;
      return null;
    })
    .finally(() => {
      pending = null;
      attempted = true;
      publish();
    });
  // The probe is in flight, so a caller that renders from `loading` sees it.
  publish();
  return pending;
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

/** The cached answer with the backend's reply folded in. */
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
      } else {
        // Nothing cached to fold the reply into, so read the whole picture.
        fetchOnce();
      }
      return true;
    } catch (error) {
      cached = previous;
      lastError = computeTargetError(error).message;
      publish();
      return false;
    }
  }, []);

  return { ...state, refresh, choose };
}
