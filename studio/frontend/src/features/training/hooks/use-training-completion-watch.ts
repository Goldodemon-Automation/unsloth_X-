// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

// eslint-disable-next-line no-restricted-imports
import { checkDiskSpace } from "@/features/settings/low-disk-check";
import { createSingleFlightRequest } from "@/lib/single-flight-request";
import { getTrainingStatus } from "../api/train-api";
import { nextPollDelay, shouldStop } from "../lib/poll-policy";
import {
  isTrainingStatusRequestCurrent,
  trainingStatusRequestKey,
} from "../lib/training-status-request";
import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";

const WATCH_INTERVAL_MS = 6000;
/**
 * Consecutive failed polls the watch spends before it stops asking. A backend
 * that is down should not be polled at the full rate until the run ends.
 */
const MAX_WATCH_ATTEMPTS = 8;

/**
 * Keep training state fresh while a run is active, even off the Train page.
 *
 * The lifecycle poll (`useTrainingRuntimeLifecycle`) only runs while Train is mounted, so a run
 * finishing on another tab would leave the sidebar spinner stuck. Polls `/api/train/status` only
 * while a run is in progress. Mount once in an always-rendered shell.
 *
 * The poll runs on one timeout loop instead of a fixed interval: the delay is
 * the base one while the backend answers, a capped backoff while it does not,
 * and longer while the tab is hidden. The loop stops once it has spent its
 * attempt budget, and it is torn down (with the disk check below) when the
 * run leaves the active state.
 */
export function useTrainingCompletionWatch(): void {
  const active = useTrainingRuntimeStore(isTrainingStartPending);

  useEffect(() => {
    if (!active) {
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let failures = 0;

    const tick = createSingleFlightRequest(async (): Promise<boolean> => {
      const initial = useTrainingRuntimeStore.getState();
      const requestKey = trainingStatusRequestKey(initial);
      try {
        const status = await getTrainingStatus(requestKey);
        const runtime = useTrainingRuntimeStore.getState();
        if (!cancelled && isTrainingStatusRequestCurrent(requestKey, runtime)) {
          runtime.applyStatus(status);
        }
        return true;
      } catch {
        // Transient network/auth hiccup; the next tick retries, later than the
        // last one did.
        return false;
      }
    });

    const scheduleNext = (failed: boolean) => {
      if (cancelled) {
        return;
      }
      timer = setTimeout(
        () => {
          timer = null;
          void run();
        },
        nextPollDelay({
          baseMs: WATCH_INTERVAL_MS,
          // The retry index: the first delay after a failure is the base delay.
          attempt: Math.max(0, failures - 1),
          failed,
          hidden: document.visibilityState === "hidden",
        }),
      );
    };

    const run = async () => {
      if (cancelled) {
        return;
      }
      const ok = await tick();
      if (cancelled) {
        return;
      }
      failures = ok ? 0 : failures + 1;
      if (
        shouldStop({
          finished: false,
          attempts: failures,
          maxAttempts: MAX_WATCH_ATTEMPTS,
        })
      ) {
        return;
      }
      scheduleNext(!ok);
    };

    scheduleNext(false);

    return () => {
      cancelled = true;
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      // The post-download half of the training disk check. startTraining takes the
      // reading BEFORE the run; the worker then downloads the base model and any
      // remote dataset on its own time, so the only honest moment to look again is
      // when the run stops being active. This cleanup is that moment, and it does
      // not care whether the run finished or failed: a run that died because the
      // disk filled is precisely the one worth reporting.
      //
      // forced, for the reason the download manager's finalize is forced: the
      // reading has to be taken AFTER the write, and unforced it would be swallowed
      // by the interval or handed a figure from before the run.
      void checkDiskSpace({ force: true });
    };
  }, [active]);
}
