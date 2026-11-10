// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hasAuthToken } from "@/features/auth";
import { useHfTokenStore } from "@/features/hub";
import {
  notifyNative,
  safeNotificationLabel,
  sanitizeNotificationBody,
} from "@/lib/native-notifications";
import { useEffect } from "react";
import {
  getTrainingMetrics,
  getTrainingStatus,
  isAbortError,
  streamTrainingProgress,
} from "../api/train-api";
import { nextPollDelay, shouldStop } from "../lib/poll-policy";
import {
  isTrainingStatusRequestCurrent,
  trainingStatusRequestKey,
} from "../lib/training-status-request";
import {
  createTrainingStreamScope,
  isTrainingProgressForScope,
  isTrainingStreamScopeCurrent,
} from "../lib/training-stream-scope";
import { useTrainingConfigStore } from "../stores/training-config-store";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingRuntimeStore } from "../types/runtime";

const STATUS_POLL_INTERVAL_MS = 3000;
const METRICS_POLL_INTERVAL_MS = 5000;
const IDLE_POLL_INTERVAL_MS = 30000;
/**
 * Consecutive failed polls a stream spends before it stops asking. The slow
 * recovery stream takes over when the fast one gives up, so the page keeps
 * healing without the fast cadence hammering a backend that is down.
 */
const MAX_POLL_ATTEMPTS = 8;
const STREAM_RECONNECT_DELAY_MS = 1500;
const GENERIC_TRAINING_STREAM_ERROR = "Training stream error";
const DEFAULT_TRAINING_ERROR_BODY = "Your training run stopped with an error.";

function shouldUseLiveSync(state: TrainingRuntimeStore): boolean {
  return state.isTrainingRunning || state.phase === "training";
}

function getTrainingErrorBody(state: TrainingRuntimeStore): string {
  return sanitizeNotificationBody(
    state.error ?? state.message,
    DEFAULT_TRAINING_ERROR_BODY,
  );
}

function shouldNotifyTrainingError(
  before: TrainingRuntimeStore,
  after: TrainingRuntimeStore,
): boolean {
  if (after.phase !== "error") {
    return false;
  }
  if (before.phase !== "error" || before.jobId !== after.jobId) {
    return true;
  }
  if (before.error !== GENERIC_TRAINING_STREAM_ERROR) {
    return false;
  }

  return getTrainingErrorBody(before) !== getTrainingErrorBody(after);
}

function maybeNotifyTrainingTerminalTransition(
  before: TrainingRuntimeStore,
  after: TrainingRuntimeStore,
): void {
  if (!after.jobId) {
    return;
  }
  if (
    before.stopRequested ||
    after.stopRequested ||
    after.phase === "stopped"
  ) {
    return;
  }

  const sameJob = before.jobId === after.jobId;
  const samePhase = before.phase === after.phase;

  if (after.phase === "completed" && !(sameJob && samePhase)) {
    const model = safeNotificationLabel(
      after.startModelName ?? useTrainingConfigStore.getState().selectedModel,
      "Your training run",
    );
    notifyNative({
      key: `training-completed:${after.jobId}`,
      title: "Training finished",
      body: `${model} is complete.`,
      requestPermission: false,
    }).catch(() => undefined);
  }

  if (shouldNotifyTrainingError(before, after)) {
    notifyNative({
      key: `training-error:${after.jobId}`,
      title: "Training failed",
      body: getTrainingErrorBody(after),
      requestPermission: false,
    }).catch(() => undefined);
  }
}

interface PollStreamSpec {
  /** The stream's name, used by `isStopped`. */
  name: string;
  /** The delay a healthy poll of this stream uses. */
  baseMs: number;
  /**
   * The attempt budget of the stream. It gives up when the budget is spent,
   * unless a later poll succeeds, which puts it back to work.
   */
  maxAttempts: number;
  /** True once the thing being polled is done and the stream can stop. */
  finished: () => boolean;
  /** True when a request is worth making right now. */
  eligible: () => boolean;
  /** Makes the request. Answers false when the request failed. */
  request: () => Promise<boolean>;
}

interface PollStreamState {
  spec: PollStreamSpec;
  dueAt: number;
  /** Consecutive failed attempts. A success clears it. */
  failures: number;
  failed: boolean;
  /** An armed stream has a due time and may wake the timer. */
  armed: boolean;
  stopReason: "finished" | "budget" | null;
}

interface PollScheduler {
  /** Arms the streams that can poll now and drops the ones that cannot. */
  sync(): void;
  /** Arms the streams and starts the timer. */
  start(): void;
  /** Cancels the timer. */
  stop(): void;
  /** Whether a stream has stopped, by name. */
  isStopped: (name: string) => boolean;
}

/**
 * One timer for every poll cadence.
 *
 * The Train page used to run three `setInterval` timers. A request that took
 * longer than its interval could overlap the next one, a backend that was down
 * was still hit at the full rate, and the 3s timer had no reason to stop, so a
 * run that had already finished kept it alive for as long as the page stayed
 * open. Here every cadence is a stream with its own due time and the timer
 * wakes for the earliest one; the delay comes from the poll policy (the base
 * delay while the backend answers, a capped backoff while it does not, longer
 * while the tab is hidden), and a stream stops when its run has finished or
 * when it has spent its attempt budget.
 */
function createPollScheduler(
  specs: PollStreamSpec[],
  options: { hidden: () => boolean },
): PollScheduler {
  const states: PollStreamState[] = specs.map((spec) => ({
    spec,
    dueAt: 0,
    failures: 0,
    failed: false,
    armed: false,
    stopReason: null,
  }));

  let timer: ReturnType<typeof setTimeout> | null = null;
  let disposed = false;
  let ticking = false;

  const delayFor = (state: PollStreamState): number =>
    nextPollDelay({
      baseMs: state.spec.baseMs,
      // The retry index: the first delay after a failure is the base delay.
      attempt: Math.max(0, state.failures - 1),
      failed: state.failed,
      hidden: options.hidden(),
    });

  const revive = (state: PollStreamState): void => {
    state.failures = 0;
    state.failed = false;
    state.stopReason = null;
    state.armed = false;
  };

  const schedule = (): void => {
    if (disposed) {
      return;
    }
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    const now = Date.now();
    let dueAt = Number.POSITIVE_INFINITY;
    for (const state of states) {
      if (!state.armed) {
        continue;
      }
      dueAt = Math.min(dueAt, state.dueAt);
    }
    if (!Number.isFinite(dueAt)) {
      return;
    }
    timer = setTimeout(
      () => {
        timer = null;
        void tick();
      },
      Math.max(0, dueAt - now),
    );
  };

  const syncState = (state: PollStreamState, now: number): boolean => {
    if (state.stopReason === "finished" && !state.spec.finished()) {
      // A new run: the watch starts over.
      revive(state);
      return true;
    }
    if (state.stopReason === null && state.spec.finished()) {
      state.stopReason = "finished";
      state.armed = false;
      return true;
    }
    const eligible = state.stopReason === null && state.spec.eligible();
    if (eligible && !state.armed) {
      state.armed = true;
      state.dueAt = now + delayFor(state);
      return true;
    }
    if (!eligible && state.armed) {
      state.armed = false;
      return true;
    }
    return false;
  };

  const sync = (): void => {
    if (disposed) {
      return;
    }
    const now = Date.now();
    let changed = false;
    for (const state of states) {
      changed = syncState(state, now) || changed;
    }
    if (changed) {
      schedule();
    }
  };

  const runStream = async (state: PollStreamState): Promise<boolean> => {
    let ok = false;
    try {
      ok = await state.spec.request();
    } catch {
      ok = false;
    }
    if (ok) {
      state.failures = 0;
      state.failed = false;
    } else {
      state.failures += 1;
      state.failed = true;
    }
    state.dueAt = Date.now() + delayFor(state);
    const finished = state.spec.finished();
    if (
      shouldStop({
        finished,
        attempts: state.failures,
        maxAttempts: state.spec.maxAttempts,
      })
    ) {
      state.stopReason = finished ? "finished" : "budget";
      state.armed = false;
    }
    return ok;
  };

  const runDueStreams = async (): Promise<boolean> => {
    const now = Date.now();
    let answered = false;
    for (const state of states) {
      if (disposed) {
        break;
      }
      if (!state.armed || state.dueAt > now) {
        continue;
      }
      if (!state.spec.eligible()) {
        // The target went away between arming and firing. `sync` arms it again
        // when it comes back.
        state.armed = false;
        continue;
      }
      answered = (await runStream(state)) || answered;
    }
    return answered;
  };

  const reviveBudgeted = (): void => {
    // The backend answered, so a stream that only ran out of attempts goes
    // back to work.
    for (const state of states) {
      if (state.stopReason === "budget") {
        revive(state);
      }
    }
  };

  async function tick(): Promise<void> {
    if (disposed) {
      return;
    }
    if (ticking) {
      // Another tick is running; it reschedules the timer when it finishes.
      return;
    }
    ticking = true;
    let answered = false;
    try {
      answered = await runDueStreams();
    } finally {
      ticking = false;
    }
    if (disposed) {
      return;
    }
    if (answered) {
      reviveBudgeted();
    }
    sync();
    schedule();
  }

  return {
    sync,
    start: () => {
      sync();
      schedule();
    },
    stop: () => {
      disposed = true;
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
    },
    isStopped: (name: string) =>
      states.some(
        (state) => state.spec.name === name && state.stopReason !== null,
      ),
  };
}

export function useTrainingRuntimeLifecycle(): void {
  useEffect(() => {
    const clearStartError = () => {
      const runtime = useTrainingRuntimeStore.getState();
      if (runtime.startError !== null) {
        runtime.setStartError(null);
      }
    };
    const unsubscribeConfig = useTrainingConfigStore.subscribe(
      (state, previousState) => {
        if (state.userEditRevision !== previousState.userEditRevision) {
          clearStartError();
        }
      },
    );
    const unsubscribeToken = useHfTokenStore.subscribe(
      (state, previousState) => {
        if (state.token !== previousState.token) {
          clearStartError();
        }
      },
    );
    return () => {
      unsubscribeConfig();
      unsubscribeToken();
    };
  }, []);

  useEffect(() => {
    let disposed = false;
    let streamController: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const runtimeStore = useTrainingRuntimeStore;

    const clearReconnect = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const stopStream = () => {
      clearReconnect();
      const controller = streamController;
      streamController = null;
      controller?.abort();
      runtimeStore.getState().setSseConnected(false);
    };

    // Both pollers answer with whether the request worked: the polling loop
    // needs it to back off while the backend is down.
    const pollMetrics = async (options?: {
      fresh?: boolean;
    }): Promise<boolean> => {
      if (!hasAuthToken()) return true;
      const initial = runtimeStore.getState();
      if (!initial.jobId) return true;
      const requestKey = trainingStatusRequestKey(initial);
      try {
        const metrics = await getTrainingMetrics(initial.jobId, requestKey, {
          fresh: options?.fresh,
        });
        const current = runtimeStore.getState();
        if (disposed || !isTrainingStatusRequestCurrent(requestKey, current)) {
          return true;
        }
        current.applyMetrics(metrics);
        return true;
      } catch (error) {
        if (!isAbortError(error) && !disposed && hasAuthToken()) {
          runtimeStore.getState().setSseConnected(false);
        }
        // An aborted request is not a failure of the backend.
        return isAbortError(error);
      }
    };

    const pollStatus = async (options?: {
      fresh?: boolean;
      suppressNativeNotifications?: boolean;
    }): Promise<boolean> => {
      if (!hasAuthToken()) return true;
      const initial = runtimeStore.getState();
      const requestKey = trainingStatusRequestKey(initial);
      try {
        const status = await getTrainingStatus(requestKey, {
          fresh: options?.fresh,
        });
        const current = runtimeStore.getState();
        if (disposed || !isTrainingStatusRequestCurrent(requestKey, current)) {
          return true;
        }

        const previousState = initial;
        current.applyStatus(status);

        const nextState = runtimeStore.getState();
        if (!options?.suppressNativeNotifications) {
          maybeNotifyTrainingTerminalTransition(previousState, nextState);
        }
        const acceptedLocalStart =
          nextState.isStarting &&
          status.start_request_state === "accepted" &&
          status.start_request_id === nextState.startRequestId;
        if (
          shouldUseLiveSync(nextState) &&
          status.start_request_state !== "pending" &&
          (!nextState.isStarting || acceptedLocalStart)
        ) {
          void ensureStream();
        } else {
          stopStream();
        }
        return true;
      } catch (error) {
        if (!isAbortError(error) && !disposed && hasAuthToken()) {
          runtimeStore.getState().setSseConnected(false);
        }
        // An aborted request is not a failure of the backend.
        return isAbortError(error);
      }
    };

    const ensureStream = async () => {
      const state = runtimeStore.getState();
      const scope = createTrainingStreamScope(state);
      if (disposed || streamController || !scope || !shouldUseLiveSync(state)) {
        return;
      }

      clearReconnect();
      const controller = new AbortController();
      streamController = controller;
      const isCurrentStream = () =>
        streamController === controller &&
        !controller.signal.aborted &&
        isTrainingStreamScopeCurrent(scope, runtimeStore.getState());

      try {
        await streamTrainingProgress({
          signal: controller.signal,
          expectedJobId: scope.jobId,
          lastEventId: state.lastEventId,
          onOpen: () => {
            if (!isCurrentStream()) {
              controller.abort();
              return;
            }
            runtimeStore.getState().setSseConnected(true);
          },
          onEvent: (event) => {
            if (!isCurrentStream()) {
              controller.abort();
              return;
            }
            if (!isTrainingProgressForScope(scope, event.payload)) {
              stopStream();
              return;
            }
            const liveStore = runtimeStore.getState();
            liveStore.applyProgress(event.payload, event.id ?? undefined);

            if (event.event === "complete") {
              stopStream();
              void pollStatus({ fresh: true });
              void pollMetrics({ fresh: true });
              return;
            }

            if (event.event === "error") {
              liveStore.setRuntimeError(GENERIC_TRAINING_STREAM_ERROR);
              stopStream();
            }
          },
        });
      } catch (error) {
        if (
          isCurrentStream() &&
          !disposed &&
          !controller.signal.aborted &&
          !isAbortError(error)
        ) {
          runtimeStore.getState().setSseConnected(false);
        }
      } finally {
        if (streamController === controller) {
          streamController = null;
          runtimeStore.getState().setSseConnected(false);

          if (!disposed && !controller.signal.aborted) {
            const liveState = runtimeStore.getState();
            if (
              isTrainingStreamScopeCurrent(scope, liveState) &&
              shouldUseLiveSync(liveState)
            ) {
              reconnectTimer = setTimeout(() => {
                reconnectTimer = null;
                if (
                  !disposed &&
                  isTrainingStreamScopeCurrent(scope, runtimeStore.getState())
                ) {
                  void ensureStream();
                }
              }, STREAM_RECONNECT_DELAY_MS);
            }
          }
        }
      }
    };

    const hydrate = async () => {
      runtimeStore.getState().setHydrating(true);
      try {
        await Promise.all([
          pollStatus({ suppressNativeNotifications: true }),
          pollMetrics(),
        ]);
      } finally {
        if (!disposed) {
          runtimeStore.getState().setHydrating(false);
          runtimeStore.getState().setHasHydrated(true);
        }
      }
    };

    const isIdle = () => {
      const s = runtimeStore.getState();
      return s.phase === "idle" && !s.isTrainingRunning;
    };

    // A run that reached a terminal phase is over: the fast cadences have
    // nothing left to ask for, and the slow recovery stream takes over.
    const isTerminal = () => {
      const phase = runtimeStore.getState().phase;
      return phase === "completed" || phase === "error" || phase === "stopped";
    };

    let scheduler: PollScheduler | null = null;
    const isStatusStreamStopped = () => scheduler?.isStopped("status") === true;

    scheduler = createPollScheduler(
      [
        {
          name: "status",
          baseMs: STATUS_POLL_INTERVAL_MS,
          maxAttempts: MAX_POLL_ATTEMPTS,
          finished: () => isTerminal(),
          eligible: () => {
            const s = runtimeStore.getState();
            if (!s.hasHydrated || s.isHydrating) {
              return false;
            }
            // Neither idle nor finished: a run is on.
            return !(isIdle() || isTerminal());
          },
          request: () => pollStatus(),
        },
        {
          name: "metrics",
          baseMs: METRICS_POLL_INTERVAL_MS,
          maxAttempts: MAX_POLL_ATTEMPTS,
          finished: () => isTerminal(),
          eligible: () => {
            const s = runtimeStore.getState();
            if (isIdle() || isTerminal()) {
              return false;
            }
            return shouldUseLiveSync(s) || s.currentStep > 0;
          },
          request: () => pollMetrics(),
        },
        {
          // Low-frequency poll: recovers failed hydration, catches out-of-band
          // state changes, and keeps the page alive when the fast status
          // stream has given up its attempt budget.
          name: "idle",
          baseMs: IDLE_POLL_INTERVAL_MS,
          // The last resort: it keeps a slow heartbeat for as long as the page
          // is open, so a page that outlived an outage still heals.
          maxAttempts: Number.POSITIVE_INFINITY,
          finished: () => false,
          eligible: () => {
            const s = runtimeStore.getState();
            if (!s.hasHydrated || s.isHydrating) {
              return false;
            }
            return isIdle() || isTerminal() || isStatusStreamStopped();
          },
          request: () => pollStatus(),
        },
      ],
      {
        hidden: () => document.visibilityState === "hidden",
      },
    );

    const unsubscribeRuntime = runtimeStore.subscribe(
      (state, previousState) => {
        if (state.jobId !== previousState.jobId) {
          stopStream();
        }
        scheduler?.sync();
      },
    );

    void hydrate();

    scheduler.start();

    return () => {
      disposed = true;
      scheduler?.stop();
      unsubscribeRuntime();
      stopStream();
    };
  }, []);
}
