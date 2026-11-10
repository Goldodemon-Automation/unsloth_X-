// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Perf entry point. The re-exports are spelled out name by name and through
 * plain `import`/`export` rather than `export ... from`: a star re-export
 * would merge the modules' names silently, which is exactly the mess
 * `scripts/lint_no_duplicate_ts_imports.py` exists to prevent.
 */

import {
  inpStats,
  resetInteractions,
  trackInteraction,
} from "./interaction.ts";
import {
  type LongTaskObserverOptions,
  type LongTaskStats,
  longTaskStats,
  observeLongTasks,
  resetLongTasks,
} from "./long-task.ts";
import {
  MAX_ENTRIES,
  type PerfEntry,
  mark,
  measure,
  perfEnabled,
  reportLines,
  reset,
  setPerfEnabled,
  snapshot,
  time,
  timeAsync,
} from "./marks.ts";

export { inpStats, resetInteractions, trackInteraction };
export {
  type LongTaskObserverOptions,
  type LongTaskStats,
  longTaskStats,
  observeLongTasks,
  resetLongTasks,
};
export {
  MAX_ENTRIES,
  type PerfEntry,
  mark,
  measure,
  perfEnabled,
  reportLines,
  reset,
  setPerfEnabled,
  snapshot,
  time,
  timeAsync,
};

/**
 * Install the instrumentation for this session: long-task observation today,
 * more hooks later behind the same combined disposer. A disabled session gets
 * an inert disposer and arms nothing -- production pays one boolean check.
 */
export function installPerf(): () => void {
  if (!perfEnabled()) {
    return () => undefined;
  }
  const disposeLongTasks = observeLongTasks();
  let disposed = false;
  return () => {
    if (disposed) {
      return;
    }
    disposed = true;
    disposeLongTasks();
  };
}
