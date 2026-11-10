// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Long-task observation must degrade, not explode: Safari and Firefox have no
// `longtask` entry type at all, and the desktop webview may have no
// PerformanceObserver worth the name. Every shape below is exercised through
// a fake constructor installed on `globalThis`, which is also the only way to
// emit a long task on a host that produces none.

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_ENTRIES,
  installPerf,
  longTaskStats,
  observeLongTasks,
  resetLongTasks,
  setPerfEnabled,
} from "../src/lib/perf/index.ts";

type FakeTask = {
  duration: number;
  attribution?: {
    containerType?: string;
    containerSrc?: string;
    containerId?: string;
    containerName?: string;
  }[];
};

type FakeObserverRecord = {
  callback: (list: PerformanceObserverEntryList) => void;
  observed: PerformanceObserverInit | null;
  disconnected: boolean;
};

type FakeObserverHost = {
  observers: FakeObserverRecord[];
  emit(tasks: FakeTask[]): void;
  restore(): void;
};

function installFakeObserver(
  options: {
    supportedEntryTypes?: readonly string[];
    observeThrows?: boolean;
  } = {},
): FakeObserverHost {
  const observers: FakeObserverRecord[] = [];
  class FakePerformanceObserver {
    static supportedEntryTypes = options.supportedEntryTypes;

    readonly record: FakeObserverRecord;

    constructor(callback: PerformanceObserverCallback) {
      this.record = {
        callback: (list) =>
          callback(list, this as unknown as PerformanceObserver),
        observed: null,
        disconnected: false,
      };
      observers.push(this.record);
    }

    observe(init: PerformanceObserverInit): void {
      if (options.observeThrows === true) {
        throw new TypeError("unsupported entry type");
      }
      this.record.observed = init;
    }

    disconnect(): void {
      this.record.disconnected = true;
    }
  }
  const previous = globalThis.PerformanceObserver;
  globalThis.PerformanceObserver =
    FakePerformanceObserver as unknown as typeof PerformanceObserver;
  return {
    observers,
    emit(tasks: FakeTask[]): void {
      for (const observer of observers) {
        if (observer.disconnected) {
          continue;
        }
        const list = {
          getEntries: () => tasks,
        } as unknown as PerformanceObserverEntryList;
        observer.callback(list);
      }
    },
    restore(): void {
      globalThis.PerformanceObserver = previous;
    },
  };
}

test("without PerformanceObserver the disposer is inert and cannot throw", () => {
  const previous = globalThis.PerformanceObserver;
  try {
    assert.ok(Reflect.deleteProperty(globalThis, "PerformanceObserver"));
    const dispose = observeLongTasks();
    dispose();
    dispose();
    assert.deepEqual(longTaskStats(), {
      count: 0,
      totalMs: 0,
      worstMs: 0,
      worstAttribution: null,
    });
  } finally {
    globalThis.PerformanceObserver = previous;
  }
});

test("an engine without the longtask entry type is left alone", () => {
  const fake = installFakeObserver({
    supportedEntryTypes: ["mark", "measure"],
  });
  try {
    const dispose = observeLongTasks();
    assert.equal(fake.observers.length, 0); // nothing was even constructed
    dispose();
  } finally {
    fake.restore();
    resetLongTasks();
  }
});

test("an observe() that throws degrades to the inert disposer", () => {
  const fake = installFakeObserver({ observeThrows: true });
  try {
    const dispose = observeLongTasks();
    dispose();
    dispose();
    assert.deepEqual(longTaskStats(), {
      count: 0,
      totalMs: 0,
      worstMs: 0,
      worstAttribution: null,
    });
  } finally {
    fake.restore();
    resetLongTasks();
  }
});

test("long tasks accumulate into the stats, attribution and all", () => {
  const fake = installFakeObserver();
  try {
    resetLongTasks();
    assert.equal(fake.observers.length, 0); // inert until asked
    const dispose = observeLongTasks({ thresholdMs: 50 });
    assert.equal(fake.observers.length, 1);
    assert.deepEqual(fake.observers[0]?.observed, {
      type: "longtask",
      buffered: true,
    });
    fake.emit([
      { duration: 60, attribution: [{ containerType: "window" }] },
      {
        duration: 120,
        attribution: [{ containerName: "chat", containerSrc: "https://embed" }],
      },
      { duration: 30 }, // under the caller's threshold: ignored
    ]);
    assert.deepEqual(longTaskStats(), {
      count: 2,
      totalMs: 180,
      worstMs: 120,
      worstAttribution: "chat", // the most specific container hint wins
    });
    dispose();
  } finally {
    fake.restore();
    resetLongTasks();
  }
});

test("the record ring stays bounded at MAX_ENTRIES", () => {
  const fake = installFakeObserver();
  try {
    resetLongTasks();
    const dispose = observeLongTasks();
    const tasks: FakeTask[] = [];
    for (let i = 0; i < MAX_ENTRIES * 2; i += 1) {
      tasks.push({ duration: 50 + i });
    }
    fake.emit(tasks);
    let expectedTotal = 0;
    for (let i = MAX_ENTRIES; i < MAX_ENTRIES * 2; i += 1) {
      expectedTotal += 50 + i;
    }
    assert.deepEqual(longTaskStats(), {
      count: MAX_ENTRIES, // the oldest half was dropped
      totalMs: expectedTotal,
      worstMs: 50 + MAX_ENTRIES * 2 - 1,
      worstAttribution: null,
    });
    dispose();
  } finally {
    fake.restore();
    resetLongTasks();
  }
});

test("the disposer is idempotent and stops delivery", () => {
  const fake = installFakeObserver();
  try {
    resetLongTasks();
    const dispose = observeLongTasks();
    dispose();
    dispose();
    assert.equal(fake.observers[0]?.disconnected, true);
    fake.emit([{ duration: 300 }]);
    assert.equal(longTaskStats().count, 0);
  } finally {
    fake.restore();
    resetLongTasks();
  }
});

test("installPerf arms nothing while the session is off", () => {
  setPerfEnabled(false);
  const fake = installFakeObserver();
  try {
    const dispose = installPerf();
    assert.equal(fake.observers.length, 0);
    dispose();
  } finally {
    fake.restore();
    setPerfEnabled(false);
    resetLongTasks();
  }
});

test("installPerf installs observation and tears it down once", () => {
  setPerfEnabled(true);
  const fake = installFakeObserver();
  try {
    resetLongTasks();
    const dispose = installPerf();
    assert.equal(fake.observers.length, 1);
    fake.emit([{ duration: 200 }]);
    assert.equal(longTaskStats().count, 1);
    dispose();
    dispose(); // the combined disposer stays idempotent
    assert.equal(fake.observers[0]?.disconnected, true);
    fake.emit([{ duration: 200 }]);
    assert.equal(longTaskStats().count, 1);
  } finally {
    fake.restore();
    setPerfEnabled(false);
    resetLongTasks();
  }
});
