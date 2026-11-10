// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useRef } from "react";
import {
  type ComputeTarget,
  type NpuTrainingCapability,
  computeTargetKindLabel,
  describeReasons,
  formatMemoryBytes,
  isNpuTarget,
  isTargetSelectable,
} from "./types";

export interface ComputeTargetPickerProps {
  targets: readonly ComputeTarget[];
  value: string | null;
  onChange: (id: string) => void;
  /** The NPU's training capability: an unavailable NPU is listed but cannot be picked. */
  npu?: NpuTrainingCapability | null;
  disabled?: boolean;
  /** The radiogroup's accessible name. */
  label?: string;
  className?: string;
}

const ROW_CLASS =
  "flex w-full items-center gap-2 rounded-md border px-2.5 py-1.5 text-left text-ui-12 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";
const SELECTED_CLASS = "border-border bg-accent text-foreground";
const IDLE_CLASS =
  "border-border/60 text-foreground hover:bg-accent/60 disabled:hover:bg-transparent";
const BLOCKED_CLASS =
  "cursor-not-allowed border-border/40 text-muted-foreground opacity-60";
const SHORT_VALUE_CLASS =
  "max-w-[9rem] truncate text-ui-10 tabular-nums text-muted-foreground";

/** Which way an arrow key moves the selection, or null for every other key. */
function arrowStep(key: string): number | null {
  if (key === "ArrowDown" || key === "ArrowRight") return 1;
  if (key === "ArrowUp" || key === "ArrowLeft") return -1;
  return null;
}

/**
 * Why a row cannot be picked. An unavailable NPU says why with its own reasons,
 * every other device with the reasons the backend gave for it.
 */
function blockedReason(
  target: ComputeTarget,
  npu: NpuTrainingCapability | null,
): string {
  // A device the backend refuses speaks with its own reasons; a device it accepts
  // is only blocked when the NPU's runtime is unavailable, which the NPU says.
  if (target.supported) return describeReasons(npu?.reasons ?? []);
  return describeReasons(target.reasons);
}

interface ComputeTargetRowProps {
  target: ComputeTarget;
  npu: NpuTrainingCapability | null;
  checked: boolean;
  tabIndex: number;
  /** The row's own eligibility, ignoring the picker being disabled. */
  pickable: boolean;
  disabled: boolean;
  registerRow: (id: string, node: HTMLButtonElement | null) => void;
  onSelect: (id: string) => void;
  onMove: (id: string, step: number) => void;
}

function ComputeTargetRow({
  target,
  npu,
  checked,
  tabIndex,
  pickable,
  disabled,
  registerRow,
  onSelect,
  onMove,
}: ComputeTargetRowProps) {
  const explanation = pickable ? "" : blockedReason(target, npu);
  const note = isNpuTarget(target) ? npu?.notes : undefined;
  const title = pickable ? undefined : explanation || undefined;
  return (
    <button
      ref={(node) => registerRow(target.id, node)}
      // A row is one focusable control carrying a name, a kind badge and a memory
      // figure, which a native radio input cannot hold.
      // biome-ignore lint/a11y/useSemanticElements: a radiogroup row is a button here.
      type="button"
      role="radio"
      aria-checked={checked}
      aria-disabled={disabled}
      disabled={disabled}
      title={title}
      tabIndex={tabIndex}
      onClick={() => onSelect(target.id)}
      onKeyDown={(event) => {
        const step = arrowStep(event.key);
        if (step === null) return;
        event.preventDefault();
        onMove(target.id, step);
      }}
      className={cn(
        ROW_CLASS,
        checked ? SELECTED_CLASS : IDLE_CLASS,
        disabled && BLOCKED_CLASS,
      )}
    >
      <span className="flex min-w-0 flex-col">
        <span className="truncate font-medium">{target.name}</span>
        {note && (
          <span className="truncate text-ui-10p5 font-normal text-muted-foreground">
            {note}
          </span>
        )}
      </span>
      <span className="ml-auto flex shrink-0 items-center gap-1.5">
        <span className="rounded-[6px] border border-border/60 px-1.5 py-0.5 text-ui-10 font-medium leading-none text-muted-foreground">
          {computeTargetKindLabel(target)}
        </span>
        <span className={SHORT_VALUE_CLASS} title={title}>
          {pickable
            ? formatMemoryBytes(target.memory_bytes)
            : explanation || "Not available"}
        </span>
      </span>
    </button>
  );
}

export function ComputeTargetPicker({
  targets,
  value,
  onChange,
  npu = null,
  disabled = false,
  label = "Compute target",
  className = "",
}: ComputeTargetPickerProps) {
  // One ref per row: an arrow key moves focus, not just the selection.
  const rows = useRef(new Map<string, HTMLButtonElement>());
  const pickable: string[] = targets
    .filter((target) => isTargetSelectable(target, npu))
    .map((target) => target.id);
  // Roving focus: the selected row, or the first row that can be picked.
  const activeId =
    value !== null && pickable.includes(value) ? value : (pickable[0] ?? null);

  function registerRow(id: string, node: HTMLButtonElement | null): void {
    if (node === null) {
      rows.current.delete(id);
      return;
    }
    rows.current.set(id, node);
  }

  function move(fromId: string, step: number): void {
    const index = pickable.indexOf(fromId);
    if (index < 0 || pickable.length === 0) return;
    const next = pickable[(index + step + pickable.length) % pickable.length];
    onChange(next);
    rows.current.get(next)?.focus();
  }

  if (targets.length === 0) {
    return (
      <p className={cn("text-ui-10p5 text-muted-foreground", className)}>
        No compute devices were reported for this machine.
      </p>
    );
  }

  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn("flex flex-col gap-1", className)}
    >
      {targets.map((target) => {
        const rowPickable = isTargetSelectable(target, npu);
        return (
          <ComputeTargetRow
            key={target.id}
            target={target}
            npu={npu}
            checked={value === target.id}
            tabIndex={activeId === target.id ? 0 : -1}
            pickable={rowPickable}
            disabled={disabled || !rowPickable}
            registerRow={registerRow}
            onSelect={onChange}
            onMove={move}
          />
        );
      })}
    </div>
  );
}
