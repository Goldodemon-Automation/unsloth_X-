// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatBytes } from "@/features/hub";
import { cn } from "@/lib/utils";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ArtifactDiff, PinnedStatus } from "../api";

function params(value: number | null): string {
  if (!value) return "—";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return `${Math.round(value)}`;
}

function value(raw: unknown): string {
  if (raw === null || raw === undefined) return "—";
  if (typeof raw === "boolean") return raw ? "yes" : "no";
  if (typeof raw === "number") return String(raw);
  if (typeof raw === "string") return raw;
  try {
    return JSON.stringify(raw);
  } catch {
    return String(raw);
  }
}

function factValue(field: string, raw: unknown): string {
  if (field === "weight_bytes") return formatBytes(Number(raw) || 0);
  if (field === "params") {
    return typeof raw === "number" ? params(raw) : "—";
  }
  return value(raw);
}

function deltaLabel(delta: number | null): string {
  if (delta === null) return "—";
  const points = delta * 100;
  return `${points > 0 ? "+" : ""}${points.toFixed(1)} pts`;
}

const PINNED_LABEL: Record<PinnedStatus, string> = {
  pinned: "pinned baseline",
  regressed: "regressed vs pinned",
  improved: "improved vs pinned",
  mixed: "mixed vs pinned",
  flat: "within noise of pinned",
  unscored: "no suite in common with the pin",
  missing: "the pinned record was deleted",
  unpinned: "no pinned baseline",
};

function pinnedClass(status: PinnedStatus): string {
  if (status === "regressed") return "text-destructive";
  if (status === "improved") return "text-emerald-600 dark:text-emerald-400";
  return "text-muted-foreground";
}

/**
 * Two checkpoints side by side: the fields that moved, the facts that differ,
 * and the benchmark each side carries. Every row here comes from the manifests
 * the runs wrote, so a difference on screen is a difference on disk.
 */
export function RunDiff({
  diff,
  onClose,
}: {
  diff: ArtifactDiff;
  onClose: () => void;
}) {
  const changedFacts = diff.facts.filter((fact) => fact.changed);
  const groups = new Map<string, typeof diff.changes>();
  for (const change of diff.changes) {
    const bucket = groups.get(change.group) ?? [];
    bucket.push(change);
    groups.set(change.group, bucket);
  }
  const benchmarkDelta = diff.benchmark.delta ?? [];

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-border/60 p-4">
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-foreground">
            What changed between these two runs
          </p>
          <p className="mt-0.5 font-mono text-ui-10 break-all text-muted-foreground">
            {diff.left.name} → {diff.right.name}
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Badge variant="outline" className="text-ui-10">
            {diff.changed_count} changed
          </Badge>
          <Badge variant="outline" className="text-ui-10">
            {diff.same_count} same
          </Badge>
          <Button size="sm" variant="ghost" onClick={onClose}>
            <HugeiconsIcon icon={ArrowLeft02Icon} className="size-3.5" />
            Close
          </Button>
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {diff.facts.map((fact) => (
          <div
            key={fact.field}
            className={cn(
              "rounded-lg border px-3 py-2",
              fact.changed
                ? "border-primary/40 bg-primary/5"
                : "border-border/50 bg-background/40",
            )}
          >
            <p className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {fact.label}
            </p>
            <p className="mt-0.5 flex flex-wrap items-baseline gap-1.5 font-mono text-ui-11">
              <span className="text-muted-foreground">
                {factValue(fact.field, fact.left)}
              </span>
              <span className="text-muted-foreground/60">→</span>
              <span
                className={cn(
                  fact.changed
                    ? "font-semibold text-foreground"
                    : "text-foreground/80",
                )}
              >
                {factValue(fact.field, fact.right)}
              </span>
            </p>
          </div>
        ))}
      </div>

      {diff.changes.length === 0 ? (
        <p className="text-ui-11 text-muted-foreground">
          No manifest field differs
          {changedFacts.length === 0
            ? ", and the checkpoint facts match."
            : "; only the facts above differ."}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {[...groups.entries()].map(([group, changes]) => (
            <div
              key={group}
              className="overflow-hidden rounded-lg border border-border/60"
            >
              <p className="border-b border-border/50 bg-muted/30 px-3 py-1.5 text-ui-10 font-medium uppercase tracking-[0.05em] text-muted-foreground">
                {group}
              </p>
              <table className="w-full table-fixed text-ui-11">
                <tbody>
                  {changes.map((change) => (
                    <tr
                      key={change.field}
                      className="border-b border-border/40 last:border-b-0"
                    >
                      <td className="w-2/5 px-3 py-1.5 align-top font-mono text-ui-10 break-all text-muted-foreground">
                        {/* The group header already says the prefix. */}
                        {change.field.startsWith(`${group}.`)
                          ? change.field.slice(group.length + 1)
                          : change.field}
                      </td>
                      <td className="px-3 py-1.5 align-top font-mono text-ui-10 break-all text-muted-foreground">
                        {value(change.left)}
                      </td>
                      <td className="px-3 py-1.5 align-top font-mono text-ui-10 break-all font-medium text-foreground">
                        {value(change.right)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
          {diff.truncated ? (
            <p className="text-ui-10 text-amber-600 dark:text-amber-400">
              Only the first changes are listed; the manifests differ in more
              fields than the page shows.
            </p>
          ) : null}
        </div>
      )}

      {benchmarkDelta.length > 0 ? (
        <div className="flex flex-col gap-1.5 rounded-lg border border-border/60 p-3">
          <p className="text-ui-11 font-medium text-foreground">
            Benchmark, left → right
            {diff.benchmark.aggregate_delta !== null ? (
              <span className="ml-2 font-mono text-ui-10 text-muted-foreground">
                {deltaLabel(diff.benchmark.aggregate_delta)} overall
              </span>
            ) : null}
          </p>
          <table className="w-full text-ui-11">
            <tbody>
              {benchmarkDelta.map((entry) => (
                <tr key={entry.suite} className="border-t border-border/40">
                  <td className="py-1 font-mono text-ui-10">{entry.suite}</td>
                  <td className="py-1 text-right font-mono">
                    {entry.left === null ? "—" : (entry.left * 100).toFixed(1)}
                  </td>
                  <td className="py-1 text-right font-mono">
                    {entry.right === null
                      ? "—"
                      : (entry.right * 100).toFixed(1)}
                  </td>
                  <td className="py-1 text-right font-mono">
                    {deltaLabel(entry.delta)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-ui-10">
            {[diff.benchmark.left, diff.benchmark.right].map((side, index) =>
              side ? (
                <span key={side.id} className="text-muted-foreground">
                  {index === 0 ? "left" : "right"}:{" "}
                  <span className={pinnedClass(side.vs_pinned.status)}>
                    {PINNED_LABEL[side.vs_pinned.status]}
                  </span>
                </span>
              ) : null,
            )}
          </div>
        </div>
      ) : null}

      {diff.notes.length > 0 ? (
        <ul className="flex flex-col gap-0.5 text-ui-10 leading-relaxed text-muted-foreground">
          {diff.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
