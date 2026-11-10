// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import {
  type ComputeTargetsResponse,
  NPU_MODE_UNAVAILABLE,
  describeReasons,
} from "./types";

export interface ComputeCapabilityCardProps {
  data: ComputeTargetsResponse | null;
}

function availability(available: boolean, known: boolean): string {
  if (!known) return "Unknown";
  return available ? "Available" : "Not available";
}

/**
 * "What can this machine do?", in prose. Every reason token the backend sends
 * becomes a sentence here; that translation is this card's whole job.
 */
export function ComputeCapabilityCard({ data }: ComputeCapabilityCardProps) {
  if (!data) return null;
  const resolved = data.targets.find((target) => target.id === data.resolved);
  const headline = resolved
    ? `${resolved.name} is the target training will use.`
    : `Training will use ${data.resolved}.`;
  const precision = resolved?.recommended.dtype ?? "unknown";
  const fourBit = availability(
    resolved?.capabilities.bnb4bit ?? false,
    resolved !== undefined,
  );
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border/60 p-3">
      <p className="text-ui-12 font-medium text-foreground">{headline}</p>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-ui-11">
        <div className="flex min-w-0 flex-col">
          <dt className="text-muted-foreground">Training on</dt>
          <dd className="truncate text-foreground">
            {resolved?.name ?? data.resolved}
          </dd>
        </div>
        <div className="flex min-w-0 flex-col">
          <dt className="text-muted-foreground">Recommended precision</dt>
          <dd className="truncate text-foreground">{precision}</dd>
        </div>
        <div className="flex min-w-0 flex-col">
          <dt className="text-muted-foreground">4-bit training</dt>
          <dd className="truncate text-foreground">{fourBit}</dd>
        </div>
      </dl>
      {resolved && resolved.reasons.length > 0 && (
        <p className="text-ui-10p5 text-muted-foreground">
          {describeReasons(resolved.reasons)}
        </p>
      )}
      <p className="text-ui-10p5 text-muted-foreground">{data.budget.notes}</p>
      <ul className="flex flex-col gap-0.5">
        {data.targets.map((target) => (
          <li
            key={target.id}
            className="flex flex-wrap items-baseline gap-x-1.5 text-ui-11"
          >
            <span
              className={cn(
                "shrink-0 font-medium",
                target.supported ? "text-foreground" : "text-muted-foreground",
              )}
            >
              {target.supported ? "Supported" : "Not supported"}
            </span>
            <span className="text-muted-foreground">{target.name}</span>
            {target.supported ? null : (
              <span className="text-muted-foreground/80">
                {describeReasons(target.reasons)}
              </span>
            )}
          </li>
        ))}
      </ul>
      {data.npu.mode !== NPU_MODE_UNAVAILABLE && (
        <p className="text-ui-10p5 text-muted-foreground">{data.npu.notes}</p>
      )}
    </div>
  );
}
