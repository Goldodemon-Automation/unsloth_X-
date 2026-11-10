// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  type NpuAssistRequested,
  type NpuTrainingAssist,
  assistRefusalNote,
  assistSummaryLines,
} from "./npu-training-assist";

export interface NpuTrainingCardProps {
  assist: NpuTrainingAssist | null;
  error?: string | null;
  loading?: boolean;
  onRequest?: (requested: NpuAssistRequested) => void;
}

/**
 * Says what the NPU can and cannot do for training, and offers the one click
 * that turns the assist on. It owns no fetching: the parent owns `refresh`.
 */
export function NpuTrainingCard({
  assist,
  error = null,
  loading = false,
  onRequest,
}: NpuTrainingCardProps) {
  if (assist === null && !loading && !error) return null;

  const lines = assist ? assistSummaryLines(assist) : [];
  const enabled = assist?.enabled === true;
  const refusalNote = assistRefusalNote(assist);
  const canAsk =
    assist !== null && assist.mode !== "unavailable" && !!onRequest;

  return (
    <section
      className="flex flex-col gap-2 rounded-lg bg-muted/50 px-2.5 py-3 text-xs"
      aria-busy={loading}
    >
      <h3 className="text-sm font-medium text-foreground">
        NPU acceleration for training
      </h3>
      {error ? <p className="text-destructive">{error}</p> : null}
      {!error && loading && lines.length === 0 ? (
        <p className="text-muted-foreground">Checking your NPU…</p>
      ) : null}
      {!error && lines.length > 0 ? (
        <div className="flex flex-col gap-1">
          {lines.map((line, index) => (
            <p
              key={`${index}-${line}`}
              className={
                index === 0 && enabled
                  ? "text-foreground"
                  : "text-muted-foreground"
              }
            >
              {line}
            </p>
          ))}
        </div>
      ) : null}
      {refusalNote ? (
        <p className="text-muted-foreground">{refusalNote}</p>
      ) : null}
      {canAsk ? (
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant={enabled ? "default" : "outline"}
            aria-pressed={enabled}
            disabled={loading}
            onClick={() => onRequest?.("assist")}
          >
            Use the NPU where it helps
          </Button>
          <Button
            size="sm"
            variant={enabled ? "outline" : "default"}
            aria-pressed={!enabled}
            disabled={loading}
            onClick={() => onRequest?.("off")}
          >
            Leave it off
          </Button>
        </div>
      ) : null}
    </section>
  );
}
