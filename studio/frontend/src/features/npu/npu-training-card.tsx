// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  type NpuAssistRequested,
  type NpuTrainingAssist,
  assistRefusalNote,
  assistSummaryLines,
} from "./npu-training-assist";

/**
 * The panel's card surface, written as one literal here and in `npu-setup-notice.tsx`.
 * `tests/npu-panel-consistency.test.ts` fails if the two stop matching: the notice and this
 * card sit on the same screens, so one of them aging apart reads as a bug in the panel.
 * Tokens only, so the card takes the theme instead of assuming a light one.
 */
const CARD_CLASS = "rounded-lg border border-border/60 bg-muted/50 px-2.5 py-3";
/** One focus ring for every button in the panel, visible on keyboard focus in both themes. */
const FOCUS_RING_CLASS = "focus-visible:ring-2 focus-visible:ring-ring";

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
      className={cn("flex flex-col gap-2 text-xs", CARD_CLASS)}
      aria-busy={loading}
    >
      <h3 className="text-sm font-medium text-foreground">
        NPU acceleration for training
      </h3>
      {error ? (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      ) : null}
      {!error && loading && lines.length === 0 ? (
        // biome-ignore lint/a11y/useSemanticElements: a live status line, not a form result.
        <p role="status" className="text-muted-foreground">
          Checking your NPU…
        </p>
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
            className={FOCUS_RING_CLASS}
            aria-pressed={enabled}
            disabled={loading}
            onClick={() => onRequest?.("assist")}
          >
            Use the NPU where it helps
          </Button>
          <Button
            size="sm"
            variant={enabled ? "outline" : "default"}
            className={FOCUS_RING_CLASS}
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
