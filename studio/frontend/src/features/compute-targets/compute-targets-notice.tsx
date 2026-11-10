// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type ComputeTargetsNoticeState = "loading" | "error" | "none";

export interface ComputeTargetsNoticeInput {
  error: string | null;
  loading: boolean;
  /** Whether a device list is already on screen. */
  hasData: boolean;
}

/**
 * What the notice has to say. A list already on screen owns its own messaging (the picker's
 * stale-entry line), and a failed read outranks a retry in flight, so pressing "Try again"
 * keeps the message and its button instead of flashing back to "checking".
 */
export function noticeState({
  error,
  loading,
  hasData,
}: ComputeTargetsNoticeInput): ComputeTargetsNoticeState {
  if (hasData) {
    return "none";
  }
  if (error) {
    return "error";
  }
  if (loading) {
    return "loading";
  }
  return "none";
}

export interface ComputeTargetsNoticeProps {
  error: string | null;
  loading: boolean;
  onRetry: () => void;
  /** Set when the surface below already has a list, so the notice stays out of its way. */
  hasData?: boolean;
  className?: string;
}

/**
 * The one place the "Run on" card explains itself when it has nothing to list: a probe in
 * flight, or a failed read the user can retry without reloading the page.
 */
export function ComputeTargetsNotice({
  error,
  loading,
  onRetry,
  hasData = false,
  className = "",
}: ComputeTargetsNoticeProps) {
  const state = noticeState({ error, loading, hasData });
  if (state === "none") {
    return null;
  }
  if (state === "loading") {
    return (
      // biome-ignore lint/a11y/useSemanticElements: a live status line, not a form result.
      <p
        role="status"
        className={cn("text-ui-10p5 text-muted-foreground", className)}
      >
        Checking which devices this machine can train on…
      </p>
    );
  }
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-wrap items-center gap-2 rounded-lg bg-destructive/10 p-2.5 text-ui-11 text-destructive",
        className,
      )}
    >
      <span className="min-w-0 flex-1">{error}</span>
      <Button size="sm" variant="outline" onClick={onRetry}>
        Try again
      </Button>
    </div>
  );
}
