// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The X- app mark: a rounded-square tile carrying the "X" and the dash of the wordmark.
//
// It replaces the sloth mascot at every place the app used to show a character (sidebar
// lockup, sign-in, not-found, training overlay, chat greeting). Drawn as SVG with
// `currentColor`, so it takes its ink from whatever surface it sits on and needs no
// bitmap asset, no fallback path and no retry state — unlike `MascotImg`, whose whole
// job was surviving a 404.
//
// Decorative by default (`aria-hidden`): every place it appears already names the
// product in text beside it, and a duplicate "X-" for a screen reader is noise. Pass
// `title` when it is the only label.

import type { ComponentProps } from "react";

import { cn } from "@/lib/utils";

type XMarkProps = Omit<ComponentProps<"svg">, "viewBox" | "children"> & {
  /** Optional accessible name. Without it the mark is decorative. */
  title?: string;
};

export function XMark({ className, title, ...rest }: XMarkProps) {
  const decorative = title === undefined;
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      role={decorative ? undefined : "img"}
      aria-hidden={decorative ? true : undefined}
      aria-label={title}
      className={cn("shrink-0", className)}
      {...rest}
    >
      {!decorative && <title>{title}</title>}
      {/* Tile: a soft vertical wash rather than a flat fill, so it reads as an app icon
          at 20px and as a surface at 80px. */}
      <defs>
        <linearGradient id="x-mark-tile" x1="24" y1="0" x2="24" y2="48" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="currentColor" stopOpacity="0.16" />
          <stop offset="1" stopColor="currentColor" stopOpacity="0.06" />
        </linearGradient>
      </defs>
      <rect x="0.75" y="0.75" width="46.5" height="46.5" rx="13" fill="url(#x-mark-tile)" />
      <rect
        x="0.75"
        y="0.75"
        width="46.5"
        height="46.5"
        rx="13"
        stroke="currentColor"
        strokeOpacity="0.14"
        strokeWidth="1.5"
      />
      {/* The X: two strokes with rounded caps, slightly smaller than the tile. */}
      <path
        d="M16.5 15.5 31.5 32.5M31.5 15.5 16.5 32.5"
        stroke="currentColor"
        strokeOpacity="0.92"
        strokeWidth="3.1"
        strokeLinecap="round"
      />
      {/* The dash of "X-", set low-right where the wordmark puts it. */}
      <path
        d="M30.5 35.5h5"
        stroke="currentColor"
        strokeOpacity="0.92"
        strokeWidth="3.1"
        strokeLinecap="round"
      />
    </svg>
  );
}
