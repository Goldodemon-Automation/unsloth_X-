// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Curated set of profile pictures offered by the Profile settings page.
//
// These used to be sloth stickers from `public/Sloth emojis`. The product no longer ships a
// character, so the set is six neutral vector tiles (`public/avatars/`): duotone gradients with
// a geometric glyph, each drawn square and centred, so they fill a round 32px avatar frame
// without the cropping the PNG set needed a hand-curated subset to avoid.
//
// Paths are relative to the public folder; resolve with `publicAssetUrl(...)` before using as
// an <img> src, so spaces and subpath deploys are handled.
export const PROFILE_AVATARS: readonly string[] = [
  "avatars/a1.svg",
  "avatars/a2.svg",
  "avatars/a3.svg",
  "avatars/a4.svg",
  "avatars/a5.svg",
  "avatars/a6.svg",
];

// Kept so an import that still names the old export keeps compiling; it points at the
// current set rather than a copy, so the two cannot drift.
export const SLOTH_AVATARS = PROFILE_AVATARS;
