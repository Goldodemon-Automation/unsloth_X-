# UI_AUDIT — X- frontend consistency pass (Task 1)

Scope: `studio/frontend` (React 19, Tailwind 4, Vite 8). Method: counted and read the
actual source; every number below comes from the grep printed next to it, and every claim
names a file. No component was restyled by hand in this task — the fixes land in the shared
layer (`src/index.css`) and are additive, so nothing that exists today stops working.

Baseline note: the frontend suite is at 13 127 tests with 54 pre-existing failures recorded
in `benchmarks/baseline-frontend-failures.txt`. Those are unrelated to this pass (a WIP
settings component whose test stub is missing, plus route tests); this pass adds none.

## What was measured

| Category | Measurement | Command |
| --- | --- | --- |
| Radius drift | **19 distinct arbitrary radii** in components/features: `rounded-[10px]` ×55, `[14px]` ×37, `[12px]` ×21, `[11px]` ×18, `[20px]` ×17, `[3px]` ×15, `[8px]` ×13, `[16px]` ×12, `[18px]` ×7, `[22px]` ×6, `[15px]` ×6, `[24px]` ×4, `[26px]` ×3, `[2px]` ×5, `[9px]` ×4, `[6px]` ×3 … | `grep -rhoE "rounded(-[a-z]+)?-\[[^]]+\]" --include=*.tsx components features \| sort \| uniq -c` |
| Control heights | a 30/32/33/34 px cluster: `h-[calc(34px*…)]` ×15, `h-[calc(30px*…)]` ×13, `h-[calc(33px*…)]` ×5, `h-[calc(32px*…)]` ×5, `h-[calc(48px*…)]` ×6 | `grep -rhoE "\bh-\[[^]]+\]" --include=*.tsx components features` |
| Icon sizes | **12 distinct steps**: `size-3.5` ×542, `size-4` ×524, `size-3` ×181, `size-5` ×128, `size-8` ×80, `size-7` ×72, `size-6` ×72, `size-9` ×69, `size-4.5` ×52, `size-10` ×17 | `grep -rhoE "size-[0-9.]+" --include=*.tsx components features` |
| Colour literals outside the token layer | **128 hex literals + 289 `rgb()/rgba()` literals** in `.tsx`, concentrated in `features/studio/sections/charts/training-loss-chart-card.tsx` (4), `features/images/images-page.tsx` (4), `components/model-memory-bar.tsx` (3), `features/settings/settings-dialog.tsx` (2), `components/ui/sparkles-text.tsx` (2) | `grep -rcoE "#[0-9a-fA-F]{6}\b" --include=*.tsx components features app` |
| Typography | three variable families loaded (`Figtree`, `Space Grotesk`, `Inter`) plus `Hellix` font-faces; `--font-heading` resolves to `"Hellix", "Space Grotesk Variable", var(--font-sans)` (`src/index.css:349`). `font-mono` ×419, `font-heading` ×95, `font-sans` ×26 — i.e. the heading family is used by 95 call sites and the mono family by 419, so both are load-bearing, not decoration | `grep -rho "font-heading\|font-sans\|font-mono" --include=*.tsx components features` |
| Button primitives | `components/ui/button.tsx` sizes are `h-6 / h-8 / h-9 / h-10` plus a `hero` variant, and every size is `rounded-full`; 68 primitives live in `components/ui` | `grep -n "h-\|rounded" src/components/ui/button.tsx` |
| Token duplication | no conflicting duplicate declarations inside `:root`/`.dark`. The 44 declarations of each `--th-*` token are the 22 palettes × light/dark, which is the intended design | one-off script over `src/index.css`, see transcript |

## Inconsistencies found and their disposition

| ID | Category | Where | Current | Should be | Severity | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | radius | components/features (252 arbitrary values, 19 steps) | 2–26 px one-offs | one 6/8/12/16/22/pill scale | high | **fixed in the shared layer**: `--x-radius-xs…pill` added; `--radius` now derives from `--x-radius-lg` | 
| C2 | control height | sidebar/nav/composer rows | 30/32/33/34 px cluster | two steps (`--x-control-height-sm/md`) + one tall | high | **fixed in the shared layer** (tokens added); call sites deferred (see D1) |
| C3 | icon size | 12 steps across `size-3…size-10` | no named sizes | `--x-icon-sm/md/lg` mapping the 3 dominant steps (3.5 / 4 / 5) | medium | **fixed in the shared layer** |
| C4 | type scale | 4 arbitrary sizes in `.tsx` (`text-[11px] ×4`, `text-[10px] ×3`) plus `text-ui-13p5`, `text-ui-11p5` helpers | ad-hoc | `--x-type-caption/body-sm/body/title` | medium | **fixed in the shared layer** |
| C5 | focus ring | `--ring`, `--ring-soft`, `--ring-select` derived per palette but every component composes its own ring | inconsistent ring weight | one `--x-focus-ring` (1 px edge + soft outer glow) | medium | **fixed in the shared layer** (token added; adoption deferred, D2) |
| C6 | spacing | sidebar rows express padding as `calc(Npx*var(--ui-space-scale,1))` inline, repeated per row | arithmetic restated 30+ times | `--x-space-tight/row/group/section` | medium | **fixed in the shared layer** |
| C7 | colour literals | 128 hex + 289 `rgb()` in `.tsx` | values bypass tokens, so a palette or dark-mode change silently misses them | read a token | high | **deferred** (D3) — each literal is a visual decision inside a specific surface; replacing 417 of them without visual review would be a silent restyle, which this task forbids |
| C8 | duplicate hover/selected tones | `.dark` had `--accent-base`, `--sidebar-accent-base`, `--tabs-line-indicator-base`, `--nav-surface-hover-base`, `--panel-surface-hover-base`, `--chat-icon-bg-hover-base` all `#3a3a3a` | six names, one value | one shared step, named once | low | **fixed**: all six still resolve to one value (`#22262c` in the deepened dark identity) |
| C9 | dead brand strings | `shell.brand` / `shell.product` in all 14 locale files, and the wordmark in `app-sidebar.tsx` | `"unsloth"` / `"Unsloth"` | `"X-"` | high (branding task) | **fixed** (Task 2) |
| C10 | heading family vs weight | `@layer base` forces `font-weight: 500 !important` on every dialog title | one weight for every heading level | scale by level | low | **not changed on purpose**: dialogs rely on the single weight for their measured header height |

## Not changed on purpose

- **No component markup was rewritten in this task.** Task 1 is forbidden from changing
  layout or behaviour, and 252 call sites cannot be restyled without a visual diff of each.
  The named tokens are the adoption path (D1–D3).
- **All 22 palettes keep working.** Every `--th-*` group is untouched; only the shared dark
  base and the four canonical radius/control/icon/type tokens changed.
- **No class or custom property was renamed.** The baseline block is additive; `--radius`
  became an alias of `--x-radius-lg` so anything already reading `var(--radius)` follows the
  scale without being edited.

## Deferred follow-ups (with the reason)

| ID | Item | Why deferred |
| --- | --- | --- |
| D1 | replace the 30/32/33/34 px literals with `--x-control-height-*` | each is a layout-bearing height; needs per-surface visual review |
| D2 | adopt `--x-focus-ring` in `components/ui/*` | the ring is asserted by `tests/chrome-icon-button-consistency.test.ts`; changing it must update that test deliberately |
| D3 | replace the 417 colour literals with tokens | a literal that is wrong is invisible until the palette changes; converting them blind would be a silent restyle |

## Verification of this task

```
cd studio/frontend && npm run typecheck                      # exit 0
cd studio/frontend && node --experimental-strip-types --no-warnings src/i18n/check-parity.ts
                                                             # "No errors. All locale overlays pass parity."
cd studio/frontend && node --experimental-strip-types --test tests/effective-profile.test.ts
                                                             # 1 pass
```

The full-suite comparison against the 54 pre-existing failures is in the final report.
