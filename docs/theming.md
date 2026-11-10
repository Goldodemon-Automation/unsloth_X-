# Theming and the accent color

The UI has four independent theming dimensions. The tokens themselves live in
`studio/frontend/src/index.css`; `<html class="dark">` and `<html data-palette>`
are applied by `features/settings/stores/theme-store.ts`, and the contrast slider
and the custom accent by `features/settings/stores/appearance-custom-store.ts`:

| Dimension | Where it lives | Values |
| --- | --- | --- |
| Mode | `<html class="dark">` | light, dark |
| Palette | `<html data-palette>` | *unset* (Standard), `classic`, `minimal`, plus the flavor themes in `features/settings/lib/color-themes.ts` |
| Contrast | `<html data-contrast-adjust>` | slider, mixes tokens toward the ink |
| Custom accent | `ACCENT_VARS` | any `#rrggbb` the user picks |

## The accent is #9ceaf0

The Standard palette's brand accent used to be Unsloth green (`#17b88b`). It is
now **`#9ceaf0`**, a light cyan, in both modes:

```css
--primary: #9ceaf0;
--primary-foreground: #0f1114;
```

The foreground is the dark ink rather than white on purpose. `#9ceaf0` is light:
white on it scores about **1.3:1**, far under the 4.5:1 AA floor, while
`#0f1114` on it scores about **15:1**. Every primary button, active control and
switch track therefore reads with dark text in both light and dark mode.

The sidebar gets the same pair, but as literals: `--sidebar-primary: #9ceaf0` and
`--sidebar-primary-foreground: #0f1114` are written out in the `:root` block *and*
again in the `.dark` block, not derived with `var(--primary)`. Change one and the
other stays behind, so both blocks are on the edit list below. `--control-accent`
*is* derived (`--control-accent: var(--primary)` in `:root`, and Classic's
`#339cff` / `#4dabff` are the only overrides), so in Standard and Minimal the
toggles, badges and chips follow the accent on their own.

A custom accent is narrower than the base tokens: `ACCENT_VARS` is
`["--control-accent", "--chart-1", "--primary"]`, so the user's pick — passed
through `legibleAccent()` against the page and elevated surfaces — lands on those
three, with `readableForeground()` choosing the ink, and leaves `--sidebar-primary`
on `#9ceaf0`. The flavor palettes do wire the sidebar up
(`--sidebar-primary: var(--th-accent)` in the flavor block); Classic and Minimal
write their own sidebar literals.

The Standard entry in `COLOR_THEMES`
(`features/settings/lib/color-themes.ts`) carries the same value, which is what
the appearance picker shows as the default swatch.

### What is still green, and why

Two greens are deliberately untouched:

- `--verified: #17b88b` — the status color for verified badges and token
  indicators. Status is a signal, not a theme color: it stays put in every
  palette and mode (see `tests/appearance-accent-vars.test.ts`).
- The chart series (`--chart-1` … `--chart-5`) — data colors. A light cyan first
  series would be hard to read on a white chart surface, so the series keep
  their hues. Note that a *custom* accent does overwrite `--chart-1` (it is in
  `ACCENT_VARS`), so the two only diverge for the default palette.

### Consequence to know about

`text-primary` is used for links and active labels (165 uses across the `.tsx`
files under `studio/frontend/src`). At `#9ceaf0` on the light page those gain very
little contrast, which is the trade-off of a pale accent. If accent-colored text
needs to stay readable,
either darken the accent for text only (a `--primary-ink` mixed toward
`--foreground`) or keep links on `--foreground` with an accent underline.

## The mark and the accent

The fork's mark and the UI accent are two deliberately different blues.

| Thing | Where it is defined | Value |
| --- | --- | --- |
| Mark, X | `#xBlue` in `images/logo-x-minus-icon.svg`; `X_FROM`, `X_TO` in `scripts/make_brand_assets.py` | `#1D4ED8` → `#60A5FA` |
| Mark, dash | `#dashBlue` (plus a solid `#38BDF8` underlay path) in the SVG; `DASH_FROM`, `DASH_TO` in the script | `#38BDF8` → `#7DD3FC` |
| Mark, plate | `fill="#000000"` on the rounded rect; `PLATE` in the script | `#000000` |
| UI accent | `--primary` and `--sidebar-primary` in `src/index.css` (`--control-accent` derives from `--primary`) | `#9ceaf0` |

Those are one set of numbers in two notations: the SVG writes them as
`stop-color="#1D4ED8"` and friends, the script writes the same bytes as triples
(`X_FROM, X_TO = (0x1D, 0x4E, 0xD8), (0x60, 0xA5, 0xFA)`, `DASH_FROM, DASH_TO =
(0x38, 0xBD, 0xF8), (0x7D, 0xD3, 0xFC)`, `PLATE = (0, 0, 0, 255)`). If you change
the blue, change both files; the geometry is already shared — the script's
1024-unit canvas is the SVG's 128-unit viewBox times 8 (`STROKE = 96` there, a
12-unit `stroke-width` here), which is what keeps the rasters and the vector from
disagreeing.

The mark is darker and far more saturated than the accent because the two have
opposite jobs. The accent is a **fill that carries dark ink**, so its contrast is
spent on `--primary-foreground` sitting on top of it, which is exactly what forces
it pale. The mark is a **thin stroke on a black plate** with no label to read, and
at 16px that stroke is `stroke-width="12"` in a 128-unit viewBox — 1.5px of line.
Raw luminance against the plate is not what a pale accent would lack there: with
the WCAG formula that `tests/appearance-accent-vars.test.ts` uses, `#9ceaf0` scores
15.4:1 on `#000000` while the mark's darkest stop `#1D4ED8` scores only 3.1:1. What
a single flat pale tone at 1.5px cannot give is contrast *inside* the glyph. The
gradient supplies it — dark at the top-left of each arm (`#1D4ED8`) running to light
at the bottom-right (`#60A5FA`), and `#38BDF8` → `#7DD3FC` across the dash — so the
two arms read as separate strokes rather than one light blob, and the blue keeps
its hue instead of washing to near-white at favicon size. A darker, higher-chroma
blue is what survives that scale.

They also change through different doors. The accent is CSS: edit `src/index.css`
(and `color-themes.ts`) and the UI repaints on the next build. The mark is an
asset: `images/logo-x-minus-icon.svg` is the vector source, and
`python scripts/make_brand_assets.py` from the repository root regenerates every
raster from the same numbers — `images/logo-x-minus-icon.png`, the favicons and PWA
icons in `studio/frontend/public/` (including `favicon.svg`, which the script
copies verbatim from the SVG), and `studio/src-tauri/icons/`. No palette or accent
token reaches the mark: changing `--primary` does not recolour the logo, and the
logo is not required to match it.

## Where to change the accent

1. `src/index.css` — the `:root` block (light) and the `.dark` block, both
   `--primary` / `--primary-foreground` and `--sidebar-primary` /
   `--sidebar-primary-foreground`.
2. `src/features/settings/lib/color-themes.ts` — the `standard.light.accent` and
   `standard.dark.accent` values.
3. Leave `--verified` and the chart tokens alone unless the change is about
   status or data colors.
4. Do not touch the mark for an accent change. It is a separate pipeline:
   `images/logo-x-minus-icon.svg` plus `python scripts/make_brand_assets.py`.

Afterwards, from `studio/frontend`, run
`node --experimental-strip-types --test tests/appearance-accent-vars.test.ts tests/contrast-slider.test.ts`
(`npm test` globs the whole `tests/**/*.test.ts` set): the first covers
accent-to-foreground contrast, the second fails on any chrome that hardcodes an
alpha wash instead of a themed token.
