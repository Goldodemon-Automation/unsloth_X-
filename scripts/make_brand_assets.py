#!/usr/bin/env python3
"""Draw the fork's mark and write every raster size the app and the desktop shell use.

The mark is the stacked X- : an X above a dash, on a black rounded plate, with the X in a blue
gradient and the dash in a lighter one. It is drawn here rather than traced from the SVG so the
rasters are reproducible from one script and one set of numbers -- the SVG in ``images/`` carries the
same geometry, and both files say so.

Geometry, in the 1024-unit canvas the raster starts from (the SVG's 128-unit viewBox scaled by 8):

    plate    rounded rect 1024, corner radius 256          #000000
    X        (376,280)-(648,552) and (648,280)-(376,552)   stroke 96, round caps
    dash     (368,736)-(656,736)                           stroke 96, round caps

Run from the repository root:  python scripts/make_brand_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PUBLIC = ROOT / "studio" / "frontend" / "public"
TAURI_ICONS = ROOT / "studio" / "src-tauri" / "icons"

# Supersample, then downscale once. Every raster below comes off this one image, so a 16-pixel icon
# and a 512-pixel store image cannot disagree about the shape.
BASE = 1024
SS = 2  # draw at 2048, downscale to 1024

PLATE = (0, 0, 0, 255)
X_FROM, X_TO = (0x1D, 0x4E, 0xD8), (0x60, 0xA5, 0xFA)
DASH_FROM, DASH_TO = (0x38, 0xBD, 0xF8), (0x7D, 0xD3, 0xFC)

X_ARMS = (((376, 280), (648, 552)), ((648, 280), (376, 552)))
DASH = ((368, 736), (656, 736))
STROKE = 96
CORNER = 256


def _scaled(value: int) -> int:
    return value * SS


def _diagonal(size: int, start: tuple, end: tuple) -> Image.Image:
    """A linear gradient along the top-left to bottom-right diagonal."""
    y, x = np.mgrid[0:size, 0:size]
    t = ((x + y) / (2.0 * (size - 1))).astype(np.float32)
    flat = [np.float32(0.0), np.float32(1.0)]
    ramp = np.clip((t - flat[0]) / (flat[1] - flat[0]), 0.0, 1.0)[..., None]
    channels = [
        (start[index] + (end[index] - start[index]) * ramp).astype(np.uint8) for index in range(3)
    ]
    alpha = np.full((size, size, 1), 255, dtype = np.uint8)
    return Image.fromarray(np.concatenate(channels + [alpha], axis = 2), mode = "RGBA")


def _stroke_mask(size: int, segments: tuple, width: int) -> Image.Image:
    """Thick round-capped strokes: a wide line plus a disc at each end, which is what a round cap is."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    half = width // 2
    for start, end in segments:
        draw.line(
            [_scaled(start[0]), _scaled(start[1]), _scaled(end[0]), _scaled(end[1])],
            fill = 255,
            width = _scaled(width),
        )
        for point in (start, end):
            cx, cy = _scaled(point[0]), _scaled(point[1])
            draw.ellipse(
                [cx - _scaled(half), cy - _scaled(half), cx + _scaled(half), cy + _scaled(half)],
                fill = 255,
            )
    return mask


def _plate_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius = _scaled(radius), fill = 255
    )
    return mask


def render_icon() -> Image.Image:
    """The 1024-pixel master: black plate, gradient X, lighter gradient dash."""
    size = _scaled(BASE)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    plate = Image.new("RGBA", (size, size), PLATE)
    canvas.paste(plate, (0, 0), _plate_mask(size, CORNER))

    canvas.paste(_diagonal(size, X_FROM, X_TO), (0, 0), _stroke_mask(size, X_ARMS, STROKE))
    canvas.paste(_diagonal(size, DASH_FROM, DASH_TO), (0, 0), _stroke_mask(size, (DASH,), STROKE))

    return canvas.resize((BASE, BASE), Image.LANCZOS)


def _write(image: Image.Image, path: Path, size: int) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    resized = image.resize((size, size), Image.LANCZOS)
    resized.save(path, format = "PNG", optimize = True)
    print(f"  {path.relative_to(ROOT)}  {size}x{size}")


def main() -> int:
    master = render_icon()

    _write(master, ROOT / "images" / "logo-x-minus-icon.png", 512)

    print("frontend/public:")
    _write(master, FRONTEND_PUBLIC / "favicon.png", 32)
    _write(master, FRONTEND_PUBLIC / "pwa-192.png", 192)
    _write(master, FRONTEND_PUBLIC / "rounded-512.png", 512)
    # The sidebar renders this one inside `rounded-full`, so a square plate reads as a circle.
    _write(master, FRONTEND_PUBLIC / "circle-logo-small.png", 128)
    # Startup and update screens, plus the browser's new-tab shortcut icon.
    _write(master, FRONTEND_PUBLIC / "sticker.png", 256)
    # The hub's owner avatar and other round placements.
    _write(master, FRONTEND_PUBLIC / "rounded.png", 256)
    (FRONTEND_PUBLIC / "favicon.svg").write_text(
        (ROOT / "images" / "logo-x-minus-icon.svg").read_text(encoding = "utf-8"), encoding = "utf-8"
    )
    print(f"  {Path('studio/frontend/public/favicon.svg')}")

    print("src-tauri/icons:")
    for name, size in (
        ("32x32.png", 32),
        ("128x128.png", 128),
        ("128x128@2x.png", 256),
        ("icon.png", 512),
    ):
        _write(master, TAURI_ICONS / name, size)
    print("src-tauri/icons/macos:")
    for name, size in (("32x32.png", 32), ("128x128.png", 128)):
        _write(master, TAURI_ICONS / "macos" / name, size)
    print("src-tauri/icons/linux:")
    for size in (16, 22, 24, 32, 48, 64, 96, 128, 256, 512):
        _write(master, TAURI_ICONS / "linux" / f"{size}x{size}.png", size)

    # Multi-resolution .ico: Windows picks the closest size per context. .icns needs macOS tooling
    # (iconutil) and is left alone rather than written as an invalid file.
    ico = TAURI_ICONS / "icon.ico"
    master.resize((256, 256), Image.LANCZOS).save(
        ico, format = "ICO", sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )
    print(f"  {ico.relative_to(ROOT)}  multi-size")
    (FRONTEND_PUBLIC / "unsloth.ico").write_bytes(ico.read_bytes())
    print("  studio/frontend/public/unsloth.ico  (copy of icon.ico)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
