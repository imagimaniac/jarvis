#!/usr/bin/env python3
"""Generate the Jarvis PWA icons.

Written against the standard library (``zlib`` + ``struct``) rather than an
imaging package: adding Pillow purely to draw four flat-colour circles would be
a heavy dependency for a local-first app that otherwise needs nothing beyond
FastAPI.

Rendering is supersampled 4x and box-filtered down, which is what keeps the
ring edges smooth without needing a real anti-aliasing library.

    python3 scripts/gen_icons.py
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static"

BACKGROUND = (0, 0, 0)        # #000000, matches the manifest background
ACCENT = (59, 130, 246)       # #3b82f6, the existing Jarvis button blue

SS = 4  # supersampling factor

# Mark geometry, in units of the icon's half-width, measured from the centre.
RING_OUTER = 0.34
RING_INNER = 0.235
DOT_RADIUS = 0.095


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_png(path: Path, width: int, height: int, rgb_rows: list[bytes]) -> None:
    """Write 8-bit truecolour (no alpha) PNG. All icons here are fully opaque."""
    raw = b"".join(b"\x00" + row for row in rgb_rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def _render(size: int, scale: float) -> list[bytes]:
    """Return `size` scanlines of RGB bytes for the Jarvis mark.

    `scale` shrinks the artwork about the centre, for the maskable variant that
    has to survive an aggressive circular crop.
    """
    big = size * SS
    cx = cy = big / 2.0
    ring_out = RING_OUTER * big * scale
    ring_in = RING_INNER * big * scale
    dot = DOT_RADIUS * big * scale

    # Render at SS resolution first: a pixel is accent if its centre falls in the
    # ring band or the centre dot.
    rows: list[bytearray] = []
    for y in range(big):
        row = bytearray()
        dy = (y + 0.5) - cy
        for x in range(big):
            dx = (x + 0.5) - cx
            d2 = dx * dx + dy * dy
            inside = (ring_in * ring_in <= d2 <= ring_out * ring_out) or (d2 <= dot * dot)
            row += bytes(ACCENT if inside else BACKGROUND)
        rows.append(row)

    # Box-filter SS x SS blocks down to one pixel.
    out: list[bytes] = []
    area = SS * SS
    for y in range(size):
        acc = bytearray()
        for x in range(size):
            r = g = b = 0
            for dy in range(SS):
                src = rows[y * SS + dy]
                base = x * SS * 3
                for dx in range(SS):
                    i = base + dx * 3
                    r += src[i]
                    g += src[i + 1]
                    b += src[i + 2]
            acc += bytes((r // area, g // area, b // area))
        out.append(bytes(acc))
    return out


def build(name: str, size: int, scale: float = 1.0) -> None:
    path = STATIC / name
    write_png(path, size, size, _render(size, scale))
    print(f"wrote {path.relative_to(STATIC.parent)} ({size}x{size})")


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    build("icon-192.png", 192)
    build("icon-512.png", 512)
    build("apple-touch-icon.png", 180)
    # Maskable icons get cropped to a circle inscribed in the middle 80%, so
    # the artwork is scaled well inside that safe zone.
    build("maskable-512.png", 512, scale=0.75)


if __name__ == "__main__":
    main()