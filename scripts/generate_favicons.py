#!/usr/bin/env python3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = "#121110"
FOREGROUND = "#f8f6f3"
ACCENT = "#f0884a"

VIEWBOX = 64
CORNER = 14
GLYPH = "m"
GLYPH_SIZE = 40
GLYPH_CENTER = (27, 30)
DOT_CENTER = (50, 45)
DOT_RADIUS = 5

PNG_SIZES = {"favicon-32.png": 32, "apple-touch-icon.png": 180}
SITE_DIRS = ["website/public", "docs/public"]

MONO_FONTS = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
GLYPH_WEIGHT = b"Semibold"

SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEWBOX} {VIEWBOX}">
  <rect width="{VIEWBOX}" height="{VIEWBOX}" rx="{CORNER}" fill="{BACKGROUND}"/>
  <text x="{GLYPH_CENTER[0]}" y="{GLYPH_CENTER[1]}" fill="{FOREGROUND}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="{GLYPH_SIZE}" font-weight="600" text-anchor="middle" dominant-baseline="central">{GLYPH}</text>
  <circle cx="{DOT_CENTER[0]}" cy="{DOT_CENTER[1]}" r="{DOT_RADIUS}" fill="{ACCENT}"/>
</svg>
"""


def load_font(size):
    for path in MONO_FONTS:
        if not Path(path).exists():
            continue
        font = ImageFont.truetype(path, size=size)
        try:
            font.set_variation_by_name(GLYPH_WEIGHT)
        except OSError:
            pass  # static fallback fonts have no weight axis
        return font
    raise RuntimeError(f"No font found among {MONO_FONTS}")


def render_png(size):
    scale = size / VIEWBOX
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=CORNER * scale, fill=BACKGROUND)
    font = load_font(round(GLYPH_SIZE * scale))
    center = (GLYPH_CENTER[0] * scale, GLYPH_CENTER[1] * scale)
    draw.text(center, GLYPH, font=font, fill=FOREGROUND, anchor="mm")
    dot = (DOT_CENTER[0] * scale, DOT_CENTER[1] * scale)
    r = DOT_RADIUS * scale
    draw.ellipse((dot[0] - r, dot[1] - r, dot[0] + r, dot[1] + r), fill=ACCENT)
    return image


def main():
    for site in SITE_DIRS:
        out = Path(site)
        out.mkdir(parents=True, exist_ok=True)
        (out / "favicon.svg").write_text(SVG)
        for name, size in PNG_SIZES.items():
            render_png(size).save(out / name, optimize=True)


if __name__ == "__main__":
    main()
