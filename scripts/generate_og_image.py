#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = (1200, 630)
BACKGROUND = "#121110"
FOREGROUND = "#f8f6f3"
ACCENT = "#f0884a"
TAGLINE_COLOR = "#c7c4c1"

NAME = "mistral"
SUFFIX = ".rs"
TAGLINE = "Fast, flexible LLM inference"

WORDMARK_SIZE = 112
TAGLINE_SIZE = 38
GAP = 34
WORDMARK_WEIGHT = b"Medium"

MONO_FONTS = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
SANS_FONTS = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(candidates, size, weight=None):
    for path in candidates:
        if not Path(path).exists():
            continue
        font = ImageFont.truetype(path, size=size)
        if weight:
            try:
                font.set_variation_by_name(weight)
            except OSError:
                pass  # static fallback fonts have no weight axis
        return font
    raise RuntimeError(f"No font found among {candidates}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("website/public/og.png"))
    args = parser.parse_args()

    image = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    mono = load_font(MONO_FONTS, WORDMARK_SIZE, WORDMARK_WEIGHT)
    sans = load_font(SANS_FONTS, TAGLINE_SIZE)

    name_width = draw.textlength(NAME, font=mono)
    wordmark_width = name_width + draw.textlength(SUFFIX, font=mono)
    _, wm_top, _, wm_bottom = draw.textbbox((0, 0), NAME + SUFFIX, font=mono)
    _, tl_top, tl_right, tl_bottom = draw.textbbox((0, 0), TAGLINE, font=sans)
    block_height = (wm_bottom - wm_top) + GAP + (tl_bottom - tl_top)

    top = (SIZE[1] - block_height) / 2
    left = (SIZE[0] - wordmark_width) / 2
    draw.text((left, top - wm_top), NAME, font=mono, fill=FOREGROUND)
    draw.text((left + name_width, top - wm_top), SUFFIX, font=mono, fill=ACCENT)

    tagline_y = top + (wm_bottom - wm_top) + GAP - tl_top
    draw.text(((SIZE[0] - tl_right) / 2, tagline_y), TAGLINE, font=sans, fill=TAGLINE_COLOR)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, optimize=True)


if __name__ == "__main__":
    main()
