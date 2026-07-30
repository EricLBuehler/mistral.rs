#!/usr/bin/env python3
import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ASCII_ART = [
    "  __  __ _     _             _              ",
    " |  \\/  (_)___| |_ _ __ __ _| |  _ __ ___   ",
    " | |\\/| | / __| __| '__/ _` | | | '__/ __|  ",
    " | |  | | \\__ \\ |_| | | (_| | |_| |  \\__ \\  ",
    " |_|  |_|_|___/\\__|_|  \\__,_|_(_)_|  |___/  ",
]

TAGLINE = "Fast, flexible LLM inference."

CANVAS_SIZE = (1600, 480)
MAX_ART_WIDTH = 1360
ART_TOP = 72
TAGLINE_GAP = 12
ART_FRAME_COUNT = 30
TAGLINE_FRAME_MS = 34
ART_FRAME_MS = 30
BLINK_FRAME_MS = 180
FINAL_HOLD_MS = 600_000
RANDOM_SEED = 17

PANEL_RECT = (44, 30, 1556, 456)
PANEL_RADIUS = 28
PANEL_SHADOW_OFFSET = 10


@dataclass(frozen=True)
class Theme:
    canvas: str
    panel: str | None
    panel_border: str | None
    panel_shadow: str | None
    fg: str
    blue: str
    cursor: str


THEMES = {
    "dark": Theme(
        canvas="#020305",
        panel=None,
        panel_border=None,
        panel_shadow=None,
        fg="#f8fafc",
        blue="#38bdf8",
        cursor="#7dd3fc",
    ),
    "light": Theme(
        canvas="#f8fafc",
        panel="#020305",
        panel_border="#d8dee9",
        panel_shadow="#dbe3ee",
        fg="#f8fafc",
        blue="#38bdf8",
        cursor="#7dd3fc",
    ),
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the README banner animation."
    )
    parser.add_argument("--theme", choices=THEMES.keys(), default="dark")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final-output", type=Path)
    args = parser.parse_args()
    args.output = args.output or Path(f"res/banner-{args.theme}.gif")
    args.final_output = args.final_output or Path(f"res/banner-{args.theme}.png")
    return args


def font_path():
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError("No monospace font found. Add one to FONT_CANDIDATES.")


def load_font(size):
    return ImageFont.truetype(font_path(), size=size)


def fit_art_font():
    image = Image.new("RGB", CANVAS_SIZE)
    draw = ImageDraw.Draw(image)
    longest = max(ASCII_ART, key=len)
    for size in range(58, 23, -1):
        font = load_font(size)
        width = draw.textbbox((0, 0), longest, font=font)[2]
        if width <= MAX_ART_WIDTH:
            return font
    return load_font(24)


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def char_positions(draw, font):
    _, text_height = text_size(draw, "Mg", font)
    line_height = text_height + 14
    art_width = max(
        int(draw.textlength(line.rstrip(), font=font)) for line in ASCII_ART
    )
    x0 = (CANVAS_SIZE[0] - art_width) // 2
    cells = []
    rng = random.Random(RANDOM_SEED)

    for row, line in enumerate(ASCII_ART):
        for col, char in enumerate(line):
            if char == " ":
                continue
            jitter = rng.uniform(-0.35, 0.35)
            order = col + row * 2.8 + jitter
            x = x0 + int(draw.textlength(line[:col], font=font))
            y = ART_TOP + row * line_height
            cells.append((order, row, col, char, x, y))

    cells.sort(key=lambda item: item[0])
    art_height = len(ASCII_ART) * line_height
    return cells, line_height, art_height


def draw_banner(
    theme,
    visible_cells,
    tagline_prefix,
    cursor_on,
    art_font,
    tagline_font,
    cells,
    art_height,
):
    image = Image.new("RGB", CANVAS_SIZE, theme.canvas)
    draw = ImageDraw.Draw(image)

    if theme.panel:
        shadow_rect = (
            PANEL_RECT[0],
            PANEL_RECT[1] + PANEL_SHADOW_OFFSET,
            PANEL_RECT[2],
            PANEL_RECT[3] + PANEL_SHADOW_OFFSET,
        )
        draw.rounded_rectangle(
            shadow_rect, radius=PANEL_RADIUS, fill=theme.panel_shadow
        )
        draw.rounded_rectangle(
            PANEL_RECT,
            radius=PANEL_RADIUS,
            fill=theme.panel,
            outline=theme.panel_border,
            width=2,
        )

    for _, _, _, char, x, y in cells[:visible_cells]:
        draw.text((x, y), char, font=art_font, fill=theme.fg)

    final_width = draw.textbbox((0, 0), TAGLINE, font=tagline_font)[2]
    tagline_x = (CANVAS_SIZE[0] - final_width) // 2
    tagline_y = ART_TOP + art_height + TAGLINE_GAP
    draw.text(
        (tagline_x, tagline_y), tagline_prefix, font=tagline_font, fill=theme.blue
    )

    if cursor_on:
        prefix_width = int(draw.textlength(tagline_prefix, font=tagline_font))
        _, cursor_height = text_size(draw, "Mg", tagline_font)
        cursor_x = tagline_x + prefix_width + 8
        draw.rectangle(
            (cursor_x, tagline_y + 5, cursor_x + 13, tagline_y + cursor_height + 5),
            fill=theme.cursor,
        )

    return image


def main():
    args = parse_args()
    theme = THEMES[args.theme]
    art_font = fit_art_font()
    tagline_font = load_font(44)

    probe = Image.new("RGB", CANVAS_SIZE)
    probe_draw = ImageDraw.Draw(probe)
    cells, _, art_height = char_positions(probe_draw, art_font)

    frames = []
    durations = []

    for idx in range(ART_FRAME_COUNT):
        visible = round(len(cells) * (idx + 1) / ART_FRAME_COUNT)
        frames.append(
            draw_banner(
                theme, visible, "", False, art_font, tagline_font, cells, art_height
            )
        )
        durations.append(ART_FRAME_MS)

    for idx in range(len(TAGLINE) + 1):
        frames.append(
            draw_banner(
                theme,
                len(cells),
                TAGLINE[:idx],
                True,
                art_font,
                tagline_font,
                cells,
                art_height,
            )
        )
        durations.append(TAGLINE_FRAME_MS)

    for idx in range(6):
        frames.append(
            draw_banner(
                theme,
                len(cells),
                TAGLINE,
                idx % 2 == 0,
                art_font,
                tagline_font,
                cells,
                art_height,
            )
        )
        durations.append(BLINK_FRAME_MS)

    final = draw_banner(
        theme, len(cells), TAGLINE, False, art_font, tagline_font, cells, art_height
    )
    frames.append(final)
    durations.append(FINAL_HOLD_MS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.final_output.parent.mkdir(parents=True, exist_ok=True)
    final.save(args.final_output)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        optimize=True,
        disposal=2,
    )


if __name__ == "__main__":
    main()
