#!/usr/bin/env python3
"""Save a reproducible image and ask a local multimodal server to describe it."""

import argparse
import base64
import json
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_SIZE = (256, 256)
RECTANGLE_BOUNDS = (24, 56, 112, 200)
CIRCLE_BOUNDS = (140, 72, 232, 164)
MAX_TOKENS = 96
TIMEOUT_SECONDS = 120


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_path = args.output.with_suffix(".png")
    image = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(RECTANGLE_BOUNDS, fill="red")
    draw.ellipse(CIRCLE_BOUNDS, fill="blue")
    image.save(image_path, format="PNG")
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    body = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Name the two colored shapes and their colors.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + encoded},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        result = json.load(response)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
