"""
LiquidAI LFM3-VL image understanding through the OpenAI-compatible HTTP API.

Start the server:
    mistralrs serve -p 1234 -m LiquidAI/LFM3-VL-2B --isq q4k

Then run `python examples/server/lfm3_vl.py photo.jpg`.
The image path must be readable by the server on the same filesystem.
"""

import argparse
from pathlib import Path

from openai import OpenAI

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("image", type=Path, help="Path to an image readable by the server")
args = parser.parse_args()

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": args.image.resolve().as_uri()},
                },
                {
                    "type": "text",
                    "text": "Describe this image and identify the main subject.",
                },
            ],
        },
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
