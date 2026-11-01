"""
LiquidAI LFM3-VL video understanding through the OpenAI-compatible HTTP API.

Start the server:
    mistralrs serve -p 1234 -m LiquidAI/LFM3-VL-2B --isq q4k

Then run `python examples/server/lfm3_vl_video.py clip.mp4`.
The video path must be readable by the server on the same filesystem.
Non-GIF video requires FFmpeg on the server PATH; GIFs are decoded natively.
"""

import argparse
from pathlib import Path

from openai import OpenAI

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path, help="Path to a video readable by the server")
args = parser.parse_args()

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": args.video.resolve().as_uri()},
                },
                {
                    "type": "text",
                    "text": "Describe what happens in this video, in order.",
                },
            ],
        }
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
