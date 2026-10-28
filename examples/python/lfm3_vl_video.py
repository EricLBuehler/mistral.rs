"""
LiquidAI LFM3-VL video understanding with the Python SDK.

Run with `python examples/python/lfm3_vl_video.py clip.mp4`.
Non-GIF video requires FFmpeg on PATH; animated GIFs are decoded natively.
"""

import argparse
from pathlib import Path

from mistralrs import ChatCompletionRequest, Runner, Which

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path, help="Path to a video or animated GIF")
args = parser.parse_args()

runner = Runner(
    which=Which.MultimodalPlain(model_id="LiquidAI/LFM3-VL-2B"),
    in_situ_quant="q4k",
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
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
)
print(response.choices[0].message.content)
print(response.usage)
