"""
Evaluate named questions with d1 through POST /v1/systemone.

Start `mistralrs serve -p 1234 -m LiquidAI/d1-Omni-2B`.
Run `python examples/server/lfm3_d1.py` or add --image, --audio, or --video.
Media runs omit the sample customer message; use --question for a specific yes/no check.
Use --instruction-audio question.wav to ask the observation question aloud.
d1-2B accepts images and video; audio input requires d1-Omni-2B.
"""

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://localhost:1234/v1/"
REQUEST_TIMEOUT_SECONDS = 120


def media_part(path: Path, kind: str) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": kind, kind: {"url": f"data:{mime};base64,{data}"}}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
parser.add_argument("--model", default="default")
parser.add_argument("--image", type=Path)
parser.add_argument("--audio", type=Path)
parser.add_argument("--video", type=Path)
question_args = parser.add_mutually_exclusive_group()
question_args.add_argument("--question")
question_args.add_argument("--instruction-audio", type=Path)
args = parser.parse_args()

state = {
    "customer": "Sam",
    "message": "My card was charged twice. Please help today.",
}
questions = {
    "urgent": {"type": "noul", "instructions": "Does the customer need urgent help?"},
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "Payments and invoices",
            "support": "Technical problems",
        },
    },
    "customer": {"type": "text", "instructions": "Extract the customer's name."},
}
media = [
    media_part(path, kind)
    for path, kind in [
        (args.image, "image_url"),
        (args.audio, "audio_url"),
        (args.video, "video_url"),
    ]
    if path is not None
]
if media:
    state = None
if media or args.question or args.instruction_audio:
    questions = {
        "observation": {
            "type": "noul",
            "instructions": media_part(args.instruction_audio, "audio_url")
            if args.instruction_audio
            else args.question or "Does the attached media show or describe a problem?",
        }
    }
body = {
    "model": args.model,
    "state": state,
    "questions": questions,
    "media": media,
}
request = Request(
    args.base_url.rstrip("/") + "/systemone",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
    print(json.dumps(json.load(response), indent=2))
