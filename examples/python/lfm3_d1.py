"""
Evaluate named questions with d1 using the native Python SDK.

Run `python examples/python/lfm3_d1.py` or add --image, --audio, or --video.
Media runs omit the sample customer message; use --question for a specific yes/no check.
Use --instruction-audio question.wav to ask the observation question aloud.
The default d1-Omni-2B accepts all input modalities; d1-2B accepts images and video.
"""

import argparse
import json
from pathlib import Path

from mistralrs import DecisionQuestion, DecisionRequest, Runner, Which

DEFAULT_MODEL = "LiquidAI/d1-Omni-2B"


def media_part(path: Path, kind: str) -> dict:
    return {"type": kind, kind: {"url": path.resolve().as_uri()}}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default=DEFAULT_MODEL)
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
    "urgent": DecisionQuestion("noul", "Does the customer need urgent help?"),
    "department": DecisionQuestion(
        "choice",
        "Which department should handle this request?",
        criteria={"billing": "Payments and invoices", "support": "Technical problems"},
    ),
    "customer": DecisionQuestion("text", "Extract the customer's name."),
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
        "observation": DecisionQuestion(
            "noul",
            media_part(args.instruction_audio, "audio_url")
            if args.instruction_audio
            else args.question or "Does the attached media show or describe a problem?",
        )
    }
runner = Runner(which=Which.MultimodalPlain(model_id=args.model))
result = runner.send_decision_request(
    DecisionRequest(state=state, questions=questions, media=media)
)
print(json.dumps(result.to_dict(), indent=2))
