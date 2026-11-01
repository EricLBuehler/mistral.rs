"""
LiquidAI LFM3-Omni text and speech answers from audio, images, and video over HTTP.

Start the server:
    mistralrs serve -p 1234 -m LiquidAI/LFM3-Omni-2B --isq q8_0

Run `python examples/server/lfm3_omni.py --audio question.wav --image photo.jpg`.
Each media flag is optional; use --prompt for a text-only request.
WAV/MP3 audio uses standard input_audio parts. Other formats use the audio_url extension.
Images and video are read locally and sent as base64 data URLs.
Video needs FFmpeg unless it is a GIF. Supply its soundtrack with --audio.
Add --speech-out answer.wav for text and speech, or --audio-only for speech alone.
--max-tokens counts both text tokens and speech frames (default: 1024).
Increase it for longer speech.
"""

import argparse
import base64
import mimetypes
import sys
from pathlib import Path

from openai import OpenAI

DEFAULT_MAX_TOKENS = 1024

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://localhost:1234/v1/")
parser.add_argument(
    "--max-tokens",
    type=int,
    default=DEFAULT_MAX_TOKENS,
    help="Positive budget for text tokens and speech frames; increase for longer speech",
)
parser.add_argument("--audio", type=Path, help="Path to an audio clip")
parser.add_argument("--image", type=Path, help="Path to an image")
parser.add_argument("--video", type=Path, help="Path to a video or animated GIF")
parser.add_argument("--speech-out", type=Path, help="Save the generated speech as WAV")
parser.add_argument(
    "--audio-only", action="store_true", help="Generate speech without text"
)
parser.add_argument(
    "--prompt", default="Describe the supplied media and answer any spoken question."
)
args = parser.parse_args()
if args.max_tokens <= 0:
    parser.error("--max-tokens must be positive")
if args.audio_only and args.speech_out is None:
    parser.error("--audio-only requires --speech-out")
output = {}
if args.speech_out is not None:
    output = {
        "modalities": ["audio"] if args.audio_only else ["text", "audio"],
        "audio": {"format": "wav", "voice": "default"},
    }

content = []
for kind, path in [("image", args.image), ("video", args.video), ("audio", args.audio)]:
    if path is not None:
        key = f"{kind}_url"
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        audio_format = path.suffix.lower().removeprefix(".")
        if kind == "audio" and audio_format in {"wav", "mp3"}:
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": audio_format},
                }
            )
        else:
            content.append(
                {"type": key, key: {"url": f"data:{mime_type};base64,{data}"}}
            )
content.append({"type": "text", "text": args.prompt})

client = OpenAI(api_key="foobar", base_url=args.base_url)
completion = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": content}],
    max_tokens=args.max_tokens,
    temperature=0.1,
    **output,
)
choice = completion.choices[0]
if choice.finish_reason == "length":
    print(
        "Generation reached --max-tokens; output may be incomplete. Increase --max-tokens for longer output.",
        file=sys.stderr,
    )
message = choice.message
if message.content:
    print(message.content)
if args.speech_out is not None and message.audio is not None:
    args.speech_out.write_bytes(base64.b64decode(message.audio.data))
    print(f"Saved speech to {args.speech_out}")
