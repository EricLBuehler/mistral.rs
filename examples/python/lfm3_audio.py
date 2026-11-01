"""
LiquidAI LFM3-Audio transcription, text-to-speech, and voice chat with the Python SDK.

Run `python examples/python/lfm3_audio.py --task asr --audio speech.wav`.
Use `--task tts --prompt "Hello."` for speech, or `--task chat --audio question.wav`.
TTS and chat save WAV to --speech-out (answer.wav by default).
WAV/MP3 audio uses standard input_audio parts; other formats use the audio_url extension.
This example uses non-streaming chat requests. PCM16 chat streaming is also supported.
--max-tokens counts both text tokens and speech frames (default: 1024).
Increase it for longer speech.
"""

import argparse
import base64
import sys
from pathlib import Path

from mistralrs import ChatCompletionRequest, Runner, Which

DEFAULT_MAX_TOKENS = 1024

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=DEFAULT_MAX_TOKENS,
    help="Positive budget for text tokens and speech frames; increase for longer speech",
)
parser.add_argument("--task", choices=["asr", "tts", "chat"], default="chat")
parser.add_argument("--audio", type=Path, help="Input audio clip for ASR or voice chat")
parser.add_argument("--prompt", help="Text input for TTS or chat")
parser.add_argument("--speech-out", type=Path, default=Path("answer.wav"))
args = parser.parse_args()
if args.max_tokens <= 0:
    parser.error("--max-tokens must be positive")
if args.task == "asr" and args.audio is None:
    parser.error("--task asr requires --audio")
if args.task == "tts" and args.audio is not None:
    parser.error("--task tts accepts text; use --task chat for audio input")

system = {
    "asr": "Perform ASR in English.",
    "tts": "Perform TTS in American English using a female voice.",
    "chat": "Respond with interleaved text and audio.",
}[args.task]
output = {}
if args.task != "asr":
    output = {
        "modalities": ["audio"] if args.task == "tts" else ["text", "audio"],
        "audio": {"format": "wav", "voice": "default"},
    }

content = []
if args.audio is not None:
    audio_format = args.audio.suffix.lower().removeprefix(".")
    if audio_format in {"wav", "mp3"}:
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(args.audio.read_bytes()).decode("ascii"),
                    "format": audio_format,
                },
            }
        )
    else:
        content.append(
            {"type": "audio_url", "audio_url": {"url": args.audio.resolve().as_uri()}}
        )
if args.prompt is not None or args.audio is None:
    content.append({"type": "text", "text": args.prompt or "Hello."})
messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]

runner = Runner(
    which=Which.MultimodalPlain(model_id="LiquidAI/LFM3-Audio-2B"),
    in_situ_quant="q8_0",
)
response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0.1,
        **output,
    )
)
choice = response.choices[0]
if choice.finish_reason == "length":
    print(
        "Generation reached --max-tokens; output may be incomplete. Increase --max-tokens for longer output.",
        file=sys.stderr,
    )
message = choice.message
if message.content:
    print(message.content)
if args.task != "asr" and message.audio is not None:
    args.speech_out.write_bytes(base64.b64decode(message.audio.data))
    print(f"Saved speech to {args.speech_out}")
print(response.usage)
