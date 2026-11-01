"""
Use the OpenAI SDK speech, transcription, and translation endpoints with LFM3-Audio.

Start the server:
    mistralrs serve -p 1234 -m LiquidAI/LFM3-Audio-2B --isq q8_0

Generate speech:
    python examples/server/lfm3_audio_endpoints.py --task tts --text "Hello." --output hello.mp3
    python examples/server/lfm3_audio_endpoints.py --task tts --text "Hello." --stream --format pcm

Transcribe or translate an uploaded clip:
    python examples/server/lfm3_audio_endpoints.py --task asr --audio speech.wav
    python examples/server/lfm3_audio_endpoints.py --task translate --audio speech.wav

Set --base-url to use another server. LFM3-Omni supports these endpoints too.
MP3 is the default output format. Compressed output and speed adjustment require FFmpeg.
The checkpoint has one default voice; OpenAI's named voices are unavailable.
Add --stream to ASR for transcript text events or to TTS for speech audio events.
TTS streams incrementally with --format pcm at speed 1.0; other settings buffer the audio.
PCM output is mono, 24 kHz, signed 16-bit little-endian audio without a file header.
"""

import argparse
import base64
import json
from pathlib import Path

from openai import OpenAI

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://localhost:1234/v1/")
parser.add_argument("--model", default="default")
parser.add_argument("--task", choices=["tts", "asr", "translate"], required=True)
parser.add_argument(
    "--audio", type=Path, help="Input clip for transcription or translation"
)
parser.add_argument("--text", default="Hello.", help="Text to synthesize")
parser.add_argument("--output", type=Path, help="Output speech file")
parser.add_argument(
    "--format", choices=["mp3", "opus", "aac", "flac", "wav", "pcm"], default="mp3"
)
parser.add_argument(
    "--speed", type=float, default=1.0, help="Speech speed from 0.25 to 4.0"
)
parser.add_argument("--instructions", help="Additional speech-generation instructions")
parser.add_argument("--language", help="ISO 639-1 transcription language, such as en")
parser.add_argument("--prompt", help="Spelling or vocabulary context for transcription")
parser.add_argument(
    "--stream", action="store_true", help="Stream ASR text or TTS audio events"
)
args = parser.parse_args()
if args.task != "tts" and args.audio is None:
    parser.error("--audio is required for asr and translate")
if not 0.25 <= args.speed <= 4.0:
    parser.error("--speed must be between 0.25 and 4.0")
if args.stream and args.task == "translate":
    parser.error("--stream is supported for --task asr or tts")

client = OpenAI(api_key="foobar", base_url=args.base_url)
if args.task == "tts":
    options = {}
    if args.instructions is not None:
        options["instructions"] = args.instructions
    options.update(
        model=args.model,
        input=args.text,
        voice="default",
        response_format=args.format,
        speed=args.speed,
    )
    output = args.output or Path(f"speech.{args.format}")
    if args.stream:
        with (
            client.audio.speech.with_streaming_response.create(
                stream_format="sse", **options
            ) as response,
            output.open("wb") as audio,
        ):
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line.removeprefix("data:").strip())
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event["type"] == "speech.audio.delta":
                    audio.write(base64.b64decode(event["audio"]))
                    audio.flush()
                elif event["type"] == "speech.audio.done":
                    print(f"Usage: {event['usage']}")
    else:
        output.write_bytes(client.audio.speech.create(**options).content)
    print(f"Saved speech to {output}")
else:
    options = {}
    if args.prompt is not None:
        options["prompt"] = args.prompt
    if args.task == "asr" and args.language is not None:
        options["language"] = args.language
    if args.stream:
        options["stream"] = True
    create = (
        client.audio.transcriptions.create
        if args.task == "asr"
        else client.audio.translations.create
    )
    with args.audio.open("rb") as audio:
        response = create(
            model=args.model, file=audio, response_format="json", **options
        )
    if args.stream:
        for event in response:
            if event.type == "transcript.text.delta":
                print(event.delta, end="", flush=True)
        print()
    else:
        print(response.text)
