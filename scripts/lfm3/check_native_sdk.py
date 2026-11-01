"""Run opt-in native Python LFM3 speech regressions against a real checkpoint.

Install the mistralrs Python binding, then run from the repository root:
    python scripts/lfm3/check_native_sdk.py
    python scripts/lfm3/check_native_sdk.py --compressed

The default model is LiquidAI/LFM3-Audio-2B with Q8_0 quantization. Add
--no-paged-attn for a build without paged attention. --compressed requires FFmpeg.
This loads a model directly through the native binding and does not use a server.
"""

import argparse
import base64
import io
import json
import math
import shutil
import struct
import wave
from collections import defaultdict

DEFAULT_MODEL = "LiquidAI/LFM3-Audio-2B"
MODEL_QUANTIZATION = "q8_0"
CACHE_MEMORY_GOAL = "4096tokens"
MAX_SEQUENCES = 2
MAX_TOKENS = 128
STREAM_CHOICES = 2
MIN_AUDIO_CHUNKS = 2
SAMPLE_RATE = 24_000
CHANNELS = 1
PCM_SAMPLE_BYTES = 2
PCM_SCALE = 32768
TEST_PHRASE = "Hello there. How are you?"
ASR_SYSTEM = "Perform ASR in English."
TTS_SYSTEM = "Perform TTS in American English using a female voice."


def send(runner, request, model, route):
    if route == "regular":
        return runner.send_chat_completion_request(request, model_id=model)
    return runner.send_chat_completion_request_to_model(request, model)


def wav_base64(speech):
    wav = io.BytesIO()
    with wave.open(wav, "wb") as output:
        output.setnchannels(speech.channels)
        output.setsampwidth(PCM_SAMPLE_BYTES)
        output.setframerate(speech.rate)
        output.writeframes(
            b"".join(
                struct.pack(
                    "<h", max(-PCM_SCALE, min(PCM_SCALE - 1, round(value * PCM_SCALE)))
                )
                for value in speech.pcm
            )
        )
    return base64.b64encode(wav.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Also validate MP3 output using FFmpeg",
    )
    parser.add_argument(
        "--no-paged-attn",
        action="store_true",
        help="Use eager attention for builds without paged attention",
    )
    args = parser.parse_args()
    if args.compressed and shutil.which("ffmpeg") is None:
        parser.error("--compressed requires FFmpeg on PATH")

    from mistralrs import ChatCompletionRequest, Runner, Which

    runner = Runner(
        which=Which.MultimodalPlain(model_id=args.model),
        in_situ_quant=MODEL_QUANTIZATION,
        paged_attn=not args.no_paged_attn,
        no_paged_attn=args.no_paged_attn,
        memory_goal=None if args.no_paged_attn else CACHE_MEMORY_GOAL,
        max_seqs=MAX_SEQUENCES,
        seed=0,
    )
    model = runner.list_models()[0]
    speech = runner.generate_audio(TEST_PHRASE, model_id=model)
    assert speech.pcm and speech.rate == SAMPLE_RATE and speech.channels == CHANNELS
    assert all(math.isfinite(value) for value in speech.pcm)
    assert any(value != 0 for value in speech.pcm)
    data = wav_base64(speech)
    report = {"model": model, "helper_samples": len(speech.pcm), "asr": {}}

    for route in ("regular", "per_model"):
        request = ChatCompletionRequest(
            model="default",
            messages=[
                {"role": "system", "content": ASR_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data, "format": "wav"},
                        }
                    ],
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0,
            enable_thinking=False,
        )
        response = send(runner, request, model, route)
        text = response.choices[0].message.content
        assert text and "hello" in text.lower(), (route, text)
        report["asr"][route] = text

    messages = [
        {"role": "system", "content": TTS_SYSTEM},
        {"role": "user", "content": TEST_PHRASE},
    ]
    if args.compressed:
        response = runner.send_chat_completion_request(
            ChatCompletionRequest(
                model="default",
                messages=messages,
                modalities=["audio"],
                audio={"format": "mp3", "voice": "default"},
                max_tokens=MAX_TOKENS,
                temperature=0,
                enable_thinking=False,
            ),
            model_id=model,
        )
        audio = response.choices[0].message.audio
        assert audio and audio.data
        encoded = base64.b64decode(audio.data)
        assert encoded.startswith(b"ID3") or (
            len(encoded) >= 2 and encoded[0] == 0xFF and encoded[1] & 0xE0 == 0xE0
        )
        report["mp3_bytes"] = len(encoded)

    for route in ("regular", "per_model"):
        request = ChatCompletionRequest(
            model="default",
            messages=messages,
            modalities=["audio"],
            audio={"format": "pcm16", "voice": "default"},
            stream=True,
            n_choices=STREAM_CHOICES,
            max_tokens=MAX_TOKENS,
            temperature=0,
            enable_thinking=False,
        )
        stream = send(runner, request, model, route)
        chunk_counts = defaultdict(int)
        byte_counts = defaultdict(int)
        finished = set()
        for response in stream:
            for choice in response.choices:
                if choice.delta.audio and choice.delta.audio.data:
                    chunk_counts[choice.index] += 1
                    byte_counts[choice.index] += len(
                        base64.b64decode(choice.delta.audio.data)
                    )
                if choice.finish_reason:
                    finished.add(choice.index)
        assert finished == set(range(STREAM_CHOICES)), (route, finished)
        assert all(chunk_counts[index] >= MIN_AUDIO_CHUNKS for index in finished), (
            route,
            dict(chunk_counts),
        )
        assert all(byte_counts[index] > 0 for index in finished)
        report[f"{route}_stream"] = {
            index: {
                "chunks": chunk_counts[index],
                "bytes": byte_counts[index],
            }
            for index in sorted(finished)
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
