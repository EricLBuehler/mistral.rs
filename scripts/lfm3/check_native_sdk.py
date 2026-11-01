"""Check native LFM3 synthesis, single-part audio input, and multi-choice streaming."""

import argparse
import base64
import io
import math
import shutil
import struct
import wave

DEFAULT_MODEL = "LiquidAI/LFM3-Audio-2B"
MODEL_QUANTIZATION = "q8_0"
CACHE_MEMORY_GOAL = "4096tokens"
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
    samples = [
        max(-PCM_SCALE, min(PCM_SCALE - 1, round(value * PCM_SCALE)))
        for value in speech.pcm
    ]
    wav = io.BytesIO()
    with wave.open(wav, "wb") as output:
        output.setnchannels(speech.channels)
        output.setsampwidth(PCM_SAMPLE_BYTES)
        output.setframerate(speech.rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return base64.b64encode(wav.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        max_seqs=STREAM_CHOICES,
        seed=0,
    )
    model = runner.list_models()[0]
    speech = runner.generate_audio(TEST_PHRASE, model_id=model)
    assert speech.pcm and speech.rate == SAMPLE_RATE and speech.channels == CHANNELS
    assert all(math.isfinite(value) for value in speech.pcm)
    assert any(value != 0 for value in speech.pcm)
    data = wav_base64(speech)
    print(f"{model}: generated {len(speech.pcm)} PCM samples")
    parameters = dict(
        model="default", max_tokens=MAX_TOKENS, temperature=0, enable_thinking=False
    )
    for route in ("regular", "per_model"):
        request = ChatCompletionRequest(
            **parameters,
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
        )
        response = send(runner, request, model, route)
        text = response.choices[0].message.content
        assert text and "hello" in text.lower(), (route, text)
        print(f"{route} ASR: {text}")

    messages = [
        {"role": "system", "content": TTS_SYSTEM},
        {"role": "user", "content": TEST_PHRASE},
    ]
    if args.compressed:
        response = runner.send_chat_completion_request(
            ChatCompletionRequest(
                **parameters,
                messages=messages,
                modalities=["audio"],
                audio={"format": "mp3", "voice": "default"},
            ),
            model_id=model,
        )
        audio = response.choices[0].message.audio
        assert audio and audio.data
        encoded = base64.b64decode(audio.data)
        assert encoded.startswith(b"ID3") or (
            len(encoded) >= 2 and encoded[0] == 0xFF and encoded[1] & 0xE0 == 0xE0
        )
        print(f"MP3: {len(encoded)} bytes")

    for route in ("regular", "per_model"):
        request = ChatCompletionRequest(
            **parameters,
            messages=messages,
            modalities=["audio"],
            audio={"format": "pcm16", "voice": "default"},
            stream=True,
            n_choices=STREAM_CHOICES,
        )
        stream = send(runner, request, model, route)
        chunk_counts = [0] * STREAM_CHOICES
        finished = set()
        for response in stream:
            for choice in response.choices:
                if choice.delta.audio and choice.delta.audio.data:
                    pcm = base64.b64decode(choice.delta.audio.data)
                    assert pcm and len(pcm) % PCM_SAMPLE_BYTES == 0
                    chunk_counts[choice.index] += 1
                if choice.finish_reason:
                    finished.add(choice.index)
        assert finished == set(range(STREAM_CHOICES)), (route, finished)
        assert min(chunk_counts) >= MIN_AUDIO_CHUNKS, (route, chunk_counts)
        print(f"{route} streaming: PCM chunks per choice {chunk_counts}")


if __name__ == "__main__":
    main()
