---
title: Speech models
description: Voxtral for audio understanding, Dia for text-to-speech.
sidebar:
  order: 4
---

mistral.rs supports two speech-related model families:

- **Voxtral**: multimodal model accepting audio input. Used for transcription and audio understanding through `/v1/chat/completions`.
- **Dia**: dedicated text-to-speech model served via `/v1/audio/speech`.

Voxtral is classified as a multimodal model (audio is one of its input modalities); Dia is classified as a dedicated speech model.

## Voxtral: audio in, text out

```bash
mistralrs serve -m mistralai/Voxtral-Mini-3B-2507
```

Voxtral fits the multimodal chat shape: audio is an input content part, the response is text.

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "file:///clip.wav"}},
        {"type": "text", "text": "Transcribe this."}
      ]
    }]
  }'
```

The text prompt selects the task: transcription, summarization, speaker analysis, etc.

Voxtral repos use Mistral's native layout (`params.json`, `consolidated.safetensors`, `tekken.json`). The auto-loader detects that layout, so `-m` is enough in normal CLI and server usage.

Python SDK:

```python
from mistralrs import ChatCompletionRequest, MultimodalArchitecture, Runner, Which

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="mistralai/Voxtral-Mini-3B-2507",
        arch=MultimodalArchitecture.Voxtral,
    )
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "file:///absolute/path/clip.wav"}},
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            }
        ],
        max_tokens=256,
        temperature=0,
    )
)
print(response.choices[0].message.content)
```

## Dia: text-to-speech

`/v1/audio/speech` matches OpenAI:

```bash
mistralrs serve -m nari-labs/Dia-1.6B
```

```bash
curl http://localhost:1234/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "[S1] Hello. This is a test of the text-to-speech system.",
    "response_format": "wav"
  }' \
  --output out.wav
```

The response is raw audio bytes. Only `wav` and `pcm` are accepted for `response_format`; other OpenAI values (`mp3`, `opus`, `aac`, `flac`) return a validation error. `voice`, `speed`, and `instructions` are accepted for OpenAI compatibility but ignored.

Dia understands dialogue speaker tags such as `[S1]` and `[S2]`, and nonverbal parentheticals such as `(laughs)` or `(coughs)`. Use them in the `input` string when you want dialogue or expressive speech.

Python SDK:

```python
import struct
import wave
from pathlib import Path

from mistralrs import Runner, SpeechLoaderType, Which

runner = Runner(
    which=Which.Speech(
        model_id="nari-labs/Dia-1.6B",
        arch=SpeechLoaderType.Dia,
    )
)

response = runner.generate_audio("[S1] mistral r s can generate speech locally.")

output_path = Path("out.wav")
pcm_ints = [int(max(-32768, min(32767, sample * 32767))) for sample in response.pcm]
with wave.open(output_path, "wb") as wav:
    wav.setnchannels(response.channels)
    wav.setsampwidth(2)
    wav.setframerate(response.rate)
    wav.writeframes(b"".join(struct.pack("<h", sample) for sample in pcm_ints))
```

## Whisper

Whisper-style models are not in the supported list. Voxtral is the closest analog.
