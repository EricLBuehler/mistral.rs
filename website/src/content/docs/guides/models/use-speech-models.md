---
title: Speech models
description: Voxtral for speech-to-text, Dia for text-to-speech. How each works through the mistralrs API.
sidebar:
  order: 3
---

mistral.rs supports two dedicated speech architectures: Voxtral for speech-to-text and Dia for text-to-speech. Both use OpenAI-shaped endpoints.

For multimodal models that handle audio as one of several modalities (Gemma 4), see the [vision and video guide](/mistral.rs/guides/models/use-vision-input/#audio).

## Voxtral: speech-to-text

Voxtral is a streaming-capable speech model. It takes audio input and produces text — straight transcription or answers to questions about the audio.

```bash
mistralrs serve -m mistralai/Voxtral-Mini-3B
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
        {"type": "audio", "audio": {"url": "file:///clip.wav"}},
        {"type": "text", "text": "Transcribe this."}
      ]
    }]
  }'
```

The text prompt is optional. "Transcribe this" yields verbatim transcription. "Summarize what this person is saying" yields a summary. "Who is speaking and what is the tone?" yields speaker diarization and affect analysis. Voxtral is trained for all of these.

Native input formats: `.wav`, `.mp3`, `.flac`, `.ogg`. Other formats use FFmpeg.

## Dia: text-to-speech

Dia produces audio from text via `POST /v1/audio/speech`, matching OpenAI:

```bash
mistralrs serve -m nari-labs/Dia-1.6B
```

```bash
curl http://localhost:1234/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Hello. This is a test of the text-to-speech system.",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output out.wav
```

The response is raw audio bytes in the requested format. WAV is the safest choice; MP3 and others require the appropriate decoder on the server.

Voice selection is model-specific. Dia's available voices are listed in its model card. Pass the voice name in the `voice` field.

## Speed and quality

Voxtral and Dia are both small (a few billion parameters) and run fast on modest hardware. An 8 GB consumer GPU is sufficient. Apple Silicon runs both well in half or full precision; quantization helps little.

## What about Whisper?

Whisper-style models are not natively supported. Voxtral is the closest analog and is typically competitive with or better than Whisper for English transcription. For other languages, see the Voxtral model card.

Whisper-specific workflows can run outside mistral.rs and integrate via MCP or a custom tool callback.
