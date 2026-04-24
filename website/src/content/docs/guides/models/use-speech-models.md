---
title: Speech models
description: Voxtral for audio understanding, Dia for text-to-speech.
sidebar:
  order: 3
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
        {"type": "audio", "audio": {"url": "file:///clip.wav"}},
        {"type": "text", "text": "Transcribe this."}
      ]
    }]
  }'
```

The text prompt selects the task: transcription, summarization, speaker analysis, etc.

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
    "input": "Hello. This is a test of the text-to-speech system.",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output out.wav
```

The response is raw audio bytes in the requested format. Voice values are model-specific.

## Whisper

Whisper-style models are not in the supported list. Voxtral is the closest analog.
