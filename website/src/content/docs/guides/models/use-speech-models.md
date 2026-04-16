---
title: Speech models
description: Voxtral for speech-to-text, Dia for text-to-speech. How each works through the mistralrs API.
sidebar:
  order: 3
---

mistral.rs supports two dedicated speech-focused architectures: Voxtral for speech-to-text (transcription and speech understanding) and Dia for text-to-speech (generating speech audio from text). Both are accessible through OpenAI-shaped endpoints.

For multimodal models that happen to accept or produce audio as one of several modalities (Gemma 4, for example), see the [vision and video guide](/mistral.rs/guides/models/use-vision-input/#audio) instead. This page is for the dedicated speech models.

## Voxtral: speech-to-text

Voxtral is a streaming-capable speech model. It accepts audio input and produces text, either as a straight transcription or as an answer to a question about the audio.

```bash
mistralrs serve -m mistralai/Voxtral-Mini-3B
```

Voxtral fits in the multimodal chat shape: audio is an input content part, the response is text.

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

The text prompt is optional but useful. "Transcribe this" gets you the transcription verbatim. "Summarize what this person is saying" gets you a summary instead. "Who is speaking and what is the tone?" gets you speaker diarization and affect analysis. Voxtral is trained for all of these tasks.

Supported input formats are the usual ones: `.wav`, `.mp3`, `.flac`, `.ogg`. For other formats, FFmpeg does the conversion.

## Dia: text-to-speech

Dia produces audio from text. The API shape is `POST /v1/audio/speech`, which matches OpenAI's:

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

The response is raw audio bytes in the requested format. WAV is the safest choice; other formats like MP3 require the right decoder to be available on the server.

Voice selection depends on the model. Dia has several built-in voices; see its model card for the current list. To use a voice, pass it in the `voice` field.

## Speed and quality

Both Voxtral and Dia are relatively small (a few billion parameters) and run fast on modest hardware. A consumer GPU with 8 GB is enough for either. On Apple Silicon, both run well in half or full precision; quantization helps little here.

## What about Whisper?

Whisper-style models are not currently supported natively. Voxtral is the closest analog in our lineup and is typically competitive with or better than Whisper for English transcription. For other languages, Voxtral's coverage varies; check the model card.

If you need a Whisper-specific workflow, you can still run it outside mistralrs and call it through MCP or a custom tool callback if you want the results to reach a language model afterward.
