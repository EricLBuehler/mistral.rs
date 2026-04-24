---
title: OpenAI compatibility
description: Which OpenAI API fields mistralrs implements, which it extends, and which it does not support.
sidebar:
  order: 5
---

mistral.rs targets field-level OpenAI API compatibility. Most OpenAI client libraries work against mistral.rs unchanged. This page lists the exceptions.

## Chat Completions fields

### Implemented

- `model`
- `messages` (including multimodal content parts)
- `max_tokens`
- `max_completion_tokens` (OpenAI's newer alias for `max_tokens`)
- `temperature`
- `top_p`
- `stream`
- `stop`
- `tools`, `tool_choice`
- `response_format` (`text`, `json_object`)
- `logit_bias`
- `logprobs`, `top_logprobs`
- `presence_penalty`, `frequency_penalty`
- `n` (multiple completions)
- `seed`
- `user` (passed through but not used)

### Implemented with deviation

- `tool_choice` — `"auto"`, `"none"`, and specific function objects work. `"required"` is unsupported; use a specific function object to force tool use.
- `stream_options` — `include_usage` is respected.
- `response_format` with JSON schemas — uses llguidance for constrained decoding. Output shape may differ from OpenAI's on ambiguous schemas.

### Ignored

- `store` — OpenAI's response persistence flag. Use mistral.rs `session_id` instead.
- `metadata` — accepted but not surfaced.
- `service_tier`, `parallel_tool_calls` — accepted but ignored. Tools always execute in parallel when possible.

### mistralrs extensions

Accepted alongside OpenAI fields. OpenAI ignores them:

- `top_k` — hard candidate cap.
- `min_p` — min-p sampling threshold.
- `repetition_penalty` — simpler alternative to frequency/presence.
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers` — DRY sampling parameters.
- `grammar`, `grammar_type` — llguidance constraints beyond JSON schemas.
- `enable_thinking` — explicit opt-in to thinking tokens for supporting models.
- `web_search_options` — search tool configuration (de facto OpenAI field, not yet universal).
- `session_id` — multi-turn session persistence.
- `truncate_sequence` — truncate long prompts at the model's context limit instead of erroring.

## Responses API fields

See the [Responses guide](/mistral.rs/guides/serve/openai-responses-api/). Notable exceptions:

- `parallel_tool_calls` must be `true` or omitted. `false` returns an error.
- `max_tool_calls` returns an error for any value.

## Completions (legacy)

`/v1/completions` (non-chat) is supported. Same extensions as Chat Completions.

## Embeddings

- `input` accepts a string or a list of strings.
- `dimensions` is supported for Matryoshka embedding models.
- `encoding_format`: only `"float"` is supported. `"base64"` returns an error.
- `user`: accepted but not used.

Extensions:

- `instruction` — some embedding models accept an instruction prefix. See [embedding guide](/mistral.rs/guides/models/use-embeddings/).

## Image Generation

- `prompt`
- `n`
- `size`
- `response_format`: `"b64_json"` and `"url"` (data URL) both supported.
- `quality`, `style` — ignored. mistral.rs uses `steps` and `guidance_scale`.

Extensions:

- `steps` — override sampling steps.
- `guidance_scale` — classifier-free guidance strength.

## Audio

### `/v1/audio/speech` (TTS)

- `model`, `input`, `voice`, `response_format`, `speed` — all supported.

### `/v1/audio/transcriptions` and `/v1/audio/translations`

Not exposed as dedicated endpoints. Voxtral and similar STT models go through `/v1/chat/completions` with audio content parts. See [speech models guide](/mistral.rs/guides/models/use-speech-models/).

## Moderation

Not supported. mistral.rs has no built-in moderation model; run one as a separate service if needed.

## Files and Assistants APIs

Not supported. These are OpenAI-specific constructs around file uploads and stateful agents. The mistral.rs equivalent is the session-based agentic loop on the chat completions endpoint.

## Fine-tuning and Batch

Not supported. mistral.rs is an inference engine, not a training platform.

## Tokenization

mistral.rs does not expose `/v1/tokenize` or `/v1/detokenize` HTTP endpoints. Tokenizer access is available through the SDKs (`tokenize_text` / `detokenize_text` in Python; `tokenize_with_model` / `detokenize_with_model` in Rust).

## Authentication

OpenAI requires an `Authorization: Bearer ...` header. mistral.rs does not validate it. Clients that require an API key for initialization can send any non-empty string. For real authentication, place an authenticating reverse proxy in front.

## Response headers

mistral.rs returns `Content-Type: application/json` for non-streaming responses and `text/event-stream` for streaming. mistral.rs-specific headers:

- `X-Request-Id` — correlation id for logs.
- `X-Session-Id` — the session id used (when assigned or matched).
