---
title: OpenAI compatibility
description: Which OpenAI API fields mistralrs implements, which it extends, and which it does not support.
sidebar:
  order: 5
---

mistralrs aims for compatibility with the OpenAI API at the field level. Most client libraries that work against OpenAI work against mistralrs without changes. This page lists the exceptions.

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

- `tool_choice`: `"auto"`, `"none"`, and specific function objects all work. `"required"` is not supported; use a specific function object to force tool use.
- `stream_options`: `include_usage` is respected.
- `response_format` with JSON schemas: uses llguidance for constrained decoding. Works but emit shape may differ slightly from OpenAI's output when the schema is ambiguous.

### Ignored

- `store`: OpenAI's response persistence flag is ignored. Use the mistralrs `session_id` field instead.
- `metadata`: accepted but not surfaced anywhere.
- `service_tier`, `parallel_tool_calls`: accepted but ignored (we always execute tools in parallel when possible).

### mistralrs extensions

These are accepted alongside the OpenAI fields and have no effect on OpenAI's endpoint (they get ignored there):

- `top_k`: hard candidate cap.
- `min_p`: min-p sampling threshold.
- `repetition_penalty`: simpler alternative to frequency/presence.
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`: DRY sampling parameters.
- `grammar`, `grammar_type`: llguidance constraints beyond JSON schemas.
- `enable_thinking`: explicit opt-in to thinking tokens for models that support them.
- `web_search_options`: search tool configuration (actually a de-facto OpenAI field but not yet universal).
- `session_id`: multi-turn session persistence.
- `truncate_sequence`: truncate long prompts at the model's context limit instead of erroring.

## Responses API fields

See the [Responses guide](/mistral.rs/guides/serve/openai-responses-api/). Notable exceptions:

- `parallel_tool_calls` must be `true` or omitted. `false` returns an error.
- `max_tool_calls` returns an error for any value.

## Completions (legacy)

The `/v1/completions` endpoint (non-chat) is supported for legacy clients. OpenAI deprecated this endpoint for most of their models; we keep it working. Same extensions apply as Chat Completions.

## Embeddings

- `input` accepts a string or a list of strings.
- `dimensions` is supported for Matryoshka embedding models.
- `encoding_format`: only `"float"` is supported. `"base64"` returns an error.
- `user`: accepted but not used.

Extensions:

- `instruction`: some embedding models accept an instruction prefix. See [embedding guide](/mistral.rs/guides/models/use-embeddings/).

## Image Generation

- `prompt`
- `n`
- `size`
- `response_format`: `"b64_json"` and `"url"` (data URL) both supported.
- `quality`, `style`: ignored. mistralrs uses `steps` and `guidance_scale` instead.

Extensions:

- `steps`: override sampling steps.
- `guidance_scale`: classifier-free guidance strength.

## Audio

### `/v1/audio/speech` (TTS)

- `model`, `input`, `voice`, `response_format`, `speed`: all supported.

### `/v1/audio/transcriptions` and `/v1/audio/translations`

Not supported as dedicated endpoints. Voxtral and similar STT models go through `/v1/chat/completions` with audio content parts. See [speech models guide](/mistral.rs/guides/models/use-speech-models/).

## Moderation

Not supported. mistralrs has no built-in moderation model; if you need one, run it as a separate service.

## Files and Assistants APIs

Not supported. These are OpenAI-specific constructs around file uploads and stateful agents. mistralrs's equivalent is the session-based agentic loop, which is covered by the chat completions endpoint.

## Fine-tuning and Batch

Not supported. mistralrs is an inference engine, not a training platform.

## Tokenization

`/v1/tokenize` and `/v1/detokenize` are not OpenAI standard but are exposed by mistralrs for clients that need tokenizer access. See [HTTP API reference](/mistral.rs/reference/http-api/).

## Authentication

OpenAI requires an `Authorization: Bearer ...` header. mistralrs does not check it. Clients that cannot be configured without an API key can send any non-empty string; it is ignored. If you need real authentication, put an authenticating reverse proxy in front of mistralrs.

## Response headers

mistralrs returns `Content-Type: application/json` for non-streaming responses and `text/event-stream` for streaming ones. A few mistralrs-specific headers may appear:

- `X-Request-Id`: correlation id for logs.
- `X-Session-Id`: the session id used (when one was assigned or matched).
