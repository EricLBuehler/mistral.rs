---
title: OpenAI compatibility
description: Which OpenAI API fields mistralrs implements, which it extends, and which it does not support.
sidebar:
  order: 5
---

mistral.rs targets field-level OpenAI API compatibility. Most OpenAI client libraries work against mistral.rs unchanged. This page lists the exceptions. For setup and examples, see [OpenAI-compatible APIs](/mistral.rs/guides/serve/openai-compatible-apis/).

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
- `response_format` (`text`, `json_schema`)
- `logit_bias`
- `logprobs`, `top_logprobs`
- `presence_penalty`, `frequency_penalty`
- `n` (multiple completions)

### Implemented with deviation

- `tool_choice`: `"auto"`, `"none"`, and specific function objects work. `"required"` is unsupported; use a specific function object to force tool use.
- `tools[*].function.strict`: accepted on function tools. When `true`, mistral.rs constrains generated tool arguments to the tool's `parameters` JSON Schema. See [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/).
- `response_format` with `json_schema`: uses llguidance for constrained decoding. Output shape may differ from OpenAI's on ambiguous schemas. `json_object` is not accepted.

### Silently ignored

`seed`, `user`, `stream_options`, `metadata`, `service_tier`, `parallel_tool_calls`, `store`. The request body accepts these fields (unknown fields are not rejected) but no behavior is wired to them. Use mistral.rs `session_id` for persistence.

### mistralrs extensions

Accepted alongside OpenAI fields. OpenAI ignores them:

- `top_k`: hard candidate cap.
- `min_p`: min-p sampling threshold.
- `repetition_penalty`: simpler alternative to frequency/presence.
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`: DRY sampling parameters.
- `grammar`: llguidance constraints beyond JSON schemas.
- `enable_thinking`: tri-state for supporting models. `true` forces thinking on, `false` forces it off, omitting the field (or sending `null`) uses the chat template's default (currently thinking on). Note that the Python SDK's `ChatCompletionRequest` constructor defaults this to `False` rather than `None`.
- `web_search_options`: search tool configuration (de facto OpenAI field, not yet universal).
- `session_id`: multi-turn session persistence.
- `truncate_sequence`: truncate long prompts at the model's context limit instead of erroring.

## Responses API fields

See the [Responses guide](/mistral.rs/guides/serve/openai-responses-api/). Notable exceptions:

- `parallel_tool_calls` must be `true` or omitted. `false` returns an error.
- `max_tool_calls` returns an error for any value.
- Function tools support `strict: true` with the same JSON-Schema-constrained argument generation as Chat Completions.

## Completions (legacy)

`/v1/completions` (non-chat) is supported with a subset of Chat Completions extensions: `top_k`, `min_p`, `repetition_penalty`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`, `grammar`, `truncate_sequence`. The agentic, session, file, web-search, thinking, and reasoning-effort fields are not part of this endpoint's schema and have no effect.

## Embeddings

- `input` accepts a string or a list of strings.
- `encoding_format`: `"float"` (default) or `"base64"`.
- `dimensions`: passing any value returns an error. Custom dimensions are not supported.
- `user`: accepted but not used.

Extensions:

- `truncate_sequence`: truncate long prompts at the model's context limit instead of erroring.

## Image Generation

- `prompt`
- `n`
- `response_format`: `"Url"` (default; response carries a server-side filename in `url`) or `"B64Json"` (response carries a `data:image/png;base64,...` string in `b64_json`).

OpenAI's `size` string (e.g. `"1024x1024"`) is not supported. Use the `height` and `width` fields instead:

- `height` (default 720)
- `width` (default 1280)

`quality`, `style`, `steps`, `guidance_scale` are ignored.

## Audio

### `/v1/audio/speech` (TTS)

- `model`, `input`: supported.
- `response_format`: only `wav` and `pcm` are accepted; `mp3`, `opus`, `aac`, `flac` return a validation error.
- `voice`, `instructions`, `speed`: ignored.

### `/v1/audio/transcriptions` and `/v1/audio/translations`

Not exposed as dedicated endpoints. Voxtral and similar STT models go through `/v1/chat/completions` with audio content parts. See [speech models guide](/mistral.rs/guides/models/use-speech-models/).

## Moderation

Not supported. mistral.rs has no built-in moderation model; run one as a separate service if needed.

## Files and Assistants APIs

File uploads (OpenAI's `POST /v1/files`) are not supported. mistral.rs exposes `GET /v1/files`, `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, and `DELETE /v1/files/{id}` for files produced by the agentic loop. The Assistants API is not supported; the mistral.rs equivalent is the session-based agentic loop on the chat completions endpoint.

## Fine-tuning and Batch

Not supported. mistral.rs is an inference engine, not a training platform.

## Tokenization

mistral.rs does not expose `/v1/tokenize` or `/v1/detokenize` HTTP endpoints. Tokenizer access is available through the SDKs (`tokenize_text` / `detokenize_text` in Python; `tokenize_with_model` / `detokenize_with_model` in Rust).

## Authentication

OpenAI requires an `Authorization: Bearer ...` header. mistral.rs does not validate it. Clients that require an API key for initialization can send any non-empty string. For real authentication, place an authenticating reverse proxy in front.

## Response headers

`Content-Type: application/json` for non-streaming responses; `text/event-stream` for streaming. The session id (when assigned or matched) is in the response body's `session_id` field.
