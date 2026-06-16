---
title: OpenAI compatibility
description: Which OpenAI API fields mistralrs implements, which it extends, and which it does not support.
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

- `tool_choice`: `"auto"`, `"none"`, `"required"`, Chat Completions specific function objects (`{"type":"function","function":{"name":"..."}}`), and Responses-style specific function objects (`{"type":"function","name":"..."}`) work. `"required"` rejects requests with no available tools. Enforcement is validated after generation rather than by forcing the first token.
- `tools[*].function.strict`: accepted on function tools. When `true`, mistral.rs constrains generated tool arguments to the tool's `parameters` JSON Schema. See [tool calling](/mistral.rs/guides/agents/tool-calling-basics/).
- `tools[*].type="code_interpreter"`: accepted as the OpenAI-compatible opt-in for the built-in Python executor. The server must be started with code execution enabled. The only supported container form is `{"type":"auto"}`. Container ids, `container.file_ids`, `container.memory_limit`, and OpenAI container lifecycle endpoints are not supported.
- `messages[].content[]` file parts: `{"type":"file","file":{"file_id":"file-..."}}` and `{"type":"file","file":{"filename":"data.csv","file_data":"data:text/csv;base64,..."}}` are supported. Chat Completions file URLs are not supported; upload the file first or use Responses.
- `response_format` with `json_schema`: uses [llguidance](/mistral.rs/guides/serve/structured-output/) (a constrained-decoding grammar library) to constrain decoding. Output shape may differ from OpenAI's on ambiguous schemas. `json_object` is not accepted.

### Silently ignored

`seed`, `user`, `stream_options`, `metadata`, `service_tier`, `parallel_tool_calls`, `store`. The request body accepts these fields (unknown fields are not rejected) but no behavior is wired to them. Use mistral.rs `session_id` for persistence.

### mistralrs extensions

Accepted alongside OpenAI fields. OpenAI ignores them:

- `top_k`: hard candidate cap.
- `min_p`: min-p sampling threshold.
- `repetition_penalty`: simpler alternative to frequency/presence.
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`: DRY sampling parameters.
- `grammar`: llguidance constraints beyond JSON schemas.
- `enable_thinking`: tri-state for supporting models.
  - `true`: forces thinking on.
  - `false`: forces thinking off.
  - omitted or `null`: uses the chat template's default (currently thinking on).

  The Python SDK's `ChatCompletionRequest` defaults `enable_thinking` to `None`, matching the omitted-field behavior above.
- `web_search_options`: search tool configuration (de facto OpenAI field, not yet universal).
- `session_id`: multi-turn session persistence.
- `files`: required output files for server-side code execution.
- `agent_permission`, `code_execution_permission`: per-request permission tightening for server-executed tools.
- `max_tool_rounds`: cap server-side tool loop rounds for one request.
- `truncate_sequence`: truncate long prompts at the model's context limit instead of erroring.

## Responses API

mistral.rs implements the OpenAI Responses API alongside Chat Completions:

- `POST /v1/responses`: create a response. Returns a response object with a unique id.
- `GET /v1/responses/{id}`: fetch the current state of a stored response.
- `DELETE /v1/responses/{id}`: delete a stored response.
- `POST /v1/responses/{id}/cancel`: cancel a background response that has not finished.

Use Responses when the client:

- expects OpenAI's Responses object shape,
- needs response ids,
- needs polling or background processing,
- needs cancellation.

Chat Completions, by contrast, returns the full response on a single connection. Codex speaks the Responses API; see [coding agents](/mistral.rs/guides/serve/coding-agents/).

### Implemented

- `input`: messages or a raw prompt string.
- `input_file` content parts with `file_id`, `file_data`, or `file_url`.
- `previous_response_id`: continues a stored conversation.
- `max_output_tokens`: with `max_tokens` and `max_completion_tokens` as aliases.
- `instructions`, `temperature`, `top_p`, `stop`, `stream`, `tools`, `tool_choice`, `response_format`, `logit_bias`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`, `n`, `metadata`, `background`, `store`.

`store` defaults to `true`; `store: false` skips caching, which makes the response unavailable to `GET /v1/responses/{id}` and `previous_response_id`. Function tools support `strict: true` with the same JSON-Schema-constrained argument generation as Chat Completions.

Responses `tools` accepts:

- Function tools in the Responses flat form: `{"type":"function","name":"...","parameters":{...}}`.
- Function tools in the Chat Completions nested form: `{"type":"function","function":{...}}`.
- `{"type":"web_search", ...}` and `{"type":"web_search_preview"}` for server-side web search.
- `{"type":"code_interpreter","container":{"type":"auto"}}` for server-side Python code execution.
- `{"type":"shell","environment":{"type":"container_auto","skills":[{"type":"skill_reference","skill_id":"skill_...","version":"latest"}]}}` for server-side shell execution and OpenAI-compatible Skills. Skills require the shell executor, so start the server with at least `--enable-shell`; `--agent` is the recommended preset when you also want the full agentic runtime.

`tool_choice: "required"` is accepted and rejects requests that provide no tools. Specific function choices must reference a declared function tool.

### Skills API

- `POST /v1/skills`: upload one OpenAI-compatible Skill as multipart form data. Use `files` fields containing either a zip archive or the files from one top-level skill directory. The top-level directory must contain `SKILL.md` with `name` and `description` frontmatter.
- `GET /v1/skills`: list uploaded skills for the current server process and skills directory.
- `POST /v1/skills/{skill_id}/versions`: upload a new version for an existing skill.

Uploaded skill versions are stored under the server's skills directory (`--skills-dir`, or a system temp directory by default). When a Responses request references a skill, mistral.rs copies that skill version into the shell session working directory at `skills/<skill-name>/`.

### Rejected non-default values

- `parallel_tool_calls` must be `true` (default) or omitted; `false` returns an error.
- `max_tool_calls` returns an error for any value. To cap tool rounds, use the server-level `--max-tool-rounds` flag (applies to both Chat Completions and Responses).
- `tools[*].type="web_search"` rejects image search (`search_content_types: ["image"]` or `image_settings`) and `external_web_access: false`. Domain filters are supported for up to 100 allowed or blocked domains and include subdomains.
- `tools[*].type="web_search_preview"` rejects `filters` and `return_token_budget`; `external_web_access` is ignored.
- `tools[*].type="code_interpreter"` rejects container ids, `container.file_ids`, and `container.memory_limit`.
- `tools[*].type="shell"` rejects local environments, container references, local skill paths, inline/container-created skills, and OpenAI container lifecycle APIs. Uploaded `skill_reference` skills are supported.

### mistralrs extensions on Responses

`top_k`, `min_p`, `repetition_penalty`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`, `grammar`. The chat-only agentic fields (`session_id`, `agent_permission`, `files`, `max_tool_rounds`, `web_search_options`) are not part of this endpoint's schema. Use the Responses `tools` array for web search, code interpreter, shell, and OpenAI-compatible Skills.

Thinking, reasoning effort, and truncation are not top-level extension fields here; they are controlled through the standard Responses objects. Use the `reasoning` object (`reasoning.effort`) for thinking/reasoning effort and the `truncation` field for sequence truncation. Top-level `enable_thinking`, `reasoning_effort`, and `truncate_sequence` keys are silently ignored on this endpoint.

### Background runs

```bash
curl http://localhost:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Summarize today in tech news.",
    "background": true
  }'
```

Poll with `curl http://localhost:1234/v1/responses/<id>`, cancel with `curl -X POST http://localhost:1234/v1/responses/<id>/cancel`. Streaming event names are in the [HTTP API semantics page](/mistral.rs/reference/http-api/#streaming-responses); full schemas in the [generated reference](/mistral.rs/reference/http-api-generated/).

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

`POST /v1/files` multipart uploads are supported for user-provided input files. Use `purpose="user_data"` for OpenAI-compatible request attachments. Uploaded files, inline request files, URL-fetched request files, and agent-produced files are available through `GET /v1/files`, `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, and `DELETE /v1/files/{id}`.

Text-like UTF-8 files are exposed to the model as bounded decoded previews, with additional text available during agentic runs when file access is active. Binary files are stored, downloadable, and mounted into shell/code workdirs when those tools are active, but mistral.rs does not perform OpenAI's private PDF/image/spreadsheet extraction pipeline. The Assistants API is not supported; the mistral.rs equivalent is the session-based agentic loop on the chat completions endpoint.

## Fine-tuning and Batch

Not supported. mistral.rs is an inference engine, not a training platform.

## Tokenization

mistral.rs does not expose `/v1/tokenize` or `/v1/detokenize` HTTP endpoints. Tokenizer access is available through the SDKs (`tokenize_text` / `detokenize_text` in Python; `tokenize_with_model` / `detokenize_with_model` in Rust).

## Authentication

OpenAI requires an `Authorization: Bearer ...` header. mistral.rs does not validate it. Clients that require an API key for initialization can send any non-empty string. For real authentication, place an authenticating reverse proxy in front.

## Response headers

`Content-Type: application/json` for non-streaming responses; `text/event-stream` for streaming. The session id (when assigned or matched) is in the response body's `session_id` field.
