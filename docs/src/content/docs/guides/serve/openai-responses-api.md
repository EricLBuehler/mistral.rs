---
title: Use the OpenAI Responses API
description: The newer /v1/responses endpoint shape, alongside the classic /v1/chat/completions one. When to use each, and what mistral.rs supports.
sidebar:
  order: 4
---

mistral.rs implements the OpenAI Responses API at `/v1/responses` alongside Chat Completions. Responses is OpenAI's shape for agentic workloads with tool calls, background processing, and cancellation.

Both endpoints run on the same server.

## Endpoints

- `POST /v1/responses`: create a new response. Returns a response object with a unique id.
- `GET /v1/responses/{id}`: fetch the current state, including any streamed deltas.
- `DELETE /v1/responses/{id}`: delete a response.
- `POST /v1/responses/{id}/cancel`: cancel a background response that has not finished.

## Choosing an endpoint

Responses supports polling, mid-flight cancellation via `/cancel`, and background processing. Chat Completions returns the full response on a single connection.

Function tools use the same OpenAI-compatible definitions as Chat Completions, including `strict: true` for JSON-Schema-constrained tool arguments. See [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/).

## Supported fields

A few fields are accepted for compatibility but reject non-default values:

- `parallel_tool_calls` must be `true` (default) or omitted. `false` returns an error.
- `max_tool_calls` is unsupported; any value returns an error. To cap tool rounds, use the server-level `--max-tool-rounds` flag (applies to both Chat Completions and Responses).

## mistral.rs extensions

Non-OpenAI fields accepted in Responses requests (also accepted on Chat Completions):

- `stop`: custom stop sequences.
- `repetition_penalty`, `top_k`, `min_p`: sampling options not in OpenAI's API.
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`: DRY sampling.
- `grammar`: constrained generation via llguidance.
- `web_search_options`: per-request search behavior (matches OpenAI's syntax).

Full field reference: [HTTP API reference](/mistral.rs/reference/http-api/).

## Example

Create a response:

```bash
curl http://localhost:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Summarize today in tech news.",
    "background": true
  }'
```

Poll progress:

```bash
curl http://localhost:1234/v1/responses/resp_abc123
```

Cancel:

```bash
curl -X POST http://localhost:1234/v1/responses/resp_abc123/cancel
```

Request and response schemas match OpenAI's spec, with the additions listed above.
