---
title: Use the OpenAI Responses API
description: The newer /v1/responses endpoint shape, alongside the classic /v1/chat/completions one. When to use each, and what mistral.rs supports.
sidebar:
  order: 4
---

mistral.rs implements the OpenAI Responses API at `/v1/responses` alongside Chat Completions. Responses is OpenAI's newer shape for agentic workloads with tool calls, background processing, and cancellation. It suits long-running agent tasks; Chat Completions remains preferable for simple request-response chat.

Both endpoints run on the same server. Clients pick whichever they prefer; no startup choice is required.

## Endpoints

- `POST /v1/responses` — create a new response. Returns a response object with a unique id.
- `GET /v1/responses/{id}` — fetch the current state, including any streamed deltas.
- `DELETE /v1/responses/{id}` — delete a response.
- `POST /v1/responses/{id}/cancel` — cancel a background response that has not finished.

## When to use it

Use Responses when:

- The request is long-running and polling is preferable to holding an HTTP connection open.
- Mid-flight cancellation is needed (e.g., user navigated away).
- The client library is built around the Responses shape.

Use Chat Completions when:

- The request is short and one round trip is sufficient.
- The client library only speaks Chat Completions.
- Maximum compatibility with existing OpenAI tooling is required.

Most third-party tools target Chat Completions, which remains fully supported.

## What mistral.rs supports

The Responses surface is mostly implemented. A few fields are accepted for compatibility but reject non-default values:

- `parallel_tool_calls` must be `true` (default) or omitted. `false` returns an error.
- `max_tool_calls` is unsupported; any value returns an error. To cap tool rounds, use the server-level `--max-tool-rounds` flag (applies to both Chat Completions and Responses).

These restrictions reflect implementation choices in the tool loop: it always runs concurrent tool calls when the model requests several, and exposes no per-request rounds cap.

## mistral.rs extensions

Non-OpenAI fields accepted in Responses requests (also accepted on Chat Completions):

- `stop` — custom stop sequences.
- `repetition_penalty`, `top_k`, `min_p` — sampling options not in OpenAI's API.
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers` — DRY sampling.
- `grammar` — constrained generation via llguidance.
- `web_search_options` — per-request search behavior (matches OpenAI's syntax).

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
