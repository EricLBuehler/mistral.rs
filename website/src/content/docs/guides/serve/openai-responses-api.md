---
title: Use the OpenAI Responses API
description: The newer /v1/responses endpoint shape, alongside the classic /v1/chat/completions one. When to use each, and what mistral.rs supports.
sidebar:
  order: 4
---

Alongside the classic Chat Completions API, mistral.rs implements the OpenAI Responses API at `/v1/responses`. The Responses API is a newer shape that OpenAI introduced for agentic workloads, where a single request can involve tool calls, background processing, and cancellation. It is the right choice for long-running agent tasks; Chat Completions is still the better fit for simple request-response chat.

Both live on the same server. You do not need to choose one at startup; clients pick whichever endpoint they prefer.

## Endpoints

The four endpoints that make up the Responses surface:

- `POST /v1/responses` creates a new response. Returns a response object with a unique id.
- `GET /v1/responses/{id}` fetches the current state of a response, including any streamed deltas.
- `DELETE /v1/responses/{id}` deletes a response.
- `POST /v1/responses/{id}/cancel` cancels a background response that has not finished yet.

## When to use it

Reach for Responses when:

- The request might take a long time and you want to poll for progress rather than hold an HTTP connection open.
- You want the ability to cancel a request mid-flight, for instance because the user navigated away from your UI.
- Your client library is built around the Responses shape and you do not want to adapt it.

Stick with Chat Completions when:

- The request is short and finishing it in one round trip is what you want.
- Your client library is older and only speaks Chat Completions.
- You need the widest compatibility with existing OpenAI-compatible tools.

In practice Chat Completions is still what the majority of third-party tools target, and it remains fully supported on our end.

## What mistral.rs supports

The Responses API surface is mostly implemented. A few fields are accepted for compatibility but will refuse non-default values:

- `parallel_tool_calls` must be `true` (the default) or omitted. Setting it to `false` returns an error.
- `max_tool_calls` is not supported; any value returns an error. If you want to limit the number of tool rounds the server runs, use the server-level `--max-tool-rounds` flag, which applies to Chat Completions and Responses alike.

Both of these restrictions come from implementation choices in our tool loop. The loop always runs tools concurrently when the model requests several of them; it does not currently expose a per-request rounds cap.

## mistral.rs extensions

A handful of non-OpenAI fields are accepted in Responses requests. All of them are also accepted on Chat Completions:

- `stop` for custom stop sequences.
- `repetition_penalty`, `top_k`, `min_p` for sampling options that OpenAI's API does not expose.
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers` for DRY sampling.
- `grammar` for constrained generation using llguidance.
- `web_search_options` to configure search behavior per-request (matching the OpenAI syntax for the equivalent field).

The full field reference lives in the [HTTP API reference](/mistral.rs/reference/http-api/).

## Example

Creating a response:

```bash
curl http://localhost:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Summarize today in tech news.",
    "background": true
  }'
```

Polling its progress:

```bash
curl http://localhost:1234/v1/responses/resp_abc123
```

Cancelling it:

```bash
curl -X POST http://localhost:1234/v1/responses/resp_abc123/cancel
```

The request and response schemas match OpenAI's spec, with the additions listed above.
