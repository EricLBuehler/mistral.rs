---
title: Serve models
description: Run mistral.rs as an HTTP server, with one or more models, with the web UI, and with OpenAI-compatible and Anthropic-compatible APIs.
---

[Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/) covers basic single-model serving. These guides cover configuration beyond a single local server, including OpenAI-compatible `/v1` clients and Anthropic-compatible `/v1/messages` clients.

## Choose by task

| If you need to... | Start here |
|---|---|
| Change host, port, CORS, request limits, or authentication | [HTTP server configuration](/mistral.rs/guides/serve/http-server/) |
| Serve more than one model from one process | [Running multiple models](/mistral.rs/guides/serve/multiple-models/) |
| Use the browser chat interface | [Using the web UI](/mistral.rs/guides/serve/with-web-ui/) |
| Use OpenAI-compatible clients | [OpenAI-compatible APIs](/mistral.rs/guides/serve/openai-compatible-apis/) |
| Use the newer OpenAI Responses endpoint | [OpenAI Responses API](/mistral.rs/guides/serve/openai-responses-api/) |
| Use Anthropic-compatible clients | [Anthropic Messages API](/mistral.rs/guides/serve/anthropic-messages-api/) |
| Use Codex or Claude Code with a local server | [Use Codex and Claude Code](/mistral.rs/guides/serve/coding-agents/) |

For operational concerns (reverse proxy, Docker, health checks, TLS), see the [deployment guides](/mistral.rs/guides/deploy/).
