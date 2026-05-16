---
title: Serve models
description: Run mistral.rs as an HTTP server, with one or more models, with the web UI, and with OpenAI-compatible APIs.
---

[Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/) covers basic single-model serving. These guides cover the configuration needed beyond a single local server.

## Choose by task

| If you need to... | Start here |
|---|---|
| Change host, port, CORS, request limits, or authentication | [HTTP server configuration](/mistral.rs/guides/serve/http-server/) |
| Serve more than one model from one process | [Running multiple models](/mistral.rs/guides/serve/multiple-models/) |
| Use the browser chat interface | [Using the web UI](/mistral.rs/guides/serve/with-web-ui/) |
| Use the newer OpenAI Responses endpoint | [OpenAI Responses API](/mistral.rs/guides/serve/openai-responses-api/) |

For operational concerns (reverse proxy, Docker, health checks, TLS), see the [deployment guides](/mistral.rs/guides/deploy/).
