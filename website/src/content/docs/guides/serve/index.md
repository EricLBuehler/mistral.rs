---
title: Serve models
description: Run mistral.rs as an HTTP server, with one or more models, with the web UI, and with OpenAI-compatible APIs.
---

[Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/) covers basic single-model serving. These guides cover the configuration knobs needed beyond a single development process.

## Contents

- [HTTP server configuration](/mistral.rs/guides/serve/http-server/): host, port, CORS, request body limits, and authentication.
- [Running multiple models](/mistral.rs/guides/serve/multiple-models/): multiple models in one process, routed by the `model` field.
- [Using the web UI](/mistral.rs/guides/serve/with-web-ui/): what `--ui` provides and how to customize it.
- [OpenAI Responses API](/mistral.rs/guides/serve/openai-responses-api/): the newer endpoint shape alongside Chat Completions.

For operational concerns (reverse proxy, Docker, health checks, TLS), see the [deployment guides](/mistral.rs/guides/deploy/).
