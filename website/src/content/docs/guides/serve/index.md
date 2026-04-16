---
title: Serve models
description: Run mistral.rs as an HTTP server, with one or more models, with the web UI, and with OpenAI-compatible APIs.
---

[Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/) walked through starting a basic server with a single model. These guides pick up from there, covering the configuration knobs you are likely to reach for as you move beyond a single development process.

## What is in this section

- [HTTP server configuration](/mistral.rs/guides/serve/http-server/): host, port, CORS, request body limits, and authentication.
- [Running multiple models](/mistral.rs/guides/serve/multiple-models/): one server process, several loaded models, routed by the `model` field in each request.
- [Using the web UI](/mistral.rs/guides/serve/with-web-ui/): what `--ui` gives you and how to customize it.
- [OpenAI Responses API](/mistral.rs/guides/serve/openai-responses-api/): the newer endpoint shape that lives alongside Chat Completions.

For anything operational (running behind a reverse proxy, Docker images, health checks, TLS), see the [deployment guides](/mistral.rs/guides/deploy/).
