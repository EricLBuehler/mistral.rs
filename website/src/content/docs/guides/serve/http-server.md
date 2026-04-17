---
title: Configure the HTTP server
description: Host, port, CORS, body limits, and authentication options for mistralrs serve.
sidebar:
  order: 1
---

`mistralrs serve` defaults: bind on `0.0.0.0:1234`, CORS open, no authentication, 50 MB request body limit. These are appropriate for local development. This guide covers the flags and config used for non-local deployments.

## Host and port

```bash
mistralrs serve --host 127.0.0.1 --port 8080 -m <model>
```

`--host` controls the bind interface. `0.0.0.0` (default) accepts connections from any host on the network; `127.0.0.1` restricts to the local machine. Use loopback when fronting mistral.rs with a reverse proxy (nginx, Caddy) on the same host — the proxy handles TLS and external exposure.

`--port` is the TCP port. Default `1234`.

## CORS

The server allows requests from any origin by default. For browser-facing deployments where the UI is on a different domain, this is appropriate. To restrict origins:

```bash
mistralrs serve --allowed-origin https://app.example.com -m <model>
```

`--allowed-origin` accepts multiple values. Origins not in the list receive a CORS preflight rejection.

`Access-Control-Allow-Credentials` follows from an explicit origin list automatically.

## Request body size

Multimodal requests can be large. A single image is usually under a megabyte, but a base64-encoded video in a JSON body can exceed 50 MB. To raise the limit:

```bash
mistralrs serve --max-body-limit 200000000 -m <model>
```

The value is in bytes. The default is sufficient for text workloads.

## Authentication

mistral.rs does not implement authentication. The intended pattern is a reverse proxy handling authentication, TLS, and rate limiting. Examples:

- nginx with `auth_basic` or `ngx_http_auth_request_module` forwarding to an internal service
- Caddy with a JWT plugin
- Traefik with any middleware

OpenAI-protocol clients always send an `Authorization: Bearer ...` header because the OpenAI SDK requires an API key at initialization. mistral.rs ignores the header. An auth proxy is the component that interprets it.

## Logging

The server emits structured logs via `tracing`. Default level is `INFO`. For more detail:

```bash
RUST_LOG=debug mistralrs serve -m <model>
```

Module filters work as in any `tracing`-instrumented program: `RUST_LOG=mistralrs_core=debug,tower_http=info`.

## Config file versus flags

Most CLI flags have a TOML config equivalent. Useful for large flag sets, sharing across machines, or replaying `mistralrs tune --emit-config` output.

```bash
mistralrs from-config -f config.toml
```

Full schema: [CLI TOML config reference](/mistral.rs/reference/cli-toml-config/). Minimal example:

```toml
model = "Qwen/Qwen3-4B"
isq = "4"

[server]
host = "127.0.0.1"
port = 8080

[features]
enable_search = true
enable_code_execution = true
```

## What to read next

- [Running multiple models](/mistral.rs/guides/serve/multiple-models/) — host several models in one process.
- [Production checklist](/mistral.rs/guides/deploy/production-checklist/) — pre-deployment verification.
