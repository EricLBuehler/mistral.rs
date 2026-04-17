---
title: Production checklist
description: What to verify before a mistralrs server takes real traffic.
sidebar:
  order: 2
---

## Authentication and TLS

mistral.rs has no built-in authentication. Run behind a reverse proxy (nginx, Caddy, Traefik) terminating TLS and validating credentials.

`Authorization: Bearer ...` from OpenAI clients is not validated by mistral.rs. The proxy is the component that interprets it.

Bind to loopback when fronted by a local proxy:

```bash
mistralrs serve --host 127.0.0.1 --port 8080 -m <model>
```

## Body limit and CORS

The default body limit is 50 MB and the default CORS allows any origin. Both are not configurable via the CLI; use `MistralRsServerRouterBuilder` (`mistralrs-server-core`) for custom values.

## Health and readiness

- `GET /health` returns 200 when the server is listening. It does not verify model load.
- `GET /v1/models` includes a per-model `status` field (`loaded`, `unloaded`, `reloading`). Use it for readiness probes that require the target model to be loaded.

## Logging

`tracing` logs at `INFO` by default. Override with `RUST_LOG`.

```bash
RUST_LOG=mistralrs_core=info,tower_http=info mistralrs serve -m <model>
```

## Cache and reproducibility

- The Hugging Face cache lives at `$HF_HOME` (default `~/.cache/huggingface`). Persist this directory across container restarts to avoid re-downloading weights.
- The token saved by `mistralrs login` lives at `$HF_HOME/token`.

## Sessions across restart

Sessions are in-memory with a 30-minute idle TTL and 128-entry capacity. They do not survive a restart. Export with `GET /v1/sessions/{id}` before shutdown when persistence is required, and re-import on the new instance with `PUT /v1/sessions/{id}`.

## Multi-model

For multi-model serving, use `mistralrs from-config -f config.toml` with `[[models]]` entries. See [running multiple models](/mistral.rs/guides/serve/multiple-models/).
