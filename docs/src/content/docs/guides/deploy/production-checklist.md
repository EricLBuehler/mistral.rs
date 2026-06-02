---
title: Production checklist
description: What to verify before a mistralrs server takes real traffic.
sidebar:
  order: 2
---

Use this page when a `mistralrs serve` deployment will receive traffic from users or another service.

## Baseline server shape

Run the inference process behind a proxy and bind mistral.rs to loopback unless the host network is already private:

```bash
mistralrs serve --host 127.0.0.1 --port 8080 --quant 4 -m <model>
```

For repeatable startup, use a TOML config:

```bash
mistralrs from-config -f config.toml
```

Use `mistralrs tune -m <model>` on the target host before selecting quantization, context length, and device mapping defaults.

## Authentication and TLS

mistral.rs has no built-in authentication. Run behind a reverse proxy (nginx, Caddy, Traefik) terminating TLS and validating credentials.

`Authorization: Bearer ...` from OpenAI clients is not validated by mistral.rs. The proxy is the component that interprets it.

## Body limit and CORS

The default body limit is 50 MB and the default CORS allows any origin. Both are not configurable via the CLI; use `MistralRsServerRouterBuilder` (`mistralrs-server-core`) for custom values.

## Health and readiness

- `GET /health` returns 200 when the server is listening. It does not verify model load.
- `GET /v1/models` includes a per-model `status` field (`loaded`, `unloaded`, `reloading`). Use it for readiness probes that require the target model to be loaded.

For multi-model serving, readiness should check the specific model id required by the caller rather than only checking process liveness.

## Logging

By default, the CLI shows curated `INFO` startup logs from mistral.rs and warnings from dependencies. Use `-v` for debug details, `-vv` for trace-level file/cache internals, or `RUST_LOG` for an explicit filter.

```bash
mistralrs serve -v -m <model>
```

Use `-l, --log <path>` only when request and response bodies can be stored safely. It logs request/response data, not only metadata.

## Resource sizing

- Use `mistralrs doctor` to verify the expected accelerator is visible.
- Use `mistralrs tune -m <model>` to pick a starting quantization and memory plan for the host.
- Set `--max-seqs` deliberately for server workloads. The default is 32 concurrent sequences.
- If paged attention is enabled, choose one of `--pa-context-len`, `--pa-memory-mb`, or `--pa-memory-fraction` rather than relying on an implicit memory budget.

## Sessions across restart

Sessions are in-memory with a 30-minute idle TTL and 128-entry capacity. They do not survive a restart. Export with `GET /v1/sessions/{id}` before shutdown when persistence is required, and re-import on the new instance with `PUT /v1/sessions/{id}`.

## Multi-model

For multi-model serving, use `mistralrs from-config -f config.toml` with `[[models]]` entries. See [running multiple models](/mistral.rs/guides/serve/multiple-models/).
