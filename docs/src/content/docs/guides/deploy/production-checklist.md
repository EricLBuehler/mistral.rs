---
title: Production checklist
description: What to verify before a mistralrs server takes real traffic.
---

Work through this list before a `mistralrs serve` deployment receives traffic from users or another service. Each item links to the page that owns the details.

## Network and auth

- [ ] Bind to loopback unless the host network is private: `mistralrs serve --host 127.0.0.1 --port 8080 -m <model>`.
- [ ] Terminate TLS and validate credentials in a reverse proxy (nginx, Caddy, Traefik). **mistral.rs has no built-in authentication** - `Authorization: Bearer ...` headers from OpenAI clients are accepted but never validated by the server.
- [ ] Know the defaults you inherit: 50 MB request body limit, CORS allows any origin. Neither is CLI-configurable; embed `mistralrs-server-core`'s router builder for custom values (see [embed in axum](/mistral.rs/guides/rust/embed-in-axum/)).

## Reproducible startup

- [ ] Move the launch command into a TOML config and start with `mistralrs from-config -f config.toml`. See the [TOML config reference](/mistral.rs/reference/cli-toml-config/).
- [ ] Pin versions: a container version tag (not `*-latest`) per the [Docker guide](/mistral.rs/guides/deploy/docker/), and be aware model ids resolve to the `main` revision at download time.
- [ ] Persist the model cache across restarts (volume at `HF_HOME`).

## Resource sizing

- [ ] Run `mistralrs doctor` on the target host to confirm the expected accelerator and compiled features.
- [ ] Run `mistralrs tune -m <model>` for a starting quantization and memory plan (it estimates from configs; it does not load or benchmark the model).
- [ ] Set `--max-seqs` deliberately (default 32).
- [ ] Size the [paged-attention](/mistral.rs/guides/perf/paged-attention/) KV pool explicitly instead of relying on the implicit 90% budget, with one of:
  - `--pa-context-len`
  - `--pa-memory-mb`
  - `--pa-memory-fraction`

  See [throughput tuning](/mistral.rs/guides/perf/throughput-tuning/).

## Health, readiness, and metrics

- [ ] Liveness: `GET /health` returns 200 when the server is listening. It does not verify model load.
- [ ] Readiness: `GET /v1/models` includes a per-model `status` field (`loaded`, `unloaded`, `reloading`). Probe for the specific model id the caller needs, not just process liveness.
- [ ] Scrape `GET /metrics`: Prometheus text format with per-request counters, latency histograms, in-flight request gauges, and request-body histograms. Details live in [observability](/mistral.rs/guides/deploy/observability/) and the [HTTP API reference](/mistral.rs/reference/http-api/).
- [ ] Give startup probes a generous window; first-run model loading can take minutes.

## Logging

- [ ] Default output is curated `INFO` startup logs, dependency warnings, and HTTP access logs for non-housekeeping requests. Use [observability](/mistral.rs/guides/deploy/observability/) for request ids, Prometheus metrics, and access-log controls.
- [ ] Use `-v` for debug, `-vv` for trace, or `RUST_LOG` for an explicit filter.
- [ ] Only use `-l/--log <path>` where request and response bodies can be stored safely; it logs payloads, not just metadata.

## State across restarts

- [ ] Sessions are in-memory with a 30-minute idle TTL and 128-entry capacity; they do not survive restarts. Export with `GET /v1/sessions/{id}` before shutdown and re-import with `PUT /v1/sessions/{id}` if persistence is required. See [sessions](/mistral.rs/guides/agents/persist-sessions/).

## Multi-model

- [ ] For serving several models from one process, use `mistralrs from-config` with `[[models]]` entries and decide the default model id. See [multiple models](/mistral.rs/guides/serve/multiple-models/).
