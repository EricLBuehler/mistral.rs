---
title: Production checklist
description: What to verify before a mistralrs server takes real traffic.
---

Work through this list before a `mistralrs serve` deployment receives traffic from users or another service. Each item links to the page that owns the details.

## Network and auth

- [ ] Bind to loopback unless the host network is private: `mistralrs serve --host 127.0.0.1 --port 8080 -m <model>`.
- [ ] Terminate TLS and validate credentials in a reverse proxy (nginx, Caddy, Traefik). **mistral.rs has no built-in authentication** - `Authorization: Bearer ...` headers from OpenAI clients are accepted but never validated by the server.
- [ ] Know the defaults you inherit: 50 MB request body limit, CORS allows any origin. Neither is CLI-configurable; embed `mistralrs-server-core`'s router builder for custom values (see [embed in axum](/guides/rust/embed-in-axum/)).

## Reproducible startup

- [ ] Move the launch command into a TOML config and start with `mistralrs from-config -f config.toml`. See the [TOML config reference](/reference/cli-toml-config/).
- [ ] Pin the container to a version tag, not `*-latest` (see the [Docker guide](/guides/deploy/docker/)).
- [ ] Pin the model. A Hugging Face model ID always downloads the `main` revision, so an upstream change reaches you on the next download. To freeze a version, download it once and pass the local directory to `-m`.
- [ ] Persist the model cache across restarts (volume at `HF_HOME`).

## Resource sizing

- [ ] Run `mistralrs doctor` on the target host to confirm the expected accelerator and compiled features.
- [ ] Run `mistralrs tune -m <model>` for a starting quantization and memory plan (it estimates from configs; it does not load or benchmark the model).
- [ ] Set `--max-seqs` to the number of requests you expect at once (default 32). Requests beyond it wait in a queue. See [concurrency](/guides/perf/throughput-tuning/#concurrency).
- [ ] Send expected load and watch `mistralrs_paged_preemptions_total`. If it keeps climbing, the KV cache is too small: lower `--max-seqs` or give the cache more memory.
- [ ] Decide whether the cache size may vary. By default it gets whatever memory is free at startup, so another process on the GPU means a smaller cache. Set `--kv-cache` (for example `16GiB` or `131072tokens`) to get the same size every time; loading then fails instead of starting with less. See [memory budget](/guides/perf/paged-attention/#memory-budget).

## Health, readiness, and metrics

- [ ] Liveness: `GET /health` returns 200 when the server is listening. It does not verify model load.
- [ ] Readiness: `GET /v1/models` includes a per-model `status` field (`loaded`, `unloaded`, `reloading`). Probe for the specific model id the caller needs, not just process liveness.
- [ ] Scrape `GET /metrics` with Prometheus and alert on error rate, time to first token, and queue length. [Observability](/guides/deploy/observability/) lists the metrics to watch.
- [ ] Give startup probes a generous window; first-run model loading can take minutes.

## Logging

- [ ] By default the server logs startup progress, warnings, and two access-log lines per request (health checks and metrics scrapes are skipped). See [observability](/guides/deploy/observability/) for JSON logs and request IDs.
- [ ] Use `-v` for debug, `-vv` for trace, or `RUST_LOG` for an explicit filter.
- [ ] Only use `-l/--log <path>` where request and response bodies can be stored safely; it logs payloads, not just metadata.

## State across restarts

- [ ] Sessions are in-memory with a 30-minute idle TTL and 128-entry capacity; they do not survive restarts. Export with `GET /v1/sessions/{id}` before shutdown and re-import with `PUT /v1/sessions/{id}` if persistence is required. See [sessions](/guides/agents/persist-sessions/).

## Dynamic LoRA

- [ ] Declare adapters that must survive restart as preloads. Runtime changes do not survive a restart and must otherwise be replayed.
- [ ] Pin remote adapter revisions, preferably to commit SHAs. The CLI accepts `--lora '{"name":"production","path":"org/adapter","revision":"<commit>"}'`.
- [ ] Keep HTTP mutation behind authentication and set `MISTRALRS_LORA_ADAPTER_ROOT` to an operator-owned directory. Read-only discovery does not need mutation enabled.
- [ ] Size `--lora-max-adapters` and `--lora-max-bytes` with enough headroom to replace an adapter without interrupting in-flight requests.
- [ ] Use `expected_generation` on replacement and unload operations, then verify the published generation through `GET /v1/lora_adapters`.
- [ ] Keep live mutation out of tensor-parallel deployments; preload adapters instead.

See [LoRA and X-LoRA adapters](/guides/customize/lora-adapters/) for rollout, rollback, and errors.

## Multi-model

- [ ] For serving several models from one process, use `mistralrs from-config` with `[[models]]` entries and decide the default model id. See [multiple models](/guides/serve/multiple-models/).
