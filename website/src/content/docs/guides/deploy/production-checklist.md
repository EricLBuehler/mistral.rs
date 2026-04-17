---
title: Production checklist
description: What to verify before a mistralrs server goes on the open internet or handles real traffic.
sidebar:
  order: 2
---

mistral.rs is straightforward to stand up during development but requires more configuration for public or production traffic. This page is a pre-flight checklist.

## Authentication and TLS

mistral.rs has no built-in authentication. Anyone reaching the port can call the model. For public deployments, place the server behind a reverse proxy handling TLS and authentication.

Reasonable starting configuration:

- Terminate TLS at nginx, Caddy, or Traefik.
- Validate `Authorization: Bearer ...` at the proxy — static list for small deployments, identity provider for larger ones.
- Bind mistralrs to `127.0.0.1` so only the local proxy reaches it.

`--host 127.0.0.1 --port 8080` handles the binding. Proxy config handles the rest.

## Resource limits

Default behavior uses all available memory. In shared environments, set explicit limits.

For containers, pass `--memory` and `--gpus` explicitly. For systemd, use `MemoryMax=` and `CPUAffinity=`. On a dedicated host with one process, limits are less critical but still useful as a backstop against pathological inputs.

KV cache growth is the typical culprit for memory surprises. A 128k-context model with many concurrent streams will pressure VRAM. PagedAttention caps growth at a configurable ceiling — see the [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/).

## Model warm-up

Loading a model into GPU memory takes time. The first request after startup pays that cost.

Two avoidance strategies:

- Issue a warm-up request (any small completion) after server start, before flipping a load balancer toward it.
- Configure `unload_on_start = false` (the default) and use a readiness probe that returns 200 only after initial load completes.

The `/health` endpoint returns 200 when the server is listening, not when the model is loaded. For multi-stage startup, hit `/v1/models` and wait for the target model's status to be `loaded` before declaring the pod ready.

## Observability

mistral.rs emits `tracing` logs at `INFO` by default. Useful collected fields:

- Request latency and token throughput. Each request logs structured fields.
- Model status transitions (`loaded`, `unloaded`, `reloading`) — important for multi-model deployments with on-demand materialization.
- Error rate, particularly 4xx responses indicating bad client requests.

For metrics, scrape a sidecar polling `/v1/models` and reporting per-model gauges. A first-party Prometheus endpoint is on the roadmap.

## Request body limits

Multimodal workloads send large bodies. The default 50 MB limit suffices for chat with small images; video or long audio can exceed it. Set `--max-body-limit` high enough but not unlimited — unbounded limits enable memory-exhaustion attacks.

A reasonable value for multimodal services is 250 MB. Text-only servers can stay at the default.

## Versioning and reproducibility

Three things drift over time:

**The mistralrs binary.** Pin a specific release (or build from a specific commit) and bump deliberately. `latest` tags break reproducibility.

**Hugging Face model revisions.** Without `--hf-revision`, `main` is resolved at each server start. For production, pin a revision per model. `mistralrs cache list` shows currently resolved revisions.

**Chat templates.** mistral.rs bundles templates for some models but defaults to the model's `chat_template` from `tokenizer_config.json`. Upstream template changes change outputs. Pin the revision to freeze the template.

## Graceful shutdown

On SIGTERM, the server stops accepting new connections and lets in-flight requests finish before exiting. Default Kubernetes `terminationGracePeriodSeconds` is 30; raise to 120+ for long generations or multi-round agentic loops.

For long-lived agents with persistent sessions: SIGTERM does not persist session state to disk. Sessions are in-memory and lost on restart. Export sessions via `GET /v1/sessions/{id}` before shutdown when persistence is required.

## Rolling updates

For zero-downtime deploys, run at least two replicas behind a load balancer with `/v1/models` (or a small completion) as the health check. Drain one replica at a time, waiting for the new version's model to reach `loaded` status before proceeding.

## Final sanity checks

After a live deploy:

- Hit `/health`. Expect 200.
- Hit `/v1/models`. Expected models should report `loaded`.
- Send a one-token completion (`max_tokens: 1`). Exercises the full path with minimal work; latency is a reasonable first-token baseline.
- Send a large completion with `stream: true`. Confirm chunks arrive promptly.
- Inspect logs for warnings or errors during startup. Some are non-fatal (hardware quirks) but all merit understanding.

For failures, see the [troubleshooting reference](/mistral.rs/reference/troubleshooting/).
