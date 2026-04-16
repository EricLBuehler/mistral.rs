---
title: Production checklist
description: What to verify before a mistralrs server goes on the open internet or handles real traffic.
sidebar:
  order: 2
---

mistral.rs is easy to stand up during development and mostly secure enough for a laptop. Putting it somewhere that real users or real traffic reach it takes a bit more care. This page is a pre-flight checklist.

## Authentication and TLS

By default, mistralrs has no built-in authentication. If clients can reach the port, they can call the model. For anything public, put the server behind a reverse proxy that handles both TLS and auth.

A reasonable starting configuration:

- Terminate TLS at nginx, Caddy, or Traefik.
- Validate `Authorization: Bearer ...` headers at the proxy, either against a static list for small deployments or against an identity provider for larger ones.
- Bind mistralrs to `127.0.0.1` so nothing reaches it except the local proxy.

The `--host 127.0.0.1 --port 8080` flags on `mistralrs serve` do the binding. Your proxy config does the rest.

## Resource limits

Default behavior is to use all the memory it can. In a shared environment that is a bad citizen.

For container deployments, pass `--memory` and `--gpus` explicitly. For systemd, the unit file's `MemoryMax=` and `CPUAffinity=` do the same work. On a dedicated box with one process, limits are less critical but still useful as a backstop against pathological inputs.

KV cache growth is the usual culprit for memory surprises. A model with 128k context running many concurrent streams will eventually pressure VRAM. PagedAttention caps the growth at a configurable ceiling; see the [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/) for how to size it.

## Model warm-up

Loading a model into GPU memory takes a while. The first request after a server start pays that cost, which means anyone caught in the first wave sees a long latency.

Two ways to avoid this:

- Start the server and issue a warm-up request (any small completion) before flipping a load balancer toward it. This is the easiest pattern to wire into a deploy pipeline.
- Configure `unload_on_start = false` in the TOML config (the default) and use a readiness probe that only returns 200 once the initial load has finished.

The `/health` endpoint returns 200 as soon as the server is listening, not when the model is loaded. For multi-stage startup logic, hit `/v1/models` instead and wait for the target model to be in `loaded` status before declaring the pod ready.

## Observability

mistral.rs emits `tracing` logs at `INFO` level by default. Useful things to collect:

- Request latency and token throughput. Each request logs structured fields for these; any log aggregator that parses JSON will pick them up.
- Model status transitions (`loaded`, `unloaded`, `reloading`), which matter for multi-model deployments where the server is materializing models on demand.
- Error rate, particularly 4xx responses that indicate clients are sending bad requests.

For metrics, the easiest path today is to scrape a sidecar that polls `/v1/models` and reports gauges for each loaded model. A first-party Prometheus endpoint is on the roadmap but not there yet.

## Request body limits

Multimodal workloads push large bodies. The default 50 MB limit is fine for chat with small images; video or long audio can easily exceed it. Set `--max-body-limit` high enough for your use case, but not unlimited, because an unbounded limit makes certain memory-exhaustion attacks trivial.

A reasonable value for a multimodal service is 250 MB. Text-only servers can often stay at the default.

## Versioning and reproducibility

Three things drift over time without intervention:

**The mistralrs binary.** Pin a specific release (or build from a specific commit in your container) and bump deliberately. `latest` tags make the deploy pipeline non-reproducible.

**Hugging Face model revisions.** Without `--hf-revision`, you get whatever `main` points at during each server start. For production, pin a revision per model. `mistralrs cache list` shows what you currently have resolved.

**Chat templates.** mistralrs carries bundled chat templates for some models, but relies on the model's own `chat_template` in `tokenizer_config.json` by default. If the upstream template changes, your outputs change. Pin the revision to freeze the template.

## Graceful shutdown

When the server receives SIGTERM, it stops accepting new connections and lets in-flight requests finish before exiting. Default Kubernetes `terminationGracePeriodSeconds` is 30; bump that to 120 or higher if your requests can take a while (generation on long contexts, multi-round agentic loops).

If you are running a long-lived agent with persistent sessions, note that SIGTERM does not persist session state to disk. Sessions live in memory and are lost on restart. If you need persistence across restarts, export them through the `GET /v1/sessions/{id}` endpoint before shutdown.

## Rolling updates

For zero-downtime deploys, run at least two server replicas behind a load balancer with `/v1/models` (or a lightweight completion request) as the health check. Drain one replica at a time, waiting for its model to reach `loaded` status on the new version before draining the next.

## Final sanity checks

A short list of things to try once a deploy is live:

- Hit `/health`. Should return 200.
- Hit `/v1/models`. All expected models should report `loaded`.
- Send a one-token completion with `max_tokens: 1`. This exercises the full path with minimal work, and its latency is a reasonable first-token baseline.
- Send a large completion with `stream: true`. Confirm chunks arrive promptly.
- Check logs for warnings or errors during startup. Some are non-fatal (hardware quirks) but all are worth understanding.

If any of these fail, the [troubleshooting reference](/mistral.rs/reference/troubleshooting/) is the first place to look.
