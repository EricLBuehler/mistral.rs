---
title: Configure the HTTP server
description: Host, port, CORS, body limits, and authentication options for mistralrs serve.
sidebar:
  order: 1
---

The default `mistralrs serve` starts on `0.0.0.0:1234` with CORS open, no authentication, and a 50 MB request body limit. Those defaults are fine for local development and quick experiments. This guide covers the flags and config you will reach for once you start putting the server anywhere more public.

## Host and port

```bash
mistralrs serve --host 127.0.0.1 --port 8080 -m <model>
```

`--host` controls which interface the server binds to. `0.0.0.0` (the default) means everything on your network can reach it; `127.0.0.1` restricts it to the local machine. Use the loopback address if you are fronting mistral.rs with a reverse proxy like nginx or Caddy on the same box, because that proxy will do the work of terminating TLS and exposing the service.

`--port` is the TCP port. The default is `1234` because it is memorable and almost never used by anything else.

## CORS

By default, the server allows requests from any origin. For a browser-facing deployment where the web UI is hosted on a different domain, that is what you want. For an API that is only meant to be called server-to-server, you probably want to lock it down:

```bash
mistralrs serve --allowed-origin https://app.example.com -m <model>
```

`--allowed-origin` can be passed more than once to build a list. Requests from origins that are not on the list get a CORS preflight rejection.

Any `Access-Control-Allow-Credentials` behavior follows automatically from the presence of an explicit origin list.

## Request body size

Multimodal requests can be large. A single image is usually under a megabyte, but a video encoded as base64 inside a JSON body can easily push past 50 MB. If you see `413 Payload Too Large` errors, raise the limit:

```bash
mistralrs serve --max-body-limit 200000000 -m <model>
```

The value is in bytes. For most text workloads, the default is more than enough.

## Authentication

mistral.rs itself does not implement authentication. The intended pattern is that you run it behind a reverse proxy and let that proxy handle authentication, TLS, and any rate limiting you want. Examples of proxies that work well:

- nginx with a simple `auth_basic` or an `ngx_http_auth_request_module` that forwards to an internal service
- Caddy with a JWT plugin
- Traefik with any of its middleware options

There is one wrinkle worth knowing about. Clients that speak the OpenAI protocol will always send an `Authorization: Bearer ...` header, whether you configured one or not, because the OpenAI SDK requires an API key at initialization time. mistral.rs ignores that header on its own. If you configure an auth proxy, the proxy is the thing that gives that header meaning.

## Logging

The server emits structured logs through `tracing`. By default it logs at `INFO` level. To see more, set the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug mistralrs serve -m <model>
```

Module filters work the same way they do for any `tracing`-instrumented program: `RUST_LOG=mistralrs_core=debug,tower_http=info` narrows the verbosity down to the parts you care about.

## Config file versus flags

Almost everything you can set on the command line also has a TOML config file equivalent. Running through a config file is useful when you have a lot of flags, when the same flag set is shared across several machines, or when you are using `mistralrs tune --emit-config` to capture recommended settings and want to run them back.

```bash
mistralrs from-config -f config.toml
```

The full schema is in the [CLI TOML config reference](/mistral.rs/reference/cli-toml-config/). The shortest useful config looks something like this:

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

- [Running multiple models](/mistral.rs/guides/serve/multiple-models/) if you want to host several models behind one process.
- [Production checklist](/mistral.rs/guides/deploy/production-checklist/) for a run-through of everything worth verifying before a server goes on the open internet.
