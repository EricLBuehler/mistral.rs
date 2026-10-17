---
title: Security layer
description: Opt-in access control, attack screening, bans and budgets for mistralrs serve, with axum-guard-rs.
---

`mistralrs serve` listens on `0.0.0.0:1234` with no authentication, so anyone who can reach the port can use inference and the control routes. The optional security layer adds application-layer protection inside the server. It is off by default: nothing is added to the router unless you build with the `security` feature and pass `--security`.

The layer is [axum-guard-rs](https://github.com/Guard-Core/axum-guard-rs) 2.0.0 on top of guard-core-rs 4.4.0. It complements a reverse proxy, WAF or firewall in front of the server (TLS stays there) and is the protection when there is none.

## Build

```bash
cargo install --path mistralrs-cli --features "security"
```

Add your accelerator features as usual. Passing `--security` to a binary built without the feature is an error at startup, so a missing feature never leaves the server silently unprotected.

## What it does

- Control routes: an IP or CIDR allowlist and an API key, compared in constant time. The control routes are `/v1/models/unload`, `/v1/models/reload`, `/v1/models/tune`, `/re_isq`, `/calibration/*`, `/v1/system/doctor`, `/v1/system/info`, `/v1/files*`, `/v1/skills*`, `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/v1/agent/approvals/{approval_id}` and `/v1/containers/*`. Inference routes stay reachable without a key. With `--security-protect-all-routes` the same rules cover every route, which suits a fully private server.
- Attack screening: the path, query string and header values are screened for path traversal, injection, scanner probes and similar, and clients that keep probing are banned automatically. Request bodies are not screened, because prompts and files are free text. The `Host`, `User-Agent`, `Accept`, `Accept-Encoding`, `Connection`, `Origin` and `Referer` headers and every `sec-*` header are never screened. The API key header is screened, so use keys made of hex or base64 characters.
- Budgets: each client gets a per-minute budget on every GPU route (`/v1/chat/completions`, `/v1/responses`, `/v1/images/generations`, `/v1/audio/speech`, `/v1/embeddings`, `/v1/completions`, `/v1/messages`) and a larger one across all routes. Going over answers `429` with `Retry-After`.
- Uploads: `POST /v1/files` and `POST /v1/skills*` must be `multipart/form-data` and can have a lower size cap than the 50 MB body limit.
- Behavior rules: a client that fails the API key more than 5 times in 5 minutes on one route is banned for an hour, and so is a client that gets more than 30 `404` answers on one path in a minute. Both are counted per method and path.
- Optional: country allow or block lists (they need an ipinfo.io token), cloud provider blocking, emergency mode and passive mode.

The client address is the TCP peer. `X-Forwarded-For` is believed only when the peer is in `--security-trusted-proxies`, so a client cannot forge its identity to dodge a budget or a ban. IPv4-mapped peers (`::ffff:a.b.c.d`) are treated as their IPv4 address. State lives in memory and resets on restart, and it is shared with the built-in UI.

## Example

```bash
export MISTRALRS_SECURITY_API_KEYS="$(openssl rand -hex 24)"
mistralrs serve -m <model> --security \
  --security-control-allow-ips 127.0.0.1,10.0.0.0/8 \
  --security-trusted-proxies 10.1.1.1
```

Call a control route with the key:

```bash
curl -X POST http://localhost:1234/v1/models/unload -H "x-api-key: $MISTRALRS_SECURITY_API_KEYS" -H "content-type: application/json" -d '{"model_id": "default"}'
```

If a control IP allowlist and an API key are both set, a client needs both. If neither is set the control routes stay open and the server logs a warning.

## Options

| Flag and TOML key | Default | Effect |
|---|---|---|
| `--security`, `security` | off | Turns the layer on. |
| `--security-control-allow-ips`, `security_control_allow_ips` | none | IPs or CIDR ranges allowed on the control routes. |
| `security_api_keys` (TOML) or `MISTRALRS_SECURITY_API_KEYS` (comma separated) | none | API keys accepted on the control routes. There is no flag, so keys never show up in the process list. |
| `--security-api-key-header`, `security_api_key_header` | `x-api-key` | Header carrying the key. For `authorization` a `Bearer ` prefix is accepted. |
| `--security-protect-all-routes`, `security_protect_all_routes` | off | Apply the allowlist and key to every route. |
| `--security-trusted-proxies`, `security_trusted_proxies` | none | Proxies whose `X-Forwarded-For` is believed. |
| `--security-rate-limit`, `security_rate_limit` | 1200 | Requests per client per minute across all routes. About 20 per second, enough for the UI and polling clients. |
| `--security-gpu-rate-limit`, `security_gpu_rate_limit` | 120 | Requests per client per minute on each GPU route. |
| `--security-max-upload-mb`, `security_max_upload_mb` | none | Largest file or skill upload, in MB. |
| `--security-blocked-countries`, `--security-allowed-countries` | none | Country codes to block, or to allow exclusively. Need an ipinfo.io token in `security_ipinfo_token` (TOML) or `MISTRALRS_SECURITY_IPINFO_TOKEN`. |
| `--security-block-cloud-providers`, `security_block_cloud_providers` | none | Cloud providers to block, for example `AWS`, `GCP`, `Azure`. |
| `--security-emergency-mode`, `--security-emergency-whitelist` | off | Answer `503` to everyone outside the whitelist. |
| `--security-passive-mode`, `security_passive_mode` | off | Log what would be blocked without blocking for screening, budgets and bans. The control route allowlist and API key are still enforced. |

The same keys work under `[server]` in a [TOML config](/reference/cli-toml-config/).

## Limits

- Request bodies are not screened. The SSRF and media-loader advisories carry their URLs inside request bodies, so this layer does not replace fixing those code paths. It limits who can reach them and bans clients that probe.
- The layer reads each request body into memory up to the 50 MB body limit before checking it, and gives a client 30 seconds to send it. Very slow uploads of large files are refused with `408`.
- Behavior-rule counters are per method and path, so a scanner that requests a different missing path every time is caught by attack screening and the budgets rather than by the `404` rule.
- The separate MCP port (`--mcp-port`) is not covered. Leave it off or firewall it.
- The built-in UI is covered by screening, bans and the global budget, and by the allowlist and key only with `--security-protect-all-routes`.

With axum-guard-rs 2.0.0 and guard-core-rs 4.4.0.
