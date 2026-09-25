# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/EricLBuehler/mistral.rs/security/advisories/new). Do not open a public issue.

### Requirements

Before submitting a report, make sure that:

- you have read this policy and your report falls within [scope](#in-scope);
- you have searched existing advisories and issues; duplicates will be rejected;
- the report is written by you. AI may only be used in an assistive capacity, and reports written exclusively by AI are not accepted;
- the report includes a working proof of concept, as a script and/or attached files, ideally one that needs no model weights or GPU.

Please also include the mistral.rs version or commit, how it was installed (prebuilt binary, `cargo install`, Python wheel, Docker), and the command or configuration used to run it.

Maintainers reserve the right to close reports that do not meet these requirements.

### What happens next

mistral.rs is maintained by volunteers on a reasonable-effort basis. We request that reports are under embargo: keep the details private, including in talks, papers, and public issues, until we publish the advisory or 90 days have passed since the report, whichever comes first. This gives us time to ship a fix before the issue is public. If a fix needs longer, we will tell you and agree on a date together.

Once a fix ships in a release, we publish a GitHub Security Advisory crediting the reporter. We do not request CVEs.

## Supported versions

Only the latest release receives security fixes. Please reproduce on the latest release or `master` before reporting.

## Security model

### No built-in authentication

mistral.rs does not authenticate HTTP clients, and this is by design. Anyone who can reach the server can use every endpoint it exposes. To serve anyone other than yourself, put an authenticating reverse proxy in front of it and make sure clients can only reach the proxy.

`mistralrs serve` binds `0.0.0.0` by default and allows cross-origin requests from any origin. On a machine that is not on a private network, pass `--host 127.0.0.1`. The [production checklist](https://docs.mistralrs.dev/guides/deploy/production-checklist/) covers the rest.

### Operator-only capabilities

Anyone who can reach the server can use these, so they must only be reachable by the operator:

- model lifecycle and tuning endpoints: `/v1/models/unload`, `/v1/models/reload`, `/v1/models/tune`, `/re_isq`, `/calibration/*`;
- LoRA adapter loading: `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`;
- system endpoints: `/v1/system/info`, `/v1/system/doctor`;
- built-in code and shell execution, which requests can enable per request. It is [sandboxed by default](https://docs.mistralrs.dev/reference/sandbox/) on Linux and macOS and is not sandboxed on Windows.

### Trusted inputs

These are assumed to come from the operator. Problems that require a malicious version of them are out of scope:

- model weights and files, tokenizers, chat templates, and anything else loaded from a model repository;
- CLI flags, TOML configuration, and environment variables;
- MCP server configuration and the tool dispatch URL.

### Untrusted inputs

These may come from anyone, and mistral.rs must handle them safely:

- request bodies from HTTP clients, including message content, images, audio, video, and files;
- output from tools, including web search results, fetched pages, MCP tool results, and code execution output;
- text the model generates, including when it quotes untrusted content.

## In scope

Examples of issues we treat as vulnerabilities:

- an untrusted input causing memory unsafety, arbitrary file access, or code execution outside the operator's configuration;
- an untrusted input making the agent loop act on something the model did not produce, such as a forged tool call;
- the server reaching network destinations it should refuse, such as private addresses from built-in web fetching;
- escaping the code and shell execution sandbox on Linux or macOS, meaning access the active sandbox profile does not allow. Access the operator granted is not an escape: `--sandbox off`, extra allowed paths, the configured network mode, or SDK code execution without a sandbox policy.

## Out of scope

- Denial-of-service bugs. These are generally not treated as vulnerabilities; we look at them case by case and fix the ones worth fixing, as regular issues.
- The built-in web UI (`/ui`), including its server-side handlers. Like every other endpoint, access to it is the reverse proxy's job.
- Unauthenticated access to any endpoint, including the operator-only capabilities above. Access control belongs in the reverse proxy.
- Anything that requires a malicious model, tokenizer, chat template, or configuration.
- Vulnerabilities in dependencies that mistral.rs does not reach. Please report those upstream.

These lists are not exhaustive. Maintainers have the final say on whether a report is in scope and may deem a report out of scope even if it is not listed here.

Please report out-of-scope problems as regular issues.
