---
title: Deploy
description: Run mistralrs in production, including container images and an operational checklist.
---

When mistral.rs runs anywhere other than a development laptop, several development conveniences (ephemeral storage, no TLS, single model, forgiving network) no longer apply.

- [Docker](/mistral.rs/guides/deploy/docker/): building and running the official container images.
- [Production checklist](/mistral.rs/guides/deploy/production-checklist/): pre-flight: TLS termination, authentication, observability, model warm-up, monitoring.

For configuration options (host, port, CORS, body limits), see the [HTTP server guide](/mistral.rs/guides/serve/http-server/).
