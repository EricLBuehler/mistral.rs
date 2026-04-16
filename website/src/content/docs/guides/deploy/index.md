---
title: Deploy
description: Run mistralrs in production, including container images and an operational checklist.
---

Once mistral.rs has to run somewhere other than your laptop, a few things you got for free during development (ephemeral storage, no TLS, a single loaded model, a forgiving network) stop being free. These guides cover what to do about that.

- [Docker](/mistral.rs/guides/deploy/docker/) walks through building and running the official container images.
- [Production checklist](/mistral.rs/guides/deploy/production-checklist/) is the pre-flight list: TLS termination, authentication, observability, model warm-up, and what to monitor.

For configuration options (host, port, CORS, body limits) see the [HTTP server guide](/mistral.rs/guides/serve/http-server/).
