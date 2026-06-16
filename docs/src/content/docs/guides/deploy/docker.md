---
title: Run mistralrs in Docker
description: Pull or build the container images and run the unified CLI, with and without CUDA.
---

The published images ship the unified `mistralrs` binary as their entrypoint, so any CLI subcommand works directly: `serve`, `run`, `bench`, `quantize`. Running a container with no arguments prints the CLI help.

```bash
docker run --rm -p 1234:1234 -v hf-cache:/data -e HF_TOKEN=<token> \
  ghcr.io/ericlbuehler/mistral.rs:latest \
  serve -m Qwen/Qwen3-4B
```

`:latest` is the CPU image. For NVIDIA GPUs, pick the tag matching your GPU's compute capability and add `--gpus all`:

```bash
docker run --rm --gpus all -p 1234:1234 -v hf-cache:/data \
  ghcr.io/ericlbuehler/mistral.rs:cuda-89-latest \
  serve -m Qwen/Qwen3-4B
```

The host needs the NVIDIA Container Toolkit; see [NVIDIA's install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). To pin a specific GPU: `--gpus '"device=0"'`.

## Published tags

All images live at `ghcr.io/ericlbuehler/mistral.rs` ([package page](https://github.com/EricLBuehler/mistral.rs/pkgs/container/mistral.rs)).

- CPU: `latest` (alias of `cpu-latest`), `cpu-latest`, `cpu-X.Y.Z`, `cpu-X.Y`, `cpu-sha-<short>`.
- CUDA: `cuda-{cc}-latest`, `cuda-{cc}-X.Y.Z`, `cuda-{cc}-X.Y`, `cuda-{cc}-sha-<short>`.

CUDA compute capability variants (SM80+):
- `80` (A100)
- `86` (A-series workstation/RTX 30)
- `89` (RTX 40/L4)
- `90` (H100)
- `100` (B200)
- `120` (RTX 50)
- `121` (DGX Spark)

See [hardware support](/mistral.rs/reference/hardware-support/) for the full GPU mapping.

The CPU image and the Grace CUDA images (`90`, `100`, `121`) are multi-arch (amd64 + arm64) - the same tag runs on x86_64 and aarch64 (GH200/GB200/GB10), with Docker selecting the right architecture automatically. The other CUDA tags are x86_64 only.

The `*-latest` tags publish on releases and on manual CI dispatch from master; version tags pin a release.

For production, pin a version or sha tag rather than `*-latest`. Model ids also float: `-m Qwen/Qwen3-4B` resolves to whatever revision is tagged `main` at download time. The CLI has no revision flag; to pin a revision, use the Rust SDK's `with_hf_revision`.

## Image contract

- Entrypoint is the `mistralrs` binary; pass a subcommand and its flags as the container command.
- `mistralrs serve` listens on port 1234 by default (the image's `EXPOSE`d port). To change it, change the flag and the mapping together: `serve -p 8080` with `-p 8080:8080`. There is no `PORT` environment variable.
- `HF_HOME=/data` is set in the image: mount a volume at `/data` to persist downloaded weights (they land in `/data/hub`). HF authentication for gated models: `-e HF_TOKEN=<token>`.
- Chat templates ship at `/chat_templates` for models that need one: `--chat-template /chat_templates/<file>.json`.

## Building an image

From a repository checkout:

```bash
# CPU
docker build -t mistralrs:latest -f Dockerfile .

# CUDA (set the compute capability for your GPU)
docker build -t mistralrs:cuda -f Dockerfile.cuda-all \
  --build-arg CUDA_COMPUTE_CAP=89 .
```

- `Dockerfile.cuda-all` accepts `CUDA_COMPUTE_CAP`, `BASE_TAG`, and `WITH_FEATURES` build args. Default features are `cuda,cudnn`; CI builds add `flash-attn` and, except on compute capability 90, `cutile`.
- `Dockerfile.cuda-13.0-ubi9` is a Red Hat UBI 9 variant for air-gapped and enterprise deployments.
- The first CUDA build is slow because flash-attention compilation takes a while; later builds use the layer cache.

## Production deployment notes

**Persist the cache.** Weights are large enough that re-downloading on every restart is wasteful. Mount a named volume or host path at `/data`.

**Health check.** `/health` returns 200 when the server is up. Add a Docker healthcheck:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:1234/health || exit 1
```

The generous `--start-period` matters: first-run model loading can take minutes.

**Resource limits.** Set `--memory` and `--gpus` on `docker run` to bound the container's resources.

**Video input.** Install FFmpeg inside the image when serving video-capable models. See [set up video input](/mistral.rs/guides/models/video-setup/) for the Docker snippet and runtime check.

## Kubernetes

The pieces above translate directly:

- Use a Deployment with a readiness probe hitting `/health` (or a model-aware check; see the [production checklist](/mistral.rs/guides/deploy/production-checklist/)).
- Mount a PersistentVolumeClaim at `/data` for the Hugging Face cache.
- Use the NVIDIA device plugin and a `nvidia.com/gpu` resource request for CUDA.
- Use an initContainer to pre-download weights for fast pod startup.

There is no official Helm chart. Contributions welcome.

## See also

- [Production checklist](/mistral.rs/guides/deploy/production-checklist/): operational concerns regardless of container layer.
- [Serve flag reference](/mistral.rs/reference/cli/serve/): all `mistralrs serve` options.
