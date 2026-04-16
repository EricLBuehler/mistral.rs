---
title: Run mistralrs in Docker
description: Build and run the published container images, with and without CUDA.
sidebar:
  order: 1
---

The mistral.rs repository ships a few Dockerfiles covering the common deployment targets. The one you want depends on what kind of hardware the container will run on.

- `Dockerfile` is the default. A multi-stage build that produces a Debian-based image with the server binary, CPU-only.
- `Dockerfile.cuda-all` is the CUDA variant, targeting NVIDIA GPUs with flash attention enabled.
- `Dockerfile.cuda-13.0-ubi9` is pinned to CUDA 13.0 on Red Hat UBI 9, useful for air-gapped and enterprise deployments.
- `Dockerfile.manylinux` is for producing the Python wheels we publish to PyPI; you probably do not need this one.

## Building an image

From a repository checkout:

```bash
# CPU
docker build -t mistralrs:latest -f Dockerfile .

# CUDA
docker build -t mistralrs:cuda -f Dockerfile.cuda-all .
```

The CUDA build is noticeably slower the first time because flash-attention takes a while to compile. Subsequent builds use the Docker layer cache and are much faster as long as the source tree has not changed extensively.

## Running the CPU image

```bash
docker run --rm -it \
  -p 1234:80 \
  -v $HOME/.cache/huggingface:/data \
  mistralrs:latest \
  mistralrs-server -m Qwen/Qwen3-4B
```

A few things to note about the command:

The image binds port 80 internally by default, controlled by the `PORT` environment variable. `-p 1234:80` publishes it as 1234 on the host, matching the default port you would use without Docker.

The Hugging Face cache directory inside the container is `/data`. Mounting your host cache there makes the first run fast, because weights you have already downloaded do not have to be fetched again.

## Running the CUDA image

```bash
docker run --rm -it --gpus all \
  -p 1234:80 \
  -v $HOME/.cache/huggingface:/data \
  mistralrs:cuda \
  mistralrs-server -m Qwen/Qwen3-4B
```

The `--gpus all` flag exposes all detected NVIDIA GPUs to the container. To pin a specific GPU, use `--gpus '"device=0"'`. Running without the flag falls back to CPU inference, which is almost certainly not what you want from the CUDA image.

The host needs the NVIDIA Container Toolkit installed. [NVIDIA's documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) has platform-specific install steps.

## Production deployment notes

A few things worth doing when the container is not just running on your desktop:

**Persist the cache.** The weights Hugging Face hosts are large enough that pulling them every time a container restarts is a waste. A persistent volume mounted at `/data` keeps them around between runs.

**Pin model versions.** When you pass `-m Qwen/Qwen3-4B`, you get whatever revision is tagged `main` on Hugging Face at download time. For reproducible deployments, append a specific revision with the `--hf-revision` flag and pin it.

**Health check.** `/health` returns 200 when the server is up. Put it in your container healthcheck so the orchestrator knows whether the process is alive:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:80/health || exit 1
```

The generous `--start-period` matters: model loading can take a few minutes on first run, and you do not want the orchestrator to kill the container before it finishes.

**Resource limits.** Without `--memory` and `--gpus`, a runaway process can consume everything on the host. Set them explicitly, especially when you are running multi-model with unloading.

## Kubernetes

If you are deploying to Kubernetes, the pieces above translate directly:

- Use a Deployment with a readiness probe hitting `/health`.
- Mount a PersistentVolumeClaim at `/data` for the Hugging Face cache.
- Use the NVIDIA device plugin and a `nvidia.com/gpu` resource request for CUDA.
- Use an initContainer to pre-download weights if you want pods to start fast.

There is no official Helm chart yet. If you write one, a PR would be welcome.

## What to read next

- [Production checklist](/mistral.rs/guides/deploy/production-checklist/) for operational concerns that apply regardless of how you containerize.
- [HTTP server guide](/mistral.rs/guides/serve/http-server/) for the config options you will want to set.
