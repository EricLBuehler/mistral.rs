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

`:latest` is the CPU image. For NVIDIA GPUs, choose a CUDA tag and add `--gpus all`:

```bash
docker run --rm --gpus all -p 1234:1234 -v hf-cache:/data \
  ghcr.io/ericlbuehler/mistral.rs:cuda128-sm89-latest \
  serve -m Qwen/Qwen3-4B
```

The host needs the NVIDIA Container Toolkit; see [NVIDIA's install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). To pin a specific GPU: `--gpus '"device=0"'`.

## Published tags

All images live at `ghcr.io/ericlbuehler/mistral.rs` ([package page](https://github.com/EricLBuehler/mistral.rs/pkgs/container/mistral.rs)). Each one has a `-latest` tag that follows the newest release and an `X.Y.Z` tag, such as `cpu-0.9.5`, that stays on one release.

| Image | Tags |
|---|---|
| CPU | `latest` (same as `cpu-latest`), `cpu-X.Y.Z` |
| CUDA | `cuda<ver>-sm<cc>-latest`, `cuda<ver>-sm<cc>-X.Y.Z` |

For CUDA, `<ver>` is the CUDA version the image was built with, such as `cuda132` for CUDA 13.2. Pick the newest one your driver supports, using the CUDA version `nvidia-smi` reports. Only `cuda132` and `cuda133` include cuTile support:

| `nvidia-smi` reports | Tag |
|---|---|
| CUDA 13.3+ on Hopper (`sm90`) | `cuda133-sm90-latest` |
| CUDA 13.2+ on Ampere, Ada, or Blackwell | `cuda132-sm<cc>-latest` |
| CUDA 13.1+ | `cuda131-sm<cc>-latest` |
| CUDA 13.0+ | `cuda130-sm<cc>-latest` |
| CUDA 12.9+ on DGX Spark (`sm121`) | `cuda129-sm121-latest` |
| CUDA 12.8+ | `cuda128-sm<cc>-latest` (not `sm121`) |

Then fill in `<cc>`, your GPU's compute capability:

| `<cc>` | GPUs |
|---|---|
| `80` | A100 |
| `86` | RTX 30 series, A-series workstation cards |
| `89` | RTX 40 series, L4 |
| `90` | H100 |
| `100` | B200 |
| `120` | RTX 50 series |
| `121` | DGX Spark |

The full mapping is in [hardware support](/reference/hardware-support/#cuda-artifacts). Older tags without a CUDA version, `cuda-sm<cc>-latest` and `cuda-sm<cc>-X.Y.Z`, point at the `cuda131` images.

The CPU image and the `sm90` and `sm100` CUDA images run on both x86_64 and arm64 (NVIDIA Grace), and `sm121` is arm64 only. The other CUDA images are x86_64 only. Docker pulls the right architecture automatically.

For production, use an `X.Y.Z` tag instead of `-latest`. Model IDs also move: `-m Qwen/Qwen3-4B` downloads whatever the `main` branch holds at that moment. The CLI has no flag to pick a model revision, so to pin one, download that revision and pass its local path to `-m`, or use the Rust SDK's `with_hf_revision`.

## Ports and model cache

- Entrypoint is the `mistralrs` binary; pass a subcommand and its flags as the container command.
- `mistralrs serve` listens on port 1234 by default (the image's `EXPOSE`d port). To change it, change the flag and the mapping together: `serve -p 8080` with `-p 8080:8080`. There is no `PORT` environment variable.
- `HF_HOME=/data` is set in the image: mount a volume at `/data` to persist downloaded weights (they land in `/data/hub`). HF authentication for gated models: `-e HF_TOKEN=<token>`.
- Chat templates ship at `/chat_templates` for models that need one: `--chat-template /chat_templates/<file>.json`.

## Enabling cuTile in a published image

cuTile is an optional NVIDIA backend that speeds up MoE and quantized models, and NVFP4 models require it. The `cuda132` and `cuda133` images can use it, but they don't include NVIDIA's `tileiras` tool, which you add in your own image. For Ampere, Ada, or Blackwell:

```dockerfile
FROM ghcr.io/ericlbuehler/mistral.rs:cuda132-sm100-latest

RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && python3 -m pip install --no-cache-dir --upgrade "cuda-tile[tileiras]" \
    && TILEIRAS_DIR="$(python3 -c 'import nvidia.cu13.bin as b; print(next(iter(b.__path__)))')" \
    && ln -sf "$TILEIRAS_DIR/tileiras" /usr/local/bin/tileiras \
    && rm -rf /var/lib/apt/lists/*

ENV CUTILE_TILEIRAS_PATH=/usr/local/bin/tileiras
```

Change the base tag to match your GPU. For Hopper, start from `cuda133-sm90-latest` and install `cuda-toolkit[tileiras,nvvm,nvcc]>=13.3` instead. Run `mistralrs doctor` in the container to check that cuTile is available.

## Building an image

From a repository checkout:

```bash
# CPU
docker build -t mistralrs:latest -f Dockerfile .

# CUDA source build (set the compute capability for your GPU)
docker build -t mistralrs:cuda -f Dockerfile.cuda-13.0-ubi9 \
  --build-arg CUDA_COMPUTE_CAP=89 \
  --build-arg WITH_FEATURES=cuda,cudnn,flash-attn .
```

`Dockerfile.cuda-13.0-ubi9` compiles mistral.rs on Red Hat UBI 9. It takes two build arguments:

| Argument | Default | What it sets |
|---|---|---|
| `CUDA_COMPUTE_CAP` | `80` | Your GPU's compute capability, from the table above. |
| `WITH_FEATURES` | `cuda,cudnn` | [Cargo features](/reference/cargo-features/) to build with. Add `flash-attn` for faster prompt processing on compute capability 8.0 and newer. |

The first build with `flash-attn` is slow; later builds reuse the layer cache.

`Dockerfile.cuda-all` is what the published CUDA images are built from. It doesn't compile anything: it copies a prebuilt release binary into an NVIDIA CUDA runtime image, so use `Dockerfile.cuda-13.0-ubi9` to build from source.

## Production deployment notes

**Persist the cache.** Weights are large enough that re-downloading on every restart is wasteful. Mount a named volume or host path at `/data`.

**Health check.** `/health` returns 200 when the server is up. Add a Docker healthcheck:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:1234/health || exit 1
```

The generous `--start-period` matters: first-run model loading can take minutes.

**Resource limits.** Set `--memory` and `--gpus` on `docker run` to bound the container's resources.

**Video input.** Install FFmpeg inside the image when serving video-capable models. See [set up video input](/guides/models/video-setup/) for the Docker snippet and runtime check.

## Kubernetes

The pieces above translate directly:

- Use `/health` for liveness and check the required model's status in `/v1/models` for readiness. See the [production checklist](/guides/deploy/production-checklist/).
- Mount a PersistentVolumeClaim at `/data` for the Hugging Face cache.
- Use the NVIDIA device plugin and a `nvidia.com/gpu` resource request for CUDA.
- Use an initContainer to pre-download weights for fast pod startup.

There is no official Helm chart. Contributions welcome.

## See also

- [Production checklist](/guides/deploy/production-checklist/): operational concerns regardless of container layer.
- [Serve flag reference](/reference/cli/serve/): all `mistralrs serve` options.
