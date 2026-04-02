# mistral.rs Configuration Wizard

Configuration assistant for mistral.rs. Generates `config.toml`, `.env`, `quickstart.sh`, and Kubernetes manifests for every deployment scenario, with hardware-matched TurboQuant KV-cache compression recommendations.

## Quick Start

| Command | What it does |
|---------|-------------|
| `/mistralrs-config` | Auto-detect intent and route to the right mode |
| `/mistralrs-wizard` | Guided first-time setup Q&A |
| `/mistralrs-validate` | Check existing `config.toml` / `.env` for errors |
| `/mistralrs-migrate` | Convert CLI shell scripts to TOML config |
| `/mistralrs-model-select` | Hardware profile → model recommendation → TurboQuant sizing |
| `/mistralrs-stack` | Full bundle: `config.toml` + `.env` + `quickstart.sh` |
| `/mistralrs-k8s` | Generate Kubernetes Secret + ConfigMap + Deployment YAML |

## TurboQuant KV-Cache Compression (`/mistralrs-model-select`)

The model selection advisor includes a hardware-aware compression sizing step:

1. **Hardware profile** — GPU VRAM, RAM, platform (CUDA / Metal / CPU)
2. **Requirements** — context window target, vision, tool-calling, reasoning
3. **Live model research** — searches HuggingFace and model catalogs
4. **VRAM-fit scoring** — ranks models by suitability for your hardware
5. **TurboQuant settings** — recommends `kv_compression_bits` (2/3/4) and threshold

Produces: recommended model + ready-to-paste TOML snippet + quickstart command.

### Compression Quick Reference

| VRAM headroom | `kv_compression_bits` | `kv_compression_threshold` |
|--------------|----------------------|---------------------------|
| > 30 % free | (omit — disabled) | — |
| 15–30 % free | `4` | `4096` |
| 5–15 % free | `3` | `4096` |
| < 5 % free | `3` | `128` |

Three ways to configure compression:

```bash
# 1. CLI flag
mistralrs serve -m meta-llama/Llama-3.1-8B-Instruct --kv-cache-bits 3 --kv-cache-threshold 4096

# 2. Environment variable
export MISTRALRS_KV_CACHE_BITS=3
export MISTRALRS_KV_CACHE_THRESHOLD=4096
mistralrs serve -m meta-llama/Llama-3.1-8B-Instruct

# 3. TOML config (from-config command)
# config.toml:
# [[models]]
# [models.cache]
# kv_compression_bits = 3
# kv_compression_threshold = 4096
```

## Full Stack Bundle (`/mistralrs-stack`)

One command to configure the entire mistral.rs deployment:

```
Output:
  config.toml       mistral.rs TOML configuration
  .env              All environment variables
  quickstart.sh     Launch commands
  k8s/              Kubernetes manifests (optional)
```

## Supported Configuration Sections

`[server]` · `[runtime]` · `[paged_attn]` · `[global]` · `[[models]]` · `[models.format]` · `[models.adapter]` · `[models.quantization]` · `[models.device]` · `[models.vision]` · `[models.cache]`

## State Management

Sessions are tracked in `.mistralrs-config-wizard/sessions/{name}/`:
- `state.json` — current session
- `checkpoints/` — mid-session snapshots
- `output/` — generated files
- `history/` — previous iterations
