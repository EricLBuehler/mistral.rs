---
name: mistralrs-config-wizard
description: >
  Configuration assistant for mistral.rs. Covers all modes: interactive wizard
  (guided Q&A → tailored config.toml + .env + quickstart.sh), advisor (explain
  any option on demand), validator (check existing config.toml / .env for
  errors and conflicts), model-select (hardware profiling → model recommendation
  → TurboQuant KV-cache compression sizing), migration assistant (upgrade
  legacy CLI-only scripts to TOML file config), Kubernetes/Docker manifest
  generation, and full-stack bundle generator (config.toml + .env +
  quickstart.sh + optional k8s manifests). Generates immediately-runnable
  configs for every supported deployment scenario.
allowed-tools: file_system code_interpreter web_search tavily
---

# mistral.rs Configuration Wizard

A PMPO-driven configuration assistant for mistral.rs. Handles the full configuration lifecycle — from first-run guided setup through Docker/Kubernetes production deployment — with special focus on TurboQuant KV-cache compression sizing.

## Modes

### `/mistralrs-config` — Main entry point (auto-routes)
Analyzes intent and routes to the appropriate mode automatically. Use this if you're unsure which command to run.

### `/mistralrs-wizard` — Interactive guided setup
Asks targeted questions about hardware, deployment scenario, model requirements, and optional features. Produces a tailored `config.toml`, `.env`, and `quickstart.sh` in one pass.

### `/mistralrs-advise` — On-demand option explanation
Explains any mistral.rs configuration option, CLI flag, or environment variable in plain language with examples. Use this to understand what a specific setting does before committing to it.

### `/mistralrs-validate` — Validate existing files
Checks an existing `config.toml` and/or `.env` for syntax errors, missing required fields, type mismatches, conflicting settings, and deprecated usage.

### `/mistralrs-model-select` — AI model selection advisor
Five-phase guided model selection:
1. **Hardware profile** — GPU type/count/VRAM, RAM, platform (CUDA/Metal/CPU)
2. **Requirements** — context window, vision, tool-calling, reasoning, latency vs quality
3. **Live catalog search** — queries HuggingFace and model catalogs via Tavily
4. **Scoring + ranking** — VRAM fit × capability × speed × cost weighted rubric
5. **TurboQuant sizing** — recommends `kv_compression_bits` (2/3/4) + threshold based on VRAM headroom

Produces: recommended model list with rationale + ready-to-paste `config.toml` snippet + quickstart command.

### `/mistralrs-stack` — Full bundle (one command)
Runs wizard + model-select in sequence and emits a complete, immediately-runnable bundle:
- `config.toml` — full mistral.rs configuration
- `.env` — all environment variables including `MISTRALRS_KV_CACHE_BITS`
- `quickstart.sh` — copy-pastable launch commands
- `k8s/` manifests (optional)

### `/mistralrs-k8s` — Kubernetes / Docker manifest generation
Generates Kubernetes `Secret` (sensitive env vars) + `ConfigMap` (non-sensitive settings) + `Deployment` YAML, or a `docker-compose.yml` for local container deployments.

### `/mistralrs-migrate` — Legacy CLI script migration
Converts existing shell scripts that use raw `mistralrs serve ...` CLI flags into a structured `config.toml` + `.env` that is easier to maintain and version-control. Identifies any flags with no TOML equivalent and notes workarounds.

## Configuration Dimensions Covered

| Dimension | Key options |
|-----------|------------|
| Build | `kvcache-compression`, `cuda`, `metal`, `flash-attn`, `server-only` features |
| Runtime | `--max-seqs`, `--no-kv-cache`, `--prefix-cache-n`, `--chat-template` |
| Model source | HuggingFace ID, local path, GGUF/GGML, UQFF |
| Quantization | ISQ levels (q4k, q8, etc.), from-uqff, imatrix |
| KV compression | `--kv-cache-bits` (2/3/4), `--kv-cache-threshold`, env vars |
| Paged attention | mode (auto/on/off), context_len, memory_fraction, block_size |
| Device mapping | CPU/GPU layer splits, topology YAML |
| Adapters | LoRA, X-LoRA |
| Vision | max_edge, max_num_images |
| Server | port, host, UI, MCP |
| Deployment | local-dev, docker-compose, Kubernetes, Apple Silicon |

## Execution Model (PMPO Loop)

### Startup
1. **Resolve mode** — detect slash command or analyze free-form intent
2. **Init state** — load or create named config session in `.mistralrs-config-wizard/`
3. **Route** — dispatch to wizard, advise, validate, model-select, or generate phase

### Phase Loop
1. **Clarify** — gather requirements via targeted Q&A
2. **Generate** — produce config artifacts
3. **Validate** — check output for correctness and conflicts
4. **Persist** — write validated state and output files

### Phase Hooks
After each phase: checkpoint + dispatch to next phase or surface output to user.

## State Management

State is persisted to `.mistralrs-config-wizard/` in the project root:

```
.mistralrs-config-wizard/
  registry.json              # Maps session_name → state path
  sessions/
    {session_name}/
      state.json             # Current session state
      checkpoints/           # Mid-session snapshots
      output/                # Generated config files
      history/               # Previous iterations
```

## Deployment Scenarios

| Scenario | Output |
|----------|--------|
| `local-dev` | Minimal `config.toml` + `.env`, no auth, single GPU |
| `docker-compose` | `docker-compose.yml` + `.env` with all vars |
| `kubernetes` | K8s `Secret` + `ConfigMap` + `Deployment` YAML |
| `apple-silicon` | Metal-optimized config with tuned threshold values |
| `multi-gpu` | Topology YAML + device layer mapping |
| `cpu-only` | CPU-safe config with conservative compression settings |

## Quick Start

- `/mistralrs-config` — Auto-detect what you need and get started
- `/mistralrs-wizard` — Guided first-time setup
- `/mistralrs-validate` — Check an existing config for errors
- `/mistralrs-model-select` — Get a hardware-matched model recommendation
- `/mistralrs-stack` — Generate a complete deployment bundle
- `/mistralrs-k8s` — Generate Kubernetes or Docker manifests
- `/mistralrs-migrate` — Convert CLI scripts to TOML config
