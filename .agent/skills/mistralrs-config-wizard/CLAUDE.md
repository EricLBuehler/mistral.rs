# CLAUDE.md — mistralrs-config-wizard Development Guidelines

## Purpose

This skill is a PMPO-driven configuration assistant for mistral.rs. It generates immediately-runnable `config.toml`, `.env`, `quickstart.sh`, and Kubernetes manifests. The guiding principle is **never generate a config that will fail on first run**.

## Core Files

| File | Role |
|------|------|
| `SKILL.md` | Source of truth for modes, slash commands, and behavior |
| `prompts/meta-controller.md` | Routes slash commands to sub-prompts |
| `prompts/wizard.md` | Interactive Q&A → config generation |
| `prompts/validate.md` | Config correctness checks |
| `prompts/model-select.md` | Hardware → model → compression advisor |
| `references/config-reference.md` | Complete option catalogue |
| `references/cli-args.md` | All CLI flags with env var equivalents |
| `references/turboquant-guide.md` | KV-cache compression sizing guide |
| `assets/templates/` | Annotated config templates |

## Design Principles

1. **Correctness first** — never emit a config with known errors. Run the validate phase on all generated output.
2. **Three-mechanism coverage** — every configurable value should show the CLI flag, env var, and TOML equivalent.
3. **TurboQuant by default for constrained hardware** — if hardware profile shows < 30% VRAM headroom after model load, always recommend compression.
4. **Source of truth** — `references/config-reference.md` must stay in sync with `mistralrs-core/src/toml_selector.rs` and `mistralrs-cli/src/args/`.

## Config Generation Rules

- All generated TOML must validate with `toml::from_str::<serde_json::Value>(&content).is_ok()`
- Do not emit fields that require a Cargo feature unless the feature is confirmed enabled
- Always include a `# Requires kvcache-compression feature` comment on `kv_compression_bits`
- `kv_compression_threshold` must only appear when `kv_compression_bits` is present
- `default_model_id` in serve mode must exactly match one `model_id` in `[[models]]`
- `cpu = true` must be consistent across all `[[models]]` entries

## Slash Command Registry

All slash commands are defined in `.claude-plugin/plugin.json`. To add a new mode:
1. Add the slash command to `plugin.json`
2. Add the mode description to `SKILL.md`
3. Create or update the corresponding prompt in `prompts/`
4. Add an agent in `agents/` if the mode needs multi-step reasoning

## Template Variables

Templates in `assets/templates/` use `{{variable}}` syntax. Valid variables:

| Variable | Description |
|----------|-------------|
| `{{model_id}}` | HuggingFace model ID or local path |
| `{{port}}` | HTTP server port |
| `{{kv_bits}}` | Compression bits (2, 3, 4, or empty) |
| `{{kv_threshold}}` | Compression threshold in tokens |
| `{{isq_level}}` | ISQ quantization level (q4k, q8, etc.) |
| `{{max_seqs}}` | Maximum concurrent sequences |
| `{{memory_fraction}}` | PagedAttention GPU memory fraction |
| `{{dtype}}` | Model data type (auto, f16, bf16) |

## Sync Checklist

When mistral.rs adds new configuration:
1. Add to `references/config-reference.md` (all fields: key, env var, default, type)
2. Add to `references/env-vars.md` if a `MISTRALRS_*` var is added
3. Add to `references/cli-args.md` if a `--flag` is added
4. Update `assets/templates/config.toml.template`
5. Update wizard questions in `prompts/wizard.md` if user-visible
6. Update `SKILL.md` if a new section is introduced

## Testing Generated Configs

Before surfacing any config to the user, mentally run through:
1. `mistralrs from-config --file config.toml` — does it parse?
2. Are all required fields present (`command`, `kind`, `model_id`)?
3. Are mutually exclusive options absent (e.g., only one of `context_len` / `memory_mb` / `memory_fraction`)?
4. If `kv_compression_bits` is set, is `kvcache-compression` feature available in the build?
5. Does `default_model_id` match a `model_id` in `[[models]]`?
