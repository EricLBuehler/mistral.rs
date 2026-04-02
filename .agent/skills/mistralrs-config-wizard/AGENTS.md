# Contributor Guide — mistralrs-config-wizard

## Commit Conventions

Use conventional commits:
- `feat:` — New modes, templates, platform support, new TOML sections
- `fix:` — Template fixes, validation corrections, broken references
- `docs:` — Reference updates, SKILL.md changes, CLAUDE.md updates
- `refactor:` — Internal restructuring without behavior change

## Branch Strategy

- `main` — Stable, tested
- `feat/*` — Feature branches

## Pull Request Checklist

- [ ] Templates use `{{variable}}` syntax consistently
- [ ] New files referenced in `SKILL.md` and `prompts/meta-controller.md`
- [ ] TOML templates validate (no syntax errors)
- [ ] Scripts have `#!/usr/bin/env bash` shebang and `set -euo pipefail`
- [ ] Cross-references resolve (no dangling paths)
- [ ] `plugin.json` updated if new slash commands added
- [ ] `references/config-reference.md` updated if new TOML fields added

## Architecture References

- `SKILL.md` — Skill functionality and behavior (source of truth)
- `CLAUDE.md` — Development guidelines
- `references/config-reference.md` — Config option catalogue
- `references/cli-args.md` — CLI flag catalogue with env var equivalents
- `references/turboquant-guide.md` — KV-cache compression sizing guide

## Config Reference Maintenance

The `references/` directory must stay in sync with the mistral.rs source.

**Source of truth files:**
- `mistralrs-cli/src/args/paged_attn.rs` → `references/cli-args.md` (cache options)
- `mistralrs-core/src/toml_selector.rs` → `references/config-reference.md` (TOML fields)
- `docs/CONFIGURATION.md` → `references/env-vars.md`
- `Cargo.toml` features → `references/config-reference.md` (feature gates)

When mistral.rs adds new configuration:
1. Add to `references/config-reference.md` (all fields: key, env var, default, type, description)
2. Add to `references/env-vars.md` if a `MISTRALRS_*` var is added
3. Add to `references/cli-args.md` if a `--flag` is added
4. Update `assets/templates/config.toml.template` with the new section/field
5. Update wizard questions in `prompts/wizard.md` if user-visible

## TurboQuant Guide Maintenance

`references/turboquant-guide.md` documents KV-cache compression decisions.
Source of truth: `docs/prometheus-enhancements/KVCACHE-COMPRESSION.md`

Update when:
- Compression bits or threshold defaults change in `mistralrs-cli/src/args/paged_attn.rs`
- New platform support is added (new `--features` combination)
- Benchmark numbers improve

## Validation Logic

The `prompts/validate.md` prompt performs structural checks on generated configs. Key rules (documented in `CLAUDE.md`):

1. `command` must be `"serve"` or `"run"`
2. Each `[[models]]` must have `kind` and `model_id`
3. `default_model_id` must match a `model_id` in `[[models]]`
4. Exactly one of `context_len`, `memory_mb`, `memory_fraction` in `[paged_attn]`
5. `kv_compression_threshold` requires `kv_compression_bits`
6. `kv_compression_bits` requires `kvcache-compression` feature

When adding new validation rules, add them to `prompts/validate.md` and document them here.
