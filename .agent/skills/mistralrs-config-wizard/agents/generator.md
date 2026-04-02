# Generator Agent

**Trigger**: Internal — called by wizard-guide, model-advisor, or `/mistralrs-stack`

## Role

Render final config artifacts from session state. Fill template variables, assemble multi-file bundles, and hand off to the validator before presenting to the user.

## Template Rendering

Load `assets/templates/config.toml.template` and substitute all `{{variable}}` slots from session state. Rules:

- Omit optional sections entirely when their values are null/default
- Never emit `[models.cache]` if `kv_bits` is null
- Never emit `kv_compression_threshold` without `kv_compression_bits`
- Always add `# Requires --features kvcache-compression` comment on `kv_compression_bits` line
- Use `"auto"` for dtype unless explicitly set
- Round `memory_fraction` to two decimal places

## Conditional Sections

```
[server]         → always include for "serve" command
[global]         → include only if seed or log is set
[paged_attn]     → always include; use mode="auto" if not specified
[models.format]  → include only if format != "plain" or quantized_file set
[models.adapter] → include only if lora or xlora set
[models.device]  → include only if topology, hf_cache, or device_layers set
[models.vision]  → include only if kind = "vision"
[models.cache]   → include only if kv_compression_bits is not null
```

## Bundle Assembly

```
output/
  config.toml          ← always
  .env                 ← always (even if mostly commented placeholders)
  quickstart.sh        ← always
  README.md            ← always (inline documentation)
  docker-compose.yml   ← if scenario = "docker"
  k8s/                 ← if scenario = "kubernetes"
    secret.yaml
    configmap.yaml
    deployment.yaml
    service.yaml
    pvc.yaml
```

## Post-Generation

1. Write all files to `.mistralrs-config-wizard/sessions/{session_name}/output/`
2. Pass `config.toml` to `agents/validator.md`
3. If validator returns errors → apply unambiguous fixes → re-validate (max 2 iterations)
4. Log all fixes in `state.json` under `validation_fixes[]`
5. If still errors after 2 iterations → surface to user with explanation
6. If pass → trigger `hooks/post-generate.md`

## Handoff to User

Present bundle as:
```
✅ Configuration generated successfully.

Files created in .mistralrs-config-wizard/sessions/default/output/:
  config.toml
  .env
  quickstart.sh

To use:
  cp .mistralrs-config-wizard/sessions/default/output/* .
  mistralrs from-config --file config.toml
```

Then show each file with syntax-highlighted code blocks.
