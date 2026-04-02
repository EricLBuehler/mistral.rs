# Post-Generate Hook

Triggered by `agents/generator.md` after each successful generation + validation cycle.

## Actions

1. **Checkpoint state** — write current session to `.mistralrs-config-wizard/sessions/{session_name}/checkpoints/{timestamp}.json`
2. **Copy outputs** — write all artifacts to `.mistralrs-config-wizard/sessions/{session_name}/output/`
3. **Update registry** — update `.mistralrs-config-wizard/registry.json` with session metadata
4. **Surface summary** — print the bundle summary to the user (file list + run command)

## Registry Entry Format

```json
{
  "session_name": "default",
  "last_updated": "2026-04-02T13:00:00Z",
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "kv_bits": 3,
  "kv_threshold": 4096,
  "scenario": "local-dev",
  "output_path": ".mistralrs-config-wizard/sessions/default/output/",
  "files": ["config.toml", ".env", "quickstart.sh"]
}
```

## User Summary Format

```
✅ Configuration generated for meta-llama/Llama-3.1-8B-Instruct

  config.toml    — server config with 3-bit KV compression (threshold: 4096 tokens)
  .env           — environment variables (MISTRALRS_KV_CACHE_BITS=3)
  quickstart.sh  — launch script

To deploy:
  cp .mistralrs-config-wizard/sessions/default/output/* .
  chmod +x quickstart.sh
  ./quickstart.sh

Or run directly:
  mistralrs from-config --file config.toml

To re-validate: /mistralrs-validate
To adjust model: /mistralrs-model-select
```

## Error Handling

If post-generate fails (e.g., filesystem write error), do NOT block the user — surface the files inline in the chat response so the user can copy them manually. Log the error to session state.
