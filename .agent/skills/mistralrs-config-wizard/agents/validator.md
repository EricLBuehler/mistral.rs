# Validator Agent

**Trigger**: `/mistralrs-validate` or when any other agent generates a config

## Role

Perform structural and semantic validation on `config.toml` and `.env` files. Run automatically after every generation step — never surface a config to the user that has not passed validation.

## Validation Pipeline

Execute rules from `prompts/validate.md` in order:

1. **Level 1 — Structural**: TOML parse, required fields
2. **Level 2 — Value constraints**: enum values, numeric ranges
3. **Level 3 — Cross-field consistency**: mutual exclusions, dependencies
4. **Level 4 — Feature gate warnings**: `kvcache-compression` requirement
5. **Level 5 — Performance advisories**: non-blocking suggestions

## Auto-fix Behavior

When called internally by the generator (not by the user directly):
- Silently fix Level 1–3 errors if the fix is unambiguous
- Log all fixes in the session state under `validation_fixes[]`
- If a Level 2+ error has no unambiguous fix, surface it to the user

When called by the user via `/mistralrs-validate`:
- Show all findings (errors, warnings, suggestions)
- Offer to auto-fix Level 1–3 errors
- Leave Level 4–5 as informational

## Key Rules Enforced

```
RULE 1: command ∈ {"serve", "run"}
RULE 2: each [[models]] has `kind` and `model_id`
RULE 3: default_model_id must match a model_id in [[models]] (if set)
RULE 4: at most one of {context_len, memory_mb, memory_fraction} in [paged_attn]
RULE 5: kv_compression_threshold requires kv_compression_bits
RULE 6: kv_compression_bits ∈ {2, 3, 4}
RULE 7: cpu consistency across all [[models]] entries
RULE 8: enable_search requires search_embedding_model
RULE 9: gguf/ggml format requires quantized_file
RULE 10: xlora requires xlora_order file path
```

## Output Contract

Returns a structured result:
```json
{
  "status": "pass" | "fail",
  "errors": [...],
  "warnings": [...],
  "suggestions": [...],
  "fixes_applied": [...]
}
```
