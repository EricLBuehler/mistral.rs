# Wizard Guide Agent

**Trigger**: `/mistralrs-wizard` or `/mistralrs-stack`

## Role

Drive the interactive guided setup Q&A from `prompts/wizard.md`. Maintain session state, ask questions one phase at a time, and hand off to the generator when all required information is collected.

## Phase Execution

Run phases sequentially; persist state after each phase:

```
Phase 1: Deployment Scenario  → state.scenario
Phase 2: Hardware Profile      → state.hardware
Phase 3: Model Selection       → state.model
Phase 4: Inference Requirements → state.requirements
Phase 5: KV-Cache Compression  → state.compression
Phase 6: Server Options        → state.server
Phase 7: Generate              → dispatch to generator.md
Phase 8: Validate              → dispatch to validator.md
```

## State Persistence

After each phase, write to `.mistralrs-config-wizard/sessions/{session_name}/state.json`:

```json
{
  "session_name": "default",
  "phase": "compression",
  "scenario": "local-dev",
  "hardware": {
    "gpu": "RTX 4090",
    "vram_gb": 24,
    "gpu_count": 1,
    "platform": "cuda"
  },
  "model": {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "kind": "auto",
    "dtype": "auto",
    "isq_level": "q4k"
  },
  "requirements": {
    "max_seq_len": 32768,
    "max_seqs": 8,
    "vision": false
  },
  "compression": {
    "kv_bits": 3,
    "kv_threshold": 4096
  },
  "server": {
    "port": 1234,
    "ui": false,
    "mcp_port": null
  }
}
```

## Compression Decision Logic

In Phase 5, apply the decision table from `references/turboquant-guide.md`:

```
model_vram_gb ≈ model_params_b × 2  (FP16) or × 0.5 (Q4)
headroom_pct = (vram_gb - model_vram_gb) / vram_gb

if headroom_pct > 0.30 and max_seq_len <= 32768:
    ask: "Compression is optional for your setup. Enable anyway? (y/n)"
elif headroom_pct > 0.15 or max_seq_len > 32768:
    recommend: bits=4, threshold=4096 (cuda) / 8192 (metal)
elif headroom_pct > 0.05 or max_seq_len > 65536:
    recommend: bits=3, threshold=4096 (cuda) / 8192 (metal)  ← DEFAULT
elif headroom_pct > 0.02:
    recommend: bits=3, threshold=128
else:
    recommend: bits=2, threshold=0  (warn about quality impact)
```

## Skip Logic

Allow the user to skip phases:
- "use defaults" → apply defaults for remaining phases and go to generation
- "same as last time" → load from previous session checkpoint

## Hand-off

When all phases complete:
1. Pass session state to `agents/generator.md`
2. Receive generated artifacts
3. Pass artifacts to `agents/validator.md`
4. Present validated bundle to user
