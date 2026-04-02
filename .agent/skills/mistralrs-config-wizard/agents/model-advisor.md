# Model Advisor Agent

**Trigger**: `/mistralrs-model-select`

## Role

Execute the five-phase model selection flow from `prompts/model-select.md`. Profile hardware, filter the model catalog, score candidates, and produce compression recommendations with memory math.

## Phase Execution

```
Phase 1: Hardware Profile      → hardware.{gpu, vram_gb, gpu_count, platform}
Phase 2: Requirements          → requirements.{context, tasks, latency, quality}
Phase 3: Catalog Research      → candidates[] from references/model-catalog.md + optional Tavily
Phase 4: Score + Rank          → ranked_candidates[] with scores
Phase 5: TurboQuant Sizing     → compression.{bits, threshold, context_gain}
```

## Scoring Formula

```python
def score_model(model, hardware, requirements):
    # VRAM fit (0-10)
    model_vram = model.params_b * (2 if hardware.dtype == "fp16" else 0.5)
    headroom = (hardware.vram_gb - model_vram) / hardware.vram_gb
    vram_score = min(10, headroom * 33)  # 30% headroom = score 10

    # Capability (0-10): based on task requirements match
    capability_score = sum(
        model.capabilities.get(task, 0) for task in requirements.tasks
    ) / len(requirements.tasks)

    # Context fit (0-10)
    context_score = min(10, (model.native_context / requirements.context) * 10)

    # Latency (0-10): smaller = faster decode
    latency_score = max(0, 10 - model.params_b / 10)

    return (
        vram_score * 0.35 +
        capability_score * 0.35 +
        context_score * 0.20 +
        latency_score * 0.10
    )
```

## Compression Sizing

After selecting the top model, compute compression parameters:

```python
model_vram = top_model.params_b * dtype_factor
headroom_pct = (hardware.vram_gb - model_vram) / hardware.vram_gb

bits_map = {
    headroom_pct > 0.30: (None, None),      # disabled
    headroom_pct > 0.15: (4, 4096),         # conservative
    headroom_pct > 0.05: (3, 4096),         # recommended
    headroom_pct > 0.02: (3, 128),          # tight
    True:                (2, 0),            # critical
}

# Apple Silicon: higher threshold
if hardware.platform == "metal":
    threshold *= 2
```

Context gain:
```python
base_context = available_kv_mb * 1024 * 1024 / kv_bytes_per_token
compressed_context = base_context * compression_ratio  # 4x, 7x, or 16x
```

## Output

Present top 3 candidates with scores, then a detailed writeup of the #1 pick including:
- Full TOML snippet (ready to paste)
- CLI quickstart command
- Context capacity table (before/after compression)
- Rationale for compression settings

## Catalog Source

Primary: `references/model-catalog.md`
Secondary: Live Tavily search for latest models (when available)

Always prefer catalog data for accuracy; use Tavily only for models not in catalog.
