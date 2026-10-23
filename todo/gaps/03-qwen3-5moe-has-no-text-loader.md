# `qwen3_5moe` has no text-only loader

- **Repo:** mistral.rs
- **Status:** open, unrequested. Asymmetry, not a defect.
- **Impact:** text-only Qwen3.5-MoE safetensors checkpoints

## What

Qwen3.5-MoE exists only as a **multimodal** loader:

- `MultimodalLoaderType::Qwen3_5Moe`, `#[serde(rename = "qwen3_5moe")]`
  (`mistralrs-core/src/pipeline/loaders/multimodal_loaders.rs:308-309`), resolving
  `Qwen3_5MoeForConditionalGeneration` at `:350`.
- Nothing in `NormalLoaderType`. `grep -c Qwen3_5Moe normal_loaders.rs` is 0.
- `Qwen3_5MoeForCausalLM`, which the sibling repo registers, resolves to nothing here.

The dense sibling has both: `NormalLoaderType::Qwen3_5` for `Qwen3_5ForCausalLM` and
`MultimodalLoaderType::Qwen3_5` for `Qwen3_5ForConditionalGeneration`. So the asymmetry is
specific to the MoE variant.

## Impact

Text-only Qwen3.5-MoE checkpoints, which report `Qwen3_5MoeForCausalLM`, are rejected. A
checkpoint that carries a `vision_config` but is being run as a text model works fine, because
it goes through the multimodal loader regardless.

## Why it has not bitten so far

The GGUF path sidesteps it entirely. In `mistralrs-core/src/gguf/normal_registry.rs:815-825`,
`CanonicalGgufArchitecture::Qwen35Moe` resolves to the `NormalLoaderType::Qwen3Next` adapter:

```rust
NativeModelAdapter {
    loader: NormalLoaderType::Qwen3Next,
    architectures: &[
        CanonicalGgufArchitecture::Qwen3Next,
        CanonicalGgufArchitecture::Qwen35Moe,
    ],
```

The local 35B-A3B checkpoint is GGUF, so it never needs a safetensors `qwen3_5moe` loader.

## Suggested direction

Either add a `NormalLoaderType::Qwen3_5Moe` text loader, or make the multimodal loader accept
`Qwen3_5MoeForCausalLM` as an alias so a text-only checkpoint at least resolves and simply
ignores the absent vision tower.

The first is more work but matches how the dense variant is organised, and the second is a
small change that removes a confusing "unknown architecture" failure. Which one is right
depends on whether text-only Qwen3.5-MoE checkpoints are common enough to care about, which is
a maintainer call rather than a technical one.

## Note

Found while intersecting mistral.rs support against `../vllm.cpp`, which registers both
`Qwen3_5MoeForCausalLM` and `Qwen3_5MoeForConditionalGeneration`. Recorded because the
inventory otherwise looks complete for Qwen3.5 and is not.