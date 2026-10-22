# No OPT architecture

- **Repo:** mistral.rs
- **Status:** open, unrequested. Feature gap, not a defect.
- **Impact:** cannot load `OPTForCausalLM` checkpoints

## What

`OPTForCausalLM` does not appear anywhere in the loader registry.
`NormalLoaderType` (`mistralrs-core/src/pipeline/loaders/normal_loaders.rs:178-231`) has no
OPT variant, so `from_causal_lm_name` cannot resolve it.

This matters mainly for the small dense reference models people use to test schedulers,
tokenizers and samplers without downloading something large.

## Why it came up

The local model collection includes two, which are otherwise ideal CPU test subjects because
they are tiny and load instantly:

```
/mnt/models/opt-125m-hf       (OPTForCausalLM, safetensors)
/mnt/models/opt-125m-bf16-st  (OPTForCausalLM, safetensors)
```

They are also present in the Hugging Face cache as `facebook/opt-125m`. The sibling repo
`../vllm.cpp` supports OPT, so this is a coverage difference rather than a hard architectural
blocker.

## Scope if implemented

OPT is an older, simpler architecture than anything currently in the registry:

- learned positional embeddings rather than RoPE,
- a `do_layer_norm_before` flag that flips the residual/norm order,
- an optional project-in/project-out, so the model is not purely residual-stream,
- grouped-query attention, controlled by `num_attention_heads` being a multiple of
  `num_key_value_heads`.

That last point is the only part with a precedent worth reusing: the GQA path in
`qwen3_moe.rs`. Everything else is new surface, and OPT-specific norm-order handling has
close cousins already in the registry (`gpt_oss.rs` sandwich norms, the qwen3 post-norm
handling), so there is prior art to follow for the awkward part.

## Note

Recorded as a gap only. It is not on the critical path for the CPU work, which is why it sits
in this directory rather than in `../mistralrs`.