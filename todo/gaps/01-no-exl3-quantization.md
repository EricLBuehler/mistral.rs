# No EXL3 quantization support

- **Repo:** mistral.rs
- **Status:** open, unrequested. Feature gap, not a defect.
- **Impact:** cannot load EXL3 checkpoints

## What

`exl3` does not appear anywhere in `mistralrs-quant`. The `IsqType` enum
(`mistralrs-quant/src/lib.rs:958-982`) covers Q4_0 through Q6K, HQQ, F8E4M3, AFQ, F8Q8 and
MXFP4; EXL3 is absent, and there is no `QuantizedConfig` variant for it either, so a
checkpoint carrying one is rejected at config parse rather than at load.

## Why it came up

A local model collection contains a checkpoint that is unusable for this reason:

```
/mnt/models/Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw
  architectures:    ["Qwen3_5ForConditionalGeneration"]
  model_type:       qwen3_5
  quantization_config:
    quant_method: exl3, version 1.4.2, bpw 3.5, head_bits 6,
    codebook mul1, mtp_bits 4
```

The architecture is fully supported. `Qwen3_5ForConditionalGeneration` maps to
`MultimodalLoaderType::Qwen3_5` at `mistralrs-core/src/pipeline/loaders/multimodal_loaders.rs:349`.
Only the quantization method is missing, which is the narrower of the two gaps.

## Scope if implemented

EXL3 is not a bit format; it is a codebook-based quantization with a `.dat`/`.tpl` sidecar
structure that ggml loads through its own reader. Realistically this is:

- a config variant and a `QuantMethod` impl,
- a loader for the EXL3 sidecar files, which are not safetensors,
- a CPU matmul kernel, and the question of whether to route it through the existing packed-GEMV
  machinery or write a new one.

That is a larger project than a missing enum arm, which is why this is filed as a gap and not
as a bug. The sibling repo `../vllm.cpp` supports EXL3 and has a comparator entry for it, if
the API surface is wanted as a reference.

## Alternative

For the specific model above, the same architecture is available in GGUF locally
(`/mnt/models/unsloth-qwen3.8-27B-gguf`, GGUF qwen35), which *is* supported. So the practical
gap is narrower than "we cannot run Qwen3.8": it is "we cannot run this particular
quantization of it".

## Note

This is recorded because it looked like an architecture gap at first glance and is not. Worth
separating in any future inventory of what is missing.