---
title: UQFF format
description: Layout of the UQFF quantized model file format.
---

UQFF is the native mistral.rs quantized file format. To use UQFF models, see the [UQFF guide](/mistral.rs/guides/quantization/uqff/); knowledge of the layout is not required.

:::caution
UQFF 1.x is not compatible with files produced by earlier mistral.rs releases (pre-1.0). Old files will fail with an error; regenerate them with `mistralrs quantize`.
:::

## File structure

A UQFF export is a directory containing:

- One or more `<stem>-<shard>.uqff` shards holding the quantized layers.
- `residual.safetensors` for unquantized tensors (token embeddings, norms, etc.).
- Model assets copied from the source repo so the directory is self-contained: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and (when present) `modules.json`, `chat_template.jinja`, `processor_config.json`, `preprocessor_config.json`.

A loader is pointed at one or more shard files (`from_uqff`); the residual safetensors and the JSON assets are picked up by sibling-path lookup.

## Shard layout

Each `.uqff` shard is a standard safetensors file with named entries. Every quantized layer is self-describing:

- `<key>.weight` - the layer data (raw blocks for GGML-family types, packed tensors for AFQ/MXFP4/FP8, or a native safetensors tensor for unquantized fallback layers; see [quantization types](/mistral.rs/reference/quantization-types/)).
- `<key>.weight.format` - a u8 tag naming the quantization family, used to dispatch the deserializer.
- Family-specific metadata next to it, e.g. `<key>.weight.dtype` and `<key>.weight.shape` for GGML types, `<key>.weight.scales`/`.bits`/`.group_size` for AFQ.
- `<key>.bias` when the layer has one.

`<key>` is the layer's weight path (`model.layers.0.self_attn.q_proj`). MoE (Mixture of Experts) expert layers use three canonical keys per block: `<...>.experts.gate_proj`, `.up_proj`, `.down_proj`, each holding the stacked `[num_experts, out, in]` weights.

Because every layer self-describes, a single file may mix quantization types. Two cases produce a mixed file:

- Topology-pinned layers (assigned a specific type by a [topology](/mistral.rs/guides/perf/topology/) config) keep their pinned type.
- Layers whose shape cannot support the requested type fall back per-layer. For example, AFQ layers whose input dimension is not divisible by the AFQ group size are stored unquantized.

## Sharding

The writer splits the tensor stream into `<stem>-0.uqff`, `<stem>-1.uqff`, ... with a soft cap of 10 GiB per shard. Multiple [ISQ (in-situ quantization)](/mistral.rs/reference/quantization-types/) types in one run produce one shard set per type (`q4k-0.uqff`, `afq4-0.uqff`, ...) sharing the residual and assets.

## Version compatibility

Each shard set carries three u32 scalar entries: `uqff.version.major`, `uqff.version.minor`, `uqff.version.patch`. Readers reject a different major version and reject a minor newer than they support; older minors within the same major are accepted. Files without version entries are rejected.

UQFF 1.1 adds inline unquantized linear entries (`weight.format = Unquant`) so mixed files can preserve unsupported layer shapes without moving those weights into `residual.safetensors`.

## Tensor parallelism

Shards store full tensors; under tensor parallelism each rank slices its portion at load time. Slicing the packed (input) dim requires block alignment, which holds for typical model dims. When alignment does not hold (some expert layers), the rank replicates the full tensor instead of slicing.

## Reference implementation

Canonical implementations: `mistralrs-quant/src/uqff` (reader, tensor encoding) and `mistralrs-core/src/pipeline/isq.rs` (writer).

## Caveats

- UQFF is inference-only; no optimizer state or training metadata.
- The export directory is the unit of distribution. A shard alone is not loadable -- the residual safetensors and `config.json` are required.
