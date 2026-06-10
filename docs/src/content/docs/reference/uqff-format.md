---
title: UQFF format
description: Layout of the UQFF quantized model file format.
sidebar:
  order: 12
---

UQFF is the native mistral.rs quantized file format. To use UQFF models, see the [UQFF guide](/mistral.rs/guides/perf/use-uqff/); knowledge of the layout is not required.

:::caution
UQFF 1.0 is not compatible with files produced by earlier mistral.rs releases (pre-1.0). Old files fail with a clear error; regenerate them with `mistralrs quantize`.
:::

## File structure

A UQFF export is a directory containing:

- One or more `<stem>-<shard>.uqff` shards holding the quantized layers.
- `residual.safetensors` for unquantized tensors (token embeddings, norms, etc.).
- Model assets copied from the source repo so the directory is self-contained: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and (when present) `modules.json`, `chat_template.jinja`, `processor_config.json`, `preprocessor_config.json`.

A loader is pointed at one or more shard files (`from_uqff`); the residual safetensors and the JSON assets are picked up by sibling-path lookup.

## Shard layout

Each `.uqff` shard is a standard safetensors file with named entries. Every quantized layer is self-describing:

- `<key>.weight` - the quantized data (raw blocks for GGML-family types, packed tensors for AFQ/MXFP4/FP8).
- `<key>.weight.format` - a u8 tag naming the quantization family, used to dispatch the deserializer.
- Family-specific metadata next to it, e.g. `<key>.weight.dtype` and `<key>.weight.shape` for GGML types, `<key>.weight.scales`/`.bits`/`.group_size` for AFQ.
- `<key>.bias` when the layer has one.

`<key>` is the layer's weight path (`model.layers.0.self_attn.q_proj`). MoE expert layers use three canonical keys per block: `<...>.experts.gate_proj`, `.up_proj`, `.down_proj`, each holding the stacked `[num_experts, out, in]` weights.

Because every layer self-describes, a single file may mix quantization types: topology-pinned layers keep their pinned type, and layers whose shape cannot support the requested type fall back per-layer.

## Sharding

The writer splits the tensor stream into `<stem>-0.uqff`, `<stem>-1.uqff`, ... with a soft cap of 10 GiB per shard. Multiple ISQ types in one run produce one shard set per type (`q4k-0.uqff`, `afq4-0.uqff`, ...) sharing the residual and assets.

## Version compatibility

Each shard set carries three u32 scalar entries: `uqff.version.major`, `uqff.version.minor`, `uqff.version.patch`. Readers reject a different major version and reject a minor newer than they support; older minors within the same major are accepted. Files without version entries are rejected.

## Tensor parallelism

Shards store full tensors; under tensor parallelism each rank slices its portion at load time. Slicing the packed (input) dim requires block alignment, which holds for typical model dims; expert layers fall back to replication when it does not.

## Reference implementation

Canonical implementations: `mistralrs-quant/src/uqff` (reader, tensor encoding) and `mistralrs-core/src/pipeline/isq.rs` (writer).

## Caveats

- UQFF is inference-only; no optimizer state or training metadata.
- The export directory is the unit of distribution. A shard alone is not loadable -- the residual safetensors and `config.json` are required.
