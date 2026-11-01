---
title: UQFF format
description: Layout of the UQFF quantized model file format.
---

UQFF is mistral.rs's own file format for quantized models. This page describes its layout. You don't need it to use UQFF files; for that, see the [UQFF guide](/guides/quantization/uqff/).

## File structure

A UQFF export is a directory containing:

- One or more `<stem>-<shard>.uqff` shards holding the quantized layers.
- `residual.safetensors` for unquantized tensors (norms, dense embeddings, etc.).
- Model assets copied from the source repo so the directory is self-contained: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and (when present) `modules.json`, `chat_template.jinja`, `processor_config.json`, `preprocessor_config.json`.

To load an export, pass one or more shard files (`--from-uqff` on the CLI). mistral.rs finds `residual.safetensors` and the JSON assets in the same repository or directory.

## Shard layout

Each `.uqff` shard is a standard safetensors file with named entries. Every quantized layer is self-describing:

- `<key>.weight` - the layer data (raw blocks for GGML-family types, packed tensors for AFQ/MXFP4/FP8, or a native safetensors tensor for unquantized fallback layers; see [quantization types](/reference/quantization-types/)).
- `<key>.weight.format` - a u8 tag naming the quantization family, used to dispatch the deserializer.
- Family-specific metadata next to it, e.g. `<key>.weight.dtype` and `<key>.weight.shape` for GGML types, `<key>.weight.scales`/`.bits`/`.group_size` for AFQ.
- `<key>.bias` when the layer has one.

The safetensors metadata also records which mistral.rs build wrote the file: `uqff.producer`, `uqff.producer.mistralrs.version`, and `uqff.producer.mistralrs.git_revision`. The loader ignores these fields.

`<key>` is the layer's weight path (`model.layers.0.self_attn.q_proj`). MoE (Mixture of Experts) expert layers use three canonical keys per block: `<...>.experts.gate_proj`, `.up_proj`, `.down_proj`, each holding the stacked `[num_experts, out, in]` weights.

Each layer records its own type, so one file can mix quantization types. That happens in three cases:

- Layers pinned to a type in a [topology](/guides/perf/topology/) config keep that type.
- Token embeddings and the output head are stored at higher precision, as described in [sensitive tensor precision](/reference/quantization-types/#sensitive-tensor-precision). For example, an `afq4` shard set stores them as AFQ6, and a `q4k` shard set stores them as Q6K.
- Layers whose shape can't use the requested type fall back one layer at a time. For example, AFQ layers whose input dimension isn't divisible by the AFQ group size are stored unquantized.

## Sharding

The writer splits the quantized layers across `<stem>-0.uqff`, `<stem>-1.uqff`, and so on, starting a new shard at about 10 GiB. Quantizing to several [ISQ](/reference/quantization-types/) types in one run writes one shard set per type (`q4k-0.uqff`, `afq4-0.uqff`, ...), all sharing the same residual file and assets.

## Version compatibility

Each shard set carries three u32 scalar entries: `uqff.version.major`, `uqff.version.minor`, and `uqff.version.patch`. The current format version is 1.2.0. A file fails to load with a version error when:

- Its major version differs from the reader's. Regenerate it with `mistralrs quantize`.
- Its minor version is newer than the reader supports. Upgrade mistral.rs.
- It has no version entries. Regenerate it with `mistralrs quantize`.

Older minor versions within the same major version load normally. What each minor version added:

- 1.1: unquantized linear layers can be stored inline (`weight.format = Unquant`), so a mixed file can keep a layer whose shape doesn't fit the requested type without moving it into `residual.safetensors`.
- 1.2: quantized token embeddings are stored as regular layer entries, and their dense weights are left out of `residual.safetensors`. 1.1 files, which keep token embeddings in the residual file, still load.

## Tensor parallelism

Shards store full tensors; under tensor parallelism each rank slices its portion at load time. Slicing the packed (input) dim requires block alignment, which holds for typical model dims. When alignment does not hold (some expert layers), the rank replicates the full tensor instead of slicing.

## Reference implementation

Canonical implementations: `mistralrs-quant/src/uqff` (reader, tensor encoding) and `mistralrs-core/src/pipeline/isq.rs` (writer).

## Caveats

- UQFF is inference-only; no optimizer state or training metadata.
- The export directory is the unit of distribution. A shard alone can't load: it needs the residual safetensors and `config.json` next to it.
