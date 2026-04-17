---
title: UQFF format
description: Binary layout of the UQFF quantized model file format.
sidebar:
  order: 12
---

UQFF is the native mistral.rs quantized file format. This page documents the on-disk layout for tool authors reading or writing UQFF from other software.

To use UQFF models, see the [UQFF guide](/mistral.rs/guides/perf/use-uqff/) — knowledge of the binary layout is not required.

## File structure

A UQFF file consists of:

1. A magic number (`UQFF` in ASCII, 4 bytes).
2. A version number (u32, little-endian).
3. A header length (u32, little-endian).
4. A header in binary-serialized form (length as above).
5. Tensor data, laid out back-to-back.

All multi-byte integers are little-endian.

## Header

The header is a `bincode`-encoded struct. Its logical shape:

```rust
struct UqffHeader {
    version: u32,
    tensors: Vec<TensorMetadata>,
    metadata: HashMap<String, String>,
}

struct TensorMetadata {
    name: String,
    shape: Vec<usize>,
    isq_type: String,
    offset: u64,  // bytes into the data region
    size: u64,    // size of this tensor's data
    original_dtype: String,
}
```

Field semantics:

- `name` — the tensor's logical name in the model (e.g., `model.layers.0.attention.wq.weight`).
- `shape` — logical shape, pre-quantization.
- `isq_type` — ISQ type used. See [quantization types](/mistral.rs/reference/quantization-types/).
- `offset` — byte offset into the data region.
- `size` — size in bytes. May be smaller than the original fp16 tensor due to quantization.
- `original_dtype` — `f16`, `bf16`, or `f32` (pre-quantization dtype).

`metadata` carries model-level information: original model id, conversion timestamp, calibration data hash, etc.

## Data region

Tensor data follows the header, concatenated back-to-back. Each tensor's bytes use its ISQ type's native encoding. Tensor-level preambles (scales, zero points) for certain ISQ types are included in the tensor's allocated `size`.

The data region is uncompressed. Transport-level gzip or zstd wrappers are out of scope for the format.

## Sharded files

Large models can be sharded across multiple files using filenames `model.<isq-type>-<shard>.uqff` (e.g., `model.q4k-0.uqff`, `model.q4k-1.uqff`).

The first shard contains a full header listing all tensors across all shards. Subsequent shards contain only data. `offset` is relative to the shard start; tensor metadata includes a shard index.

Pass the first shard's filename when loading; subsequent shards are discovered by filename pattern.

## Version compatibility

The version field in the magic-number block is the format version. Backwards compatibility on read is maintained across minor-version changes. A new major-version UQFF requires an updated mistral.rs reader; existing files continue working with newer readers indefinitely.

Writers should emit the highest known version. Current version: `1`.

## Reference implementation

Canonical implementations: `mistralrs-quant` (writer) and the model loader in `mistralrs-core` (reader). Both linked from the main README.

## Caveats

- UQFF does not store optimizer state or training metadata. Inference-only.
- The format assumes consumers have the original tokenizer. Token embeddings are included; the tokenizer vocabulary file is not.
- `metadata` entries are advisory. Readers should not rely on any particular key being present.
