---
title: UQFF format
description: Binary layout of the UQFF quantized model file format.
sidebar:
  order: 12
---

UQFF is mistral.rs's native quantized file format. This page documents the on-disk layout for tool authors who want to read or write UQFF files from other software.

If you just want to use a UQFF model, see the [UQFF guide](/mistral.rs/guides/perf/use-uqff/); you do not need to know the binary layout.

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

- `name`: the tensor's logical name in the model (e.g., `model.layers.0.attention.wq.weight`).
- `shape`: logical shape, pre-quantization.
- `isq_type`: which ISQ type was used. See [quantization types](/mistral.rs/reference/quantization-types/) for the name list.
- `offset`: byte offset into the data region.
- `size`: size in bytes. Can be smaller than the original fp16 tensor's bytes due to quantization.
- `original_dtype`: `f16`, `bf16`, or `f32` (the dtype before quantization).

The `metadata` map carries model-level information: original model id, conversion timestamp, calibration data hash if used, and similar.

## Data region

Tensor data follows the header, concatenated back-to-back. Each tensor's bytes are laid out according to its ISQ type's native encoding. Some ISQ types have tensor-level preambles (scales, zero points); those are included in the tensor's allocated `size`.

The data region is not compressed. A gzip or zstd wrapper could be applied at the transport level but is not part of the format.

## Sharded files

For large models, UQFF allows sharding across multiple files. A sharded UQFF uses filenames of the form `model.<isq-type>-<shard>.uqff`, e.g., `model.q4k-0.uqff`, `model.q4k-1.uqff`.

The first shard contains a full header listing all tensors across all shards. Subsequent shards contain only data. Each tensor's `offset` is relative to the start of its shard, and the tensor metadata includes a shard index.

When loading, pass the first shard's filename to mistralrs; subsequent shards are discovered automatically by filename pattern.

## Version compatibility

The version field in the magic-number block is the format version. Backwards compatibility on read is maintained across minor-version changes. A new major-version UQFF will require an updated mistralrs to read, but existing files keep working against newer readers indefinitely.

Writers should emit the highest version they know about to take advantage of format improvements. The current version at the time of writing is `1`.

## Reference implementation

The canonical implementation lives in `mistralrs-quant` (for writing) and in the model loader in `mistralrs-core` (for reading). Both are open source and linked from the main README.

## Caveats

- UQFF does not store optimizer state or training metadata. It is inference-only.
- The format assumes the consumer has the original model's tokenizer. Token embeddings are included, but the tokenizer vocabulary file is not.
- `metadata` entries are advisory; readers should not rely on any particular key being present.
