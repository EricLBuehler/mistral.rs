---
title: Use pre-quantized UQFF models
description: Native quantized file format. Loads directly without runtime conversion.
sidebar:
  order: 11
---

UQFF (Universal Quantized File Format) stores pre-quantized weights and loads directly without runtime conversion.

:::caution
UQFF files produced by pre-1.0 mistral.rs releases are not loadable by current builds and fail with a clear error. Regenerate them with `mistralrs quantize`.
:::

## Using a UQFF model

```bash
mistralrs run -m <repo> --from-uqff model.q4k-0.uqff
```

`-m <repo>` is required for tokenizer/base resolution. `--from-uqff` accepts a numeric shorthand (`2`, `3`, `4`, `5`, `6`, `8`) or an ISQ type name (`q4k`, `afq8`, etc.).

For sharded UQFFs, pass the first shard's filename. Subsequent shards are discovered by filename pattern (`model.<isq-type>-0.uqff`, `model.<isq-type>-1.uqff`, ...).

For locally-stored UQFF files, `-m` can be the local directory and `--from-uqff` the filename.

UQFF models work under tensor parallelism: each rank loads only its slice of the quantized weights.

## Producing a UQFF

The `quantize` subcommand converts an unquantized model to UQFF:

```bash
mistralrs quantize \
  -m google/gemma-4-E4B-it \
  --isq q4k \
  -o gemma-q4k.uqff
```

A one-time operation. The result loads directly afterward.

`--isq` can be repeated or comma-separated to produce multiple variants in one run; pass a directory as `-o` in that case. Numeric shorthands expand to all platform variants (`--isq 4` writes both `afq4.uqff` and `q4k.uqff`).

When `write_uqff` is used from the Rust or Python SDK and the session keeps serving, the in-memory model runs as the first requested type.

A [topology](/mistral.rs/guides/perf/topology/) can pin specific layers to a different type (e.g. keep `lm_head` at `q8_0` in an otherwise Q4K file); pins are preserved in every output variant.

A README is generated alongside the output unless `--no-readme` is passed. `--uqff-base-model` and `--uqff-repo-id` set fields in the README.

K-quant output quality can be improved with an importance matrix: pass `--imatrix <file>` (llama.cpp `.imatrix` files work directly) or `--calibration-file <text>` to `quantize`. See [quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/#imatrix).

## Format details

Layout and versioning: [UQFF format reference](/mistral.rs/reference/uqff-format/).
