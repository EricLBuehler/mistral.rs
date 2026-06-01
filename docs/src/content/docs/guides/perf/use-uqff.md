---
title: Use pre-quantized UQFF models
description: Native quantized file format. Loads directly without runtime conversion.
sidebar:
  order: 11
---

UQFF (Universal Quantized File Format) stores pre-quantized weights and loads directly without runtime conversion.

## Using a UQFF model

```bash
mistralrs run -m <repo> --from-uqff model.q4k-0.uqff
```

`-m <repo>` is required for tokenizer/base resolution. `--from-uqff` accepts a numeric shorthand (`2`, `3`, `4`, `5`, `6`, `8`) or an ISQ type name (`q4k`, `afq8`, etc.).

For sharded UQFFs, pass the first shard's filename. Subsequent shards are discovered by filename pattern (`model.<isq-type>-0.uqff`, `model.<isq-type>-1.uqff`, ...).

For locally-stored UQFF files, `-m` can be the local directory and `--from-uqff` the filename.

## Producing a UQFF

The `quantize` subcommand converts an unquantized model to UQFF:

```bash
mistralrs quantize \
  -m google/gemma-4-E4B-it \
  --isq q4k \
  -o gemma-q4k.uqff
```

A one-time operation. The result loads directly afterward.

`--isq` can be repeated or comma-separated to produce multiple variants in one run; pass a directory as `-o` in that case.

A README is generated alongside the output unless `--no-readme` is passed. `--uqff-base-model` and `--uqff-repo-id` set fields in the README.

## Format details

Binary layout: [UQFF format reference](/mistral.rs/reference/uqff-format/).
