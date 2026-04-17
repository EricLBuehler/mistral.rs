---
title: Use pre-quantized UQFF models
description: Our native quantized file format. Loads faster than ISQ, matches its quality, competitive with GGUF.
sidebar:
  order: 9
---

UQFF is the native quantized file format for mistral.rs. A `.uqff` file stores pre-quantized weights and loads directly without runtime conversion, eliminating ISQ's first-run latency.

UQFF matches ISQ in output quality (the underlying math is the same), loads faster (no conversion), and is competitive with GGUF on quality and speed.

## Finding UQFF models

The [mistral.rs organization on Hugging Face](https://huggingface.co/EricB) maintains UQFF versions of frequently tested models. Look for repositories with `uqff` in the name or model card.

Models without an existing UQFF can be converted once and reused.

## Using a UQFF model

CLI:

```bash
mistralrs run --format uqff -m <uqff-repo> -f model.q4k-0.uqff
```

`--format uqff` declares the file shape. `-f` selects the specific file in the repository, since one UQFF repo often contains multiple files at different bit widths (e.g., `model.q4k-0.uqff` and `model.q8_0-0.uqff`).

For locally-stored UQFF files, `-m` takes the local directory and `-f` takes the filename.

## Converting a model to UQFF

The `quantize` subcommand converts an unquantized model to UQFF:

```bash
mistralrs quantize \
  -m google/gemma-4-E4B-it \
  --isq q4k \
  --output gemma-q4k.uqff
```

A one-time operation. The result loads directly afterward.

`--isq` accepts the same values as runtime ISQ: numeric shorthands (`4`, `8`) or format names (`q4k`, `afq4`, `q8_0`).

## Publishing a UQFF

To share a converted model:

1. Upload the UQFF file to a Hugging Face repository.
2. Include the source model card with the quantization settings noted.
3. Place all bit-width variants in one repo; users select with `-f`.

There is no central registry; the Hugging Face hub is the registry.

## Versioning

UQFF files include a format version marker. Format changes are rare and backwards-compatible on read. A "UQFF format too new for this mistralrs version" error means the engine is older than the file producer; updating mistralrs resolves it.

Binary layout details: [UQFF format reference](/mistral.rs/reference/uqff-format/).

## UQFF versus GGUF

Reasons to choose UQFF:

- Slightly better quality at the same bit width on some formats.
- Native support for mistral.rs-specific quant types (AFQ for Metal).
- Cleaner integration with engine features (imatrix, topology).

Reasons to choose GGUF:

- Broad compatibility with llama.cpp tooling.
- More community pre-converted models.
- Works without mistral.rs.

For mistral.rs-only deployments, UQFF is the default. For mixed environments with llama.cpp, GGUF is more interoperable.
