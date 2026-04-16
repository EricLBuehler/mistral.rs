---
title: Use pre-quantized UQFF models
description: Our native quantized file format. Loads faster than ISQ, matches its quality, competitive with GGUF.
sidebar:
  order: 9
---

UQFF is mistral.rs's native quantized file format. A UQFF file is a single `.uqff` blob on disk that stores the model's weights in a pre-quantized form. The engine loads it directly, without doing any conversion at startup, which makes the first-run latency much lower than in-situ quantization.

In benchmarks UQFF matches ISQ in output quality (because the underlying quantization math is the same), loads significantly faster (no conversion step), and is competitive with GGUF on both quality and speed. If a model you want is available in UQFF form, using it is almost always a better experience than converting from scratch every time.

## Finding UQFF models

The [mistral.rs organization on Hugging Face](https://huggingface.co/EricB) maintains UQFF versions of the models we test most heavily. Look for repositories with `uqff` in the name or the model card.

If a model is not already available in UQFF, you can convert it yourself once and reuse it thereafter.

## Using a UQFF model

From the CLI:

```bash
mistralrs run --format uqff -m <uqff-repo> -f model.q4k-0.uqff
```

The `--format uqff` flag tells the engine what shape of file to expect. `-f` specifies the exact file inside the repository, because one UQFF repo often contains several files at different bit widths (for example `model.q4k-0.uqff` and `model.q8_0-0.uqff`).

If your UQFF is stored locally rather than on Hugging Face, `-m` takes the local directory path and `-f` takes the filename.

## Converting a model to UQFF

The `quantize` subcommand takes an unquantized model and writes a UQFF:

```bash
mistralrs quantize \
  -m google/gemma-4-E4B-it \
  --isq q4k \
  --output gemma-q4k.uqff
```

This is a one-time operation. The resulting file can be loaded directly from then on. For models you load repeatedly, this saves a lot of cumulative time.

The `--isq` flag here accepts the same values it does for runtime ISQ: numeric shorthands like `4` or `8`, or format names like `q4k`, `afq4`, `q8_0`, and so on.

## Publishing a UQFF

If you convert a model and would like to share it:

1. Put the UQFF file in a Hugging Face repository.
2. Include the model card from the source model, with a note about the quantization settings.
3. If you converted at multiple bit widths, put all the files in one repo; users can pick the one they want with `-f`.

There is no central registry; the Hugging Face hub is the registry.

## Versioning

UQFF files include a format version marker. When we change the format (rare, and always backwards-compatible on read), older files keep loading. If you see a "UQFF format too new for this mistralrs version" error, you are running an older engine against a file produced by a newer one; updating mistralrs fixes it.

The detailed binary layout is documented in the [UQFF format reference](/mistral.rs/reference/uqff-format/) for tool authors.

## UQFF versus GGUF

Both formats serve similar purposes. Reasons to pick UQFF:

- Slightly better quality on the same bit width, because the conversion preserves more information in some formats.
- Native support for mistral.rs-specific quant types (AFQ for Metal).
- Cleaner integration with the engine's features (imatrix, topology).

Reasons to pick GGUF:

- Broader compatibility with other llama.cpp-based tools.
- More models available pre-converted by the community.
- Works without mistral.rs installed.

For mistral.rs-only deployments, UQFF is the better default. For mixed environments where you use llama.cpp or similar alongside mistralrs, GGUF is the more interoperable choice.
