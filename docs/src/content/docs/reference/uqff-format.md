---
title: UQFF format
description: Binary layout of the UQFF quantized model file format.
sidebar:
  order: 12
---

UQFF is the native mistral.rs quantized file format. To use UQFF models, see the [UQFF guide](/mistral.rs/guides/perf/use-uqff/); knowledge of the binary layout is not required.

## File structure

A UQFF export is a directory containing:

- One or more `<stem>-<shard>.uqff` shards holding the quantized layer blobs.
- `residual.safetensors` for unquantized tensors (token embeddings, norms, lm_head, etc.).
- Model assets copied from the source repo so the directory is self-contained: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and (when present) `modules.json`, `chat_template.jinja`, `processor_config.json`, `preprocessor_config.json`.

A loader is pointed at one or more shard files (`from_uqff`); the residual safetensors and the JSON assets are picked up by sibling-path lookup.

## Sharded files

The output path passed to mistral.rs must end in `.uqff`. The writer emits `<stem>-0.uqff`, `<stem>-1.uqff`, ... in the same directory, splitting by cumulative size with a soft cap of 10 GiB per shard.

## Version compatibility

UQFF carries a packed `major.minor.patch` version. Readers reject a different major version and reject a minor newer than they support; older minor versions within the same major are accepted.

## Reference implementation

Canonical implementations: `mistralrs-quant` (writer) and the model loader in `mistralrs-core` (reader).

## Caveats

- UQFF is inference-only; no optimizer state or training metadata.
- The export directory is the unit of distribution. A shard alone is not loadable -- the residual safetensors and `config.json` are required.
