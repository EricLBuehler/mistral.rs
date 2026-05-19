---
title: MatFormer and elastic model sizing
description: MatFormer-trained models in mistralrs.
sidebar:
  order: 3
---

MatFormer encodes multiple model sizes in one checkpoint, with the desired size selected at load time.

## Selecting a slice

A MatFormer slice is selected with two values:

- `matformer_config_path` -- path to the slice config file (CSV or JSON) shipped with the model card.
- `matformer_slice_name` -- the named slice within that file to load.

Without these values, the model loads its default slice.

### CLI

```bash
mistralrs run \
  -m google/gemma-3n-E4B-it \
  --matformer-config-path slices.json \
  --matformer-slice-name E2B
```

The same `--matformer-config-path` / `--matformer-slice-name` flags are accepted by `mistralrs serve` and `mistralrs bench`.

### TOML config

```toml
command = "serve"

[[models]]
kind = "auto"
model_id = "google/gemma-3n-E4B-it"
matformer_config_path = "slices.json"
matformer_slice_name = "E2B"
```

### Python SDK

```python
from mistralrs import Runner, Which, MultimodalArchitecture

runner = Runner(which=Which.MultimodalPlain(
    model_id="google/gemma-3n-E4B-it",
    arch=MultimodalArchitecture.Gemma3n,
    matformer_config_path="slices.json",
    matformer_slice_name="E2B",
))
```

### Rust SDK

`NormalSpecificConfig` and `VisionSpecificConfig` carry `matformer_config_path` and `matformer_slice_name` fields. Pass them when constructing the loader.

## Models

`Gemma3n` (`google/gemma-3n-E4B-it`) is a MatFormer-trained model in the supported list.

## Further reading

The MatFormer paper is linked from the mistral.rs README. Gemma 3n's model card documents its specific MatFormer configuration.
