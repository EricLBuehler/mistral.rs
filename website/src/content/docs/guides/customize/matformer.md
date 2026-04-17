---
title: MatFormer and elastic model sizing
description: Run a slice of a MatFormer-trained checkpoint at a size you choose at load time.
sidebar:
  order: 3
---

MatFormer encodes multiple model sizes inside one checkpoint. The desired size is selected at load time.

Practical benefit: one download, multiple deployable sizes. Useful when serving different sizes on different hardware from a single base artifact.

## When it applies

MatFormer requires a model trained for it. Gemma 3n is the main example in the mistral.rs ecosystem. Regular checkpoints cannot be sliced.

## Loading a specific slice

Specify the slice in a config file:

```toml
[matformer]
config_path = "matformer_config.toml"
slice = "E4B"
```

`slice` names the size variant. Each MatFormer model documents its slices; Gemma 3n typically offers `E2B` (~2B effective parameters) and `E4B` (~4B).

Pass the config:

```bash
mistralrs run \
  -m google/gemma-3n-E4B-it \
  --matformer-config matformer.toml
```

## What changes per slice

Each slice represents a different width/depth balance:

- Fewer parameters mean less memory and faster inference.
- Output quality scales with slice size, similar to choosing a smaller model in the same family.
- Tokenizer and chat template are unchanged across slices; only weights differ.

A MatFormer checkpoint is several model sizes sharing weights efficiently. The smallest slice is equivalent to a small dedicated model; the largest is the full model.

## Slicing and quantization

MatFormer composes with ISQ:

```bash
mistralrs run \
  -m google/gemma-3n-E4B-it \
  --matformer-config matformer.toml \
  --isq 4
```

This stacks the slice's memory savings with quantization's.

## What you cannot do

The slice is fixed at load time. Dynamic slice switching at runtime is not supported. To serve multiple slices, run them as separate models with [multi-model](/mistral.rs/guides/serve/multiple-models/).

MatFormer cannot be applied to non-MatFormer-trained checkpoints. Attempting to do so fails at load time with an explanatory error.

## Further reading

The MatFormer paper (linked from the mistral.rs README) covers architecture details. Gemma 3n's model card documents its specific MatFormer configuration.
