---
title: MatFormer and elastic model sizing
description: Run a slice of a MatFormer-trained checkpoint at a size you choose at load time.
sidebar:
  order: 3
---

MatFormer is an architecture where a single trained checkpoint can be "sliced" into smaller, still-functional models at load time. The checkpoint encodes multiple sizes nested inside it; you pick which size you want when you load it.

The practical benefit: one download, many deployable sizes. If you are serving different-sized models on different hardware, MatFormer lets you use one base artifact and slice it per deployment rather than storing several full checkpoints.

## When it applies

MatFormer only works with models that were specifically trained for it. Gemma 3n is the main example in the mistral.rs ecosystem. There is no way to "slice" a regular checkpoint; the model has to be MatFormer-trained from the start.

If you are not running a MatFormer-trained model, this feature is not relevant.

## Loading a specific slice

Specify the slice in a config file:

```toml
[matformer]
config_path = "matformer_config.toml"
slice = "E4B"
```

`slice` names the size variant you want. Each MatFormer model documents its available slices; typical options for Gemma 3n include `E2B` (about 2B effective parameters) and `E4B` (about 4B).

Pass the config at startup:

```bash
mistralrs run \
  -m google/gemma-3n-E4B-it \
  --matformer-config matformer.toml
```

## What changes per slice

Each slice represents a different balance of width and depth:

- Fewer parameters means less memory and faster inference.
- Output quality scales with slice size, roughly following the same curve as picking a smaller model from the same family.
- The tokenizer and chat template are unchanged across slices; only the weights differ.

A useful mental model: a MatFormer checkpoint is several model sizes sharing weights efficiently. Loading the smallest slice is equivalent to loading a small dedicated model, and loading the largest is equivalent to loading the full model.

## Slicing and quantization

MatFormer and ISQ compose. You can load a slice and then quantize it:

```bash
mistralrs run \
  -m google/gemma-3n-E4B-it \
  --matformer-config matformer.toml \
  --isq 4
```

This gives you a quantized version of the chosen slice, stacking both memory savings.

## What you cannot do

MatFormer does not support dynamic slice switching at runtime. The slice is chosen at load time and is fixed for the life of the process. To serve multiple slices, run them as separate models with [multi-model](/mistral.rs/guides/serve/multiple-models/).

MatFormer also does not work as a drop-in for a model that was not MatFormer-trained. Attempting to apply it to a regular checkpoint will fail at load time with an informative error.

## Further reading

The MatFormer paper (linked from the mistral.rs README) has the full architecture details. Gemma 3n's model card covers the specifics of its MatFormer configuration.
