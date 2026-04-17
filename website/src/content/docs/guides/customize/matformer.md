---
title: MatFormer and elastic model sizing
description: MatFormer-trained models in mistralrs.
sidebar:
  order: 3
---

MatFormer encodes multiple model sizes in one checkpoint, with the desired size selected at load time.

## Availability

`matformer_config_path` and `matformer_slice_name` fields exist in the loader configuration but are not exposed as CLI flags or TOML config options at present. Programmatic configuration is required if needed.

## Models

`Gemma3n` (`google/gemma-3n-E4B-it`) is a MatFormer-trained model in the supported list. Without a MatFormer config, it loads as the default slice.

## Further reading

The MatFormer paper is linked from the mistral.rs README. Gemma 3n's model card documents its specific MatFormer configuration.
