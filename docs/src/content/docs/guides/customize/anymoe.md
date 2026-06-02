---
title: AnyMoE
description: Compose a mixture-of-experts model at inference time from several existing fine-tunes.
sidebar:
  order: 2
---

AnyMoE combines several fine-tuned models of the same base into one mixture-of-experts configuration without retraining.

## Availability

AnyMoE is exposed through the **Rust SDK** (`AnyMoeModelBuilder`) and the **Python SDK** (`AnyMoeConfig`, `AnyMoeExpertType`). It is not configurable via the CLI or the TOML config -- AnyMoE wraps the base loader in a custom way that does not map cleanly onto the standard `ModelSelected` surface.

## Rust SDK

Use `AnyMoeModelBuilder` from the `mistralrs` crate. Working examples live in `examples/advanced/anymoe/` and `examples/advanced/anymoe_lora/` in the source repository.

## Python SDK

`AnyMoeConfig` and `AnyMoeExpertType` are exposed as Python classes. Pass an `AnyMoeConfig` to the `Runner` constructor along with the expert paths, gating model path, and target layers. See `examples/python/anymoe.py` for a working example.

## Notes

Expert checkpoints must share the base model architecture. A small calibration dataset is required to train the per-routed-module router.

For the underlying technique and tradeoffs, the AnyMoE paper is linked from the mistral.rs README.
