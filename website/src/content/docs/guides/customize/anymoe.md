---
title: AnyMoE
description: Compose a mixture-of-experts model at inference time from several existing fine-tunes.
sidebar:
  order: 2
---

AnyMoE combines several fine-tuned models of the same base into one mixture-of-experts configuration without retraining.

## Availability

AnyMoE is exposed through the Rust SDK only. It is not configurable via the CLI or the TOML config.

## Rust SDK

Use `AnyMoeModelBuilder` from the `mistralrs` crate. Working examples live in `examples/advanced/anymoe/` and `examples/advanced/anymoe_lora/` in the source repository.

## Notes

Expert checkpoints must share the base model architecture. A small calibration dataset is required to train the per-routed-module router.

For the underlying technique and tradeoffs, the AnyMoE paper is linked from the mistral.rs README.
