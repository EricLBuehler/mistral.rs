---
title: Customize
description: Adapters, AnyMoE, MatFormer, chat templates, sampling, and TOML-based configuration.
---

These guides cover the pieces of mistral.rs you reach for when the defaults are not quite what you want.

- [LoRA and X-LoRA adapters](/mistral.rs/guides/customize/lora-adapters/): attach fine-tuned adapters to a base model.
- [AnyMoE](/mistral.rs/guides/customize/anymoe/): mixing multiple experts at inference time.
- [MatFormer](/mistral.rs/guides/customize/matformer/): elastic model sizing, picking a specific slice of a MatFormer-trained checkpoint.
- [Chat templates](/mistral.rs/guides/customize/chat-templates/): when the auto-detected template is wrong or missing.
- [Sampling parameters](/mistral.rs/guides/customize/sampling/): temperature, top-k, top-p, min-p, DRY, and how they interact.
- [TOML selector](/mistral.rs/guides/customize/toml-selector/): building requests from TOML instead of JSON, for config-driven deployments.
