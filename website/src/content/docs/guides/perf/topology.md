---
title: Configure model topology
description: Per-layer placement, per-layer quantization, and hand-tuning when the defaults are not good enough.
sidebar:
  order: 7
---

Topology is a per-layer placement and quantization configuration mechanism. A YAML file specifies, layer by layer, the device and quantization to use.

Most cases do not need topology. Defaults are reasonable and `mistralrs tune` covers common optimization. Topology applies to unusual hardware, experimentation, or deployment-specific constraints automatic placement does not address.

## When to reach for it

- **Uneven GPU memory.** One GPU has less free VRAM than another. Place fewer layers on the smaller card.
- **Specific attention placement.** Quantize attention layers less aggressively than MLP layers because attention is more sensitive.
- **CPU offload.** Place specific layers on CPU to fit a larger model. Slow, but sometimes the only option.
- **Experimentation.** Ablation studies and per-layer profiling.

## The config file

Topology is a YAML file. Each entry matches a layer range:

```yaml
- range:
    start: 0
    end: 16
  device: "cuda:0"
  isq: "q4k"
- range:
    start: 16
    end: 32
  device: "cuda:1"
  isq: "q4k"
- range:
    start: 32
    end: 40
  device: "cpu"
  isq: "q8_0"
```

Layers outside any range use defaults. `device` is a CUDA (`cuda:N`), Metal (`metal:N`), or CPU (`cpu`) specifier. `isq` accepts any ISQ type name recognized by `--isq`.

Pass with `--topology`:

```bash
mistralrs serve --topology topology.yaml -m <model>
```

## Layer numbering

Zero-indexed, corresponding to transformer blocks. A 32-layer model has layers 0–31. `mistralrs doctor -m <model>` reports the count.

Embedding layers, LM head, and pre/post-norm are not individually addressable. They follow the first or last transformer layer's placement.

## Per-layer quantization tradeoffs

Layers tolerate quantization differently. A common pattern: 8-bit attention (small, sensitive) and 4-bit MLP (large, tolerant). The research is evolving; uniform quantization is rarely optimal, and topology expresses non-uniform schemes.

For background, see the [explanation page on quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/).

## Saving a tune recommendation as topology

`mistralrs tune --emit-config <file>` produces a TOML with a single quantization level and device for the whole model. Manual conversion to YAML topology is required; no automatic conversion exists.

## Validation

Topology is validated at startup. Invalid entries (out-of-range layers, nonexistent devices, unsupported ISQ for the device) cause a startup refusal with the offending entry identified.

## A note on complexity

Topology is a power tool. Misconfiguration produces subtle quality degradation rather than crashes. For speed and memory tuning, `mistralrs tune` is the better starting point. Use topology only when automatic options are exhausted.
