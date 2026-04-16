---
title: Configure model topology
description: Per-layer placement, per-layer quantization, and hand-tuning when the defaults are not good enough.
sidebar:
  order: 7
---

The topology feature is a hand-tuning lever for advanced cases where the default placement and quantization are not what you want. You feed the engine a YAML file that specifies, layer by layer, which device each layer lives on and what quantization it uses.

Most people never need this. The defaults are reasonable and `mistralrs tune` covers the common optimization cases. Topology is for the remaining cases: unusual hardware combinations, experimentation, or deployment-specific constraints that the automatic placement does not handle well.

## When to reach for it

- **Uneven GPU memory.** One GPU has less free VRAM than another (maybe something else is running on it). You want to put fewer layers on the smaller card.
- **Specific attention placement.** You want attention layers quantized less aggressively than MLP layers, because attention is more sensitive to quantization.
- **CPU offload.** You want specific layers to run on CPU to fit a larger model. This is slow, but sometimes it is the difference between running at all and not.
- **Experimentation.** Ablation studies, per-layer profiling, anything where you want to know exactly what is where.

## The config file

Topology is specified in a YAML file. Each entry matches a range of layers:

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

Layers outside any range use the defaults. The `device` field is a standard CUDA (`cuda:N`), Metal (`metal:N`), or CPU (`cpu`) specifier. The `isq` field accepts any ISQ type name recognized by `--isq`.

Pass the file to the CLI with `--topology`:

```bash
mistralrs serve --topology topology.yaml -m <model>
```

## How layer numbers work

Layer numbering is zero-indexed and corresponds to the model's transformer blocks. A 32-layer model has layers 0 through 31. Different models have different layer counts; `mistralrs doctor -m <model>` shows the count among other things.

Embedding layers, LM head, and any pre/post-norm are not individually addressable in the topology format. They follow the first or last transformer layer's device placement.

## Per-layer quantization tradeoffs

Different layers tolerate quantization differently. A common pattern is to keep attention layers at 8 bits (they are small and sensitive) while quantizing MLP layers to 4 bits (they dominate the memory budget and tolerate it well). The research on this is still developing; what is clear is that uniform quantization across all layers is rarely optimal, and topology is how you express a non-uniform scheme.

If you are doing this seriously, the [explanation page on quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/) has pointers to the underlying research.

## Saving a tune recommendation as topology

`mistralrs tune --emit-config <file>` produces a TOML that specifies a single quantization level and device for the whole model. If you want to use tune's output as a starting point for per-layer experimentation, convert it to YAML topology manually; there is no automatic conversion yet.

## Validation

The topology is validated at startup. If you assign more layers than the model has, refer to a device that does not exist, or specify an ISQ type that is not supported on the referenced device, the server refuses to start and reports which entry is wrong. It does not silently ignore bad configuration.

## A note on complexity

Topology is a power tool. Getting it wrong can produce subtly bad output (quality degradation) rather than obvious errors (crashes). If you are tuning for speed and memory, `mistralrs tune` is almost always the better starting point. Pull out topology only when the automatic options have been exhausted.
