---
title: Configure model topology
description: Per-layer placement and quantization via a YAML topology file.
sidebar:
  order: 8
---

Topology is a per-layer placement and quantization mechanism. A YAML file specifies, per layer range, the device and quantization to use.

Most cases do not need topology. Defaults work for typical hardware; `mistralrs tune` covers common optimization.

## Config

A YAML file keyed by `start-end` layer-range selectors:

```yaml
0-16:
  device: cuda[0]
  isq: q4k
16-32:
  device: cuda[1]
  isq: q4k
32-40:
  device: cpu
  isq: q8_0
```

Layers outside any range use defaults. `device` is a CUDA (`cuda[N]`), Metal (`metal[N]`), or CPU (`cpu`) specifier. `isq` accepts any ISQ type name recognized by `--isq`.

Range selectors match the decoder layer index (the `N` in weight names like `model.layers.N.self_attn.q_proj`). A single layer can be selected with a bare index (`12:`).

Selectors wrapped in slashes are treated as regexes matched against the full weight name, for targeting specific weights instead of whole layers. Later entries win when multiple regexes match:

```yaml
/model\.layers\.\d+\.self_attn\..*/:
  isq: q8_0
/lm_head\..*/:
  device: cpu
```

Topology ISQ pins also apply when producing UQFF files: pinned layers keep their type in every written variant, with the `--isq` value as the default for the rest.

Pass with `--topology`:

```bash
mistralrs serve --topology topology.yaml -m <model>
```

## Notes

Embedding layers, LM head, and pre/post-norm are not individually addressable; they follow the first or last transformer layer's placement.

For an introduction to per-layer quantization tradeoffs, see [the explanation page on quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/).
