# Model topology configuration

<h3>Quantization and device mapping in one file.</h3>

> [!NOTE]
> Manual device mapping flags are deprecated in favor of automatic placement because it is easy to misconfigure them. Topology files remain the preferred way to express per-layer quantization, and you can still provide `device` overrides here when you truly need to. Those overrides win over the automatic mapper, so apply them sparingly. See the [device mapping documentation](DEVICE_MAPPING.md) for guidance.

Use a simple model topology to configure ISQ and device mapping for *per-layer* with a single [YAML file](https://github.com/EricLBuehler/mistral.rs/blob/master/topologies/isq_and_device.yml) (examples [here](https://github.com/EricLBuehler/mistral.rs/tree/master/topologies))!

To support per-layer mix of ISQ, Mistral.rs supports loading a model topology YAML file. This YAML file is formatted as follows:

1) Top-level keys are either:
    - A range of layers (`start-end`) where `start < end`. `start` is inclusive and `end` is exclusive
    - A single layer number
    2) The topology for the range or layer:
        - An optional key (`isq`) which maps to a single value, which can be any [ISQ type](ISQ.md#isq-quantization-types). If not specified, there is no ISQ for this range of layers applied.
        - An optional key (`device`) which maps to a single value, which is one of the below. If not specified, the default loading deice will be used.
          - `cpu`
          - `cuda[ORDINAL]`
          - `metal[ORDINAL]`

Note that:
- The topology for the range is expanded to fill the range
- If ranges overlap, the range with the higher end layer takes precedence. When two ranges share the same end layer, the one that appears later in the topology file wins.
- Any layers which are not covered will have no topology mapping. They will inherit any other ISQ (e.g. with `--isq`/`in_situ_quant`) set.
- Unless the layer is not covered by the topology, the topology value will override any other ISQ (e.g. with `--isq`/`in_situ_quant`).
- The topology device mapping will override any other device mapping.

### Using topology with UQFF models

When loading a [UQFF](UQFF.md) model, the quantization is already applied during UQFF creation. Therefore:
- **ISQ settings in the topology are ignored** - the pre-quantized weights are used as-is
- **Device mapping still applies** - you can split layers across GPUs or offload to CPU

This is useful for deploying pre-quantized models across multiple devices without re-quantizing.

Example topology for UQFF device mapping:
```yaml
# Only device mapping is used; isq would be ignored
0-16:
  device: cuda[0]
16-32:
  device: cuda[1]
```

See the [UQFF documentation](UQFF.md#using-topology-for-device-mapping-with-uqff) for complete examples.

### Regex selectors

Layer ranges are convenient when you know the numeric index, but you can also target weights by name. Keys wrapped in `/.../` are interpreted as regular expressions that are matched against the fully qualified tensor name (for example, `model.layers.3.attn.q_proj.weight`). Regex selectors may override both `isq` and `device`.

```yml
'/attn\.q_proj$/':
  isq: Q4K
'/ffn_.*\.weight$/':
  isq: Q3K
```

Regex-based ISQ overrides are applied through the immediate ISQ system, so they quantize weights as they are loaded. Numeric layer ranges continue to be handled by the post-load topology pass. Regex selectors are evaluated top-to-bottom as they appear in the YAML file, so a selector that comes later in the file overrides earlier matches.

```yml
0-8:
  isq: Q3K
  device: cuda[0]
8-16:
  isq: Q4K
  device: cpu
16-24:
  isq: Q6K
# Skip 24-28
28-32:
  isq: Q8_0
  device: cuda[0]
```

Model topologies may be applied to all model types.

## CLI example

```
mistralrs run -m microsoft/Phi-3-mini-128k-instruct --topology topologies/isq.yml
```

## HTTP server example

```
mistralrs serve -p 1234 -m microsoft/Phi-3-mini-128k-instruct --topology topologies/isq.yml
```

## Rust example
Example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/topology/main.rs).

## Python example
Example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/topology.py).
