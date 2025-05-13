# Model topology configuration

<h3>Quantization and device mapping in one file.</h3>

> [!NOTE]
> Manual device mapping is deprecated in favor of automatic device mapping due to the possibility for user error in manual.
> The topology system will remain and be used only for quantization settings.
> Please see the [device mapping documentation](DEVICE_MAPPING.md) for more information.

Use a simple model topology to configure ISQ and device mapping for *per-layer* with a single [YAML file](../topologies/isq_and_device.yml) (examples [here](../topologies))!

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
- If ranges overlap, the range with the higher end layer takes precedence and will overwrite
- Any layers which are not covered will have no topology mapping. They will inherit any other ISQ (e.g. with `--isq`/`in_situ_quant`) set.
- Unless the layer is not covered by the topology, the topology value will override any other ISQ (e.g. with `--isq`/`in_situ_quant`).
- The topology device mapping will override any other device mapping.
- When using UQFF, only the device mapping is relevant.


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

> [!NOTE]
> You should replace `--features ...` with one of the features specified [here](../README.md#supported-accelerators), or remove it for pure CPU inference.

```
cargo run --features ... -- -i plain -m microsoft/Phi-3-mini-128k-instruct --topology topologies/isq.yml   
```

## HTTP server example

> [!NOTE]
> You should replace `--features ...` with one of the features specified [here](../README.md#supported-accelerators), or remove it for pure CPU inference.

```
cargo run --features ... -- --port 1234 plain -m microsoft/Phi-3-mini-128k-instruct --topology topologies/isq.yml   
```

## Rust example
Example [here](../mistralrs/examples/topology/main.rs).

## Python example
Example [here](../examples/python/topology.py).