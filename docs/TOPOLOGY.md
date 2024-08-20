# Model topology configuration

To support per-layer mix of ISQ, Mistral.rs supports loading a model topology YAML file. This YAML file is formatted as follows:

1) Top-level keys are either:
    - A range of layers (`start-end`) where `start < end`. `start` is inclusive and `end` is inclusive
    - A single layer number
    2) The topology for the range or layer:
        - A single key (`isq`) which mapps to a single value, which can be any [ISQ type](ISQ.md#isq-quantization-types)

Note that:
- The topology for the range is expanded to fill the range
- If ranges overlap, the range with the higher end layer takes precedence and will overwrite
- Any layers which are not covered will have no topology mapping. They will inherit any other ISQ (e.g. with `--isq`/`in_situ_quant`) set.
- Unless the layer is not covered by the topology, the topology value will override any other ISQ (e.g. with `--isq`/`in_situ_quant`).


```yml
0-8:
  isq: Q3K
8-16:
  isq: Q4K
16-24:
  isq: Q6K
# Skip 24-28
28-32:
  isq: Q8_0
```

Model topologies may be applied to the following model types:
- `plain`/`Plain`
- `xlora`/`XLora`
- `lora`/`Lora`
- `vision-plain`/`VisionPlain`

## CLI example
```
cargo run --features ... -- -i plain -m microsoft/Phi-3-mini-128k-instruct -a phi3 --topology topologies/isq.yml   
```

## HTTP server example
```
cargo run --features ... -- --port 1234 plain -m microsoft/Phi-3-mini-128k-instruct -a phi3 --topology topologies/isq.yml   
```

## Rust example
Example [here](../mistralrs/examples/topology/main.rs).

## Python example
Example [here](../examples/python/topology.py).