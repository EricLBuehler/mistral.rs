# Universal Quantized File Format: UQFF

<h3 align="left">
The uniquely powerful quantized file format.
</h3>

1) **Flexible** 🌀: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Reliable** 🔒: Compatibility ensured with *embedded* and *checked* semantic versioning information from day 1.
3) **Easy** 🤗: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
3) **Customizable** 🛠️: Make and publish your own UQFF files in minutes.

# ToC
- [Motivation](#motivation)
- [Support](#support)
- [Memory layout (*for developers*)](UQFF/LAYOUT.md)

## Motivation

UQFF builds on our ISQ feature by allowing serialization and deserialization for models.

While ISQ is a powerful feature enabling easy quantization of models, the key limitation has been the time required for requantization. While the process is relatively fast with parallelization and other techniques, multiple runs can make the experience slow. 

UQFF allows everyone to experience the power and flexibility of ISQ. All in one file format.

## Support

The following quantization formats are supported in UQFF. One can, of course, be combined arbitrarily during UQFF generation or ISQ using a [model topology](TOPOLOGY.md). When loading a UQFF model, only the per-layer device mapping feature of the topology applies.

- GGUF quantized:
    - Q4_0
    - Q4_1
    - Q5_0
    - Q5_1
    - Q8_0
    - Q8_1 (*not available on CUDA*)
    - Q2K
    - Q3K
    - Q4K
    - Q5K
    - Q6K
    - Q8K  (*not available on CUDA*)

- HQQ quantized:
    - HQQ4
    - HQQ8

## Loading a UQFF model

To load a UQFF model, one should specify the artifact path. This can be either be a path to a UQFF file locally, or a Hugging Face model ID with the format `<MODEL ID>/<FILE>`. For example, the following work:

- `EricB/Phi-3.5-mini-instruct-ISQ/phi3.5-mini-isq-q4k.safetensors`
- `../UQFF/phi3.5-mini-isq-q4k.safetensors`

> Note: when loading an UQFF model, it will take precedence over any ISQ setting.

### Running with the CLI

```
cargo run --features cuda -- -i plain -m microsoft/Phi-3.5-mini-instruct --load-isq-artifact EricB/Phi-3.5-mini-instruct-ISQ/phi3.5-mini-isq-q4k.safetensors
```

### Using with the Rust API

Modify the Normal or Vision config as follows:

```diff
NormalSpecificConfig {
    use_flash_attn: false,
    prompt_batchsize: None,
    topology: None,
    organization: Default::default(),
    write_uqff: None,
-   from_uqff: None,
+   from_uqff: Some("EricB/Phi-3.5-mini-instruct-ISQ/phi3.5-mini-isq-q4k.safetensors".to_string()),
}
```

```diff
VisionSpecificConfig {
    use_flash_attn: false,
    prompt_batchsize: None,
    topology: None,
    write_uqff: None,
-   from_uqff: None,
+   from_uqff: Some("../UQFF/phi3.5-mini-isq-q4k.safetensors".to_string()),
}
```

### Using the Python API
Modify the `Which` instantiation as follows:
```diff
Which.Plain(
    model_id="microsoft/Phi-3.5-mini-instruct",
+   from_uqff="EricB/Phi-3.5-mini-instruct-ISQ/phi3.5-mini-isq-q4k.safetensors"
),
```
