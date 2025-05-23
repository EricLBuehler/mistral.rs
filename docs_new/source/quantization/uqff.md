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
- [Loading a UQFF model](#loading-a-uqff-model)
- [Creating a UQFF model](#creating-a-uqff-model)
- [List of models](#list-of-models)
- [Memory layout (*for developers*)](UQFF/LAYOUT.md)

## Motivation

UQFF builds on our ISQ feature by allowing serialization and deserialization for models.

While ISQ is a powerful feature enabling easy quantization of models, the key limitation has been the time required for requantization. While the process is relatively fast with parallelization and other techniques, multiple runs can make the experience slow. 

**Comparting UQFF to GGUF:**

In contrast to GGUF, which only supports the GGUF quantizations, UQFF is designed with flexibiliuty in mind. At its code, it extends the power and flexibility of ISQ. The ability to support multiple quantization types (more to come!) in one simple, easy-to-use file is a critical feature.

Additionally, users will no longer need to wait for GGUF support to begin using post-training quantized models. As we add new models and quantization schemes to mistral.rs, the feature set of UQFF will grow.

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

- FP8:
    - FP8 E4M3 (4-bit exponent, 3-bit mantissa)

## Loading a UQFF model

To load a UQFF model, one should specify the filename. This will be located based on the model ID, and can
be loaded locally or from Hugging Face based on the model ID.

- `phi3.5-mini-instruct-q4k.uqff`
- `../UQFF/phi3.5-mini-instruct-q4k.uqff`

You can find a [collection of UQFF models here](https://huggingface.co/collections/EricB/uqff-670e4a49d56ecdd3f7f0fd4c), which each include a simple
command to get started.

> Note: when loading an UQFF model, *any* ISQ setting will be ignored.

### Running with the CLI

```
./mistralrs-server -i plain -m EricB/Phi-3.5-mini-instruct-UQFF --from-uqff phi3.5-mini-instruct-f8e4m3.uqff
```

### Using with the Rust API

Modify the Normal or Vision config as follows and update the model ID to point to a UQFF model:

```diff
NormalSpecificConfig {
    use_flash_attn: false,
    prompt_batchsize: None,
    topology: None,
    organization: Default::default(),
    write_uqff: None,
-   from_uqff: None,
+   from_uqff: Some("phi3.5-mini-instruct-q4k.uqff".to_string()), // Pull from specified HF hub repo
}
```

```diff
VisionSpecificConfig {
    use_flash_attn: false,
    prompt_batchsize: None,
    topology: None,
    write_uqff: None,
-   from_uqff: None,
+   from_uqff: Some("../phi3.5-mini-instruct-q4k.uqff".to_string()), // Local path
}
```

### Using the Python API
Modify the `Which` instantiation as follows:
```diff
Which.Plain(
    model_id="EricB/Phi-3.5-mini-instruct-UQFF",
+   from_uqff="phi3.5-mini-instruct-q4k.uqff"
),
```


## Creating a UQFF model

Creating a UQFF model requires you to generate the UQFF file.
- This means specifying a local path to a file ending in `.uqff`, where your new UQFF model will be created.
- The quantization of a UQFF model is determined from the ISQ or model topology (see the [topology docs](TOPOLOGY.md) for more details on how ISQ and the topology mix).

Along with the UQFF file, the generation process will also output several `.json` configuration files and `residual.safetensors`. All of these files are considered the
UQFF model, and should be kept together or uploaded.

> Note: Only the `.uqff` files are unique to the quantization level(s). If you are generating multiple UQFF files, it is OK for the others to be overwritten.

After creating the UQFF file, you can upload the model to Hugging Face. To do this:
1) [Create a new model](https://huggingface.co/docs/transformers/v4.17.0/en/create_a_model).
2) Upload the UQFF file:
    - With the web interface: [guide here](https://huggingface.co/docs/hub/en/models-uploading#using-the-web-interface).
    - With Git: [steps here](#upload-with-git-lfs)
3) Locally, generate the model card file with [this Python script](../scripts/generate_uqff_card.py)..
4) In the web interface, press the `Create Model Card` button and paste the generated model card.

### Creating with the CLI

```
./mistralrs-server --isq Q4K -i plain -m microsoft/Phi-3.5-mini-instruct --write-uqff phi3.5-mini-instruct-q4k.uqff
```

### Creating with the Rust API

Modify the Normal or Vision config as follows:

```diff
NormalSpecificConfig {
    use_flash_attn: false,
    prompt_batchsize: None,
    topology: None,
    organization: Default::default(),
    from_uqff: None,
-   write_uqff: None,
+   write_uqff: Some("phi3.5-mini-instruct-q4k.uqff".to_string()),
}
```

```diff
VisionSpecificConfig {
    use_flash_attn: false,
    prompt_batchsize: None,
    topology: None,
    from_uqff: None,
-   write_uqff: None,
+   write_uqff: Some("../UQFF/phi3.5-mini-instruct-q4k.uqff".to_string()),
}
```

### Creating with the Python API
Modify the `Which` instantiation as follows. Be sure to add the `in_situ_quant`.
```diff
Which.Plain(
    model_id="microsoft/Phi-3.5-mini-instruct",
+   write_uqff="phi3.5-mini-instruct-q4k.uqff"
),
```

### Upload with Git
To upload a UQFF model using Git, you will most likely need to set up Git LFS:

1) Install [git-lfs](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing)
2) Run `git lfs install`
3) (If the files are larger than **5GB**) Run `huggingface-cli lfs-enable-largefiles .` (you will need to `pip install huggingface_hub`)

After this, you can use Git to track, commit, and push files.

## List of models

You can find a list of models in the [Hugging Face model collection](https://huggingface.co/collections/EricB/uqff-670e4a49d56ecdd3f7f0fd4c).

Have you created a UQFF model on Hugging Face? If so, please [create an issue](https://github.com/EricLBuehler/mistral.rs/issues/new).
