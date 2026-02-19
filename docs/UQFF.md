# Universal Quantized File Format: UQFF

<h3 align="left">
The uniquely powerful quantized file format.
</h3>

1) **Flexible** 🌀: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Reliable** 🔒: Compatibility ensured with *embedded* and *checked* semantic versioning information from day 1.
3) **Easy** 🤗: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
3) **Customizable** 🛠️: Make and publish your own UQFF files in minutes.

# ToC
- [Universal Quantized File Format: UQFF](#universal-quantized-file-format-uqff)
- [ToC](#toc)
  - [Motivation](#motivation)
  - [Support](#support)
  - [Loading a UQFF model](#loading-a-uqff-model)
    - [Shard auto-discovery](#shard-auto-discovery)
    - [Running with the CLI](#running-with-the-cli)
    - [Using with the Rust SDK](#using-with-the-rust-sdk)
    - [Using the Python SDK](#using-the-python-sdk)
    - [Using topology for device mapping with UQFF](#using-topology-for-device-mapping-with-uqff)
  - [Creating a UQFF model](#creating-a-uqff-model)
    - [Model card generation](#model-card-generation)
    - [Uploading to Hugging Face](#uploading-to-hugging-face)
  - [List of models](#list-of-models)

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

- AFQ quantized (🔥 AFQ is fast on **Metal**):
    - AFQ2
    - AFQ3
    - AFQ4
    - AFQ6
    - AFQ8

- F8Q8:
    - F8Q8

## Loading a UQFF model

To load a UQFF model, specify the filename of the first (or only) UQFF shard. This will be located based on the model ID, and can
be loaded locally or from Hugging Face based on the model ID.

- `phi3.5-mini-instruct-q4k-0.uqff`
- `../UQFF/phi3.5-mini-instruct-q4k-0.uqff`

You can find a [collection of UQFF models here](https://huggingface.co/collections/EricB/uqff-670e4a49d56ecdd3f7f0fd4c), which each include a simple
command to get started.

> Note: when loading an UQFF model, *any* ISQ setting will be ignored.

### Shard auto-discovery

Large models produce multiple shard files (e.g., `q4k-0.uqff`, `q4k-1.uqff`, `q4k-2.uqff`). You only need to specify **one** shard file -- the remaining shards are auto-discovered from the same directory or Hugging Face repository.

For example, if a model has shards `q4k-0.uqff`, `q4k-1.uqff`, and `q4k-2.uqff`:
```bash
# Just specify the first shard -- the rest are found automatically
mistralrs run -m EricB/MyModel-UQFF --from-uqff q4k-0.uqff
```

This also works when multiple quantizations exist in the same repo (e.g., `q4k-*` and `q8_0-*`). Only the shards matching the specified prefix are loaded.

### Running with the CLI

```bash
mistralrs run -m EricB/Phi-3.5-mini-instruct-UQFF --from-uqff phi3.5-mini-instruct-f8e4m3-0.uqff
```

### Using with the Rust SDK

Check out the following examples:
- Normal: [uqff/main.rs](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/uqff/main.rs)
- Vision: [uqff_vision/main.rs](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/uqff_vision/main.rs)

### Using the Python SDK
Modify the `Which` instantiation as follows:
```diff
Which.Plain(
    model_id="EricB/Phi-3.5-mini-instruct-UQFF",
+   from_uqff="phi3.5-mini-instruct-q4k-0.uqff"
),
```

### Using topology for device mapping with UQFF

When loading a UQFF model, the quantization is already baked in, so ISQ settings in the topology are ignored. However, **device mapping** from a topology file still applies. This is useful for splitting a pre-quantized model across multiple GPUs or offloading layers to CPU.

**CLI example:**
```bash
mistralrs run -m EricB/Phi-3.5-mini-instruct-UQFF --from-uqff phi3.5-mini-instruct-q4k.uqff --topology device_map.yml
```

**Topology file for device mapping only (`device_map.yml`):**
```yaml
0-16:
  device: cuda[0]
16-32:
  device: cuda[1]
```

**Rust SDK example:**
```rust
use mistralrs::{UqffTextModelBuilder, Topology, LayerTopology, Device};

let model = UqffTextModelBuilder::new(
    "EricB/Phi-3.5-mini-instruct-UQFF",
    vec!["phi3.5-mini-instruct-q4k.uqff".into()],
)
.into_inner()
.with_topology(
    Topology::empty()
        .with_range(0..16, LayerTopology { isq: None, device: Some(Device::Cuda(0)) })
        .with_range(16..32, LayerTopology { isq: None, device: Some(Device::Cuda(1)) })
)
.build()
.await?;
```

**Python SDK example:**
```python
runner = Runner(
    which=Which.Plain(
        model_id="EricB/Phi-3.5-mini-instruct-UQFF",
        from_uqff="phi3.5-mini-instruct-q4k.uqff",
        topology="device_map.yml",
    ),
)
```

> Note: The `isq` field in topology entries is ignored when loading UQFF models since quantization is pre-applied.

## Creating a UQFF model

Creating a UQFF model requires you to generate the UQFF file.
- Specify an output path: either a `.uqff` file path or a directory where files will be auto-named.
- The quantization of a UQFF model is determined from the ISQ or model topology (see the [topology docs](TOPOLOGY.md) for more details on how ISQ and the topology mix).

Along with the UQFF file, the generation process will also output several `.json` configuration files and `residual.safetensors`. All of these files are considered the
UQFF model, and should be kept together or uploaded.

> Note: Only the `.uqff` files are unique to the quantization level(s). If you are generating multiple UQFF files, it is OK for the others to be overwritten.

**Single quantization (file output):**
```bash
mistralrs quantize -m microsoft/Phi-3.5-mini-instruct --isq q4k -o phi3.5-uqff/phi3.5-mini-instruct-q4k.uqff
```

**Single quantization (directory output):**
```bash
mistralrs quantize -m microsoft/Phi-3.5-mini-instruct --isq q4k -o phi3.5-uqff/
```

**Multiple quantizations at once (directory output):**

Generate multiple UQFF files by specifying multiple `--isq` types. All quantizations go to the same output directory.

```bash
# Comma-separated ISQ types
mistralrs quantize -m microsoft/Phi-3.5-mini-instruct --isq q4k,q8_0 -o phi3.5-uqff/

# Equivalent: repeated --isq flags
mistralrs quantize -m microsoft/Phi-3.5-mini-instruct --isq q4k --isq q8_0 -o phi3.5-uqff/
```

This produces the following in `phi3.5-uqff/`:
- `q4k-0.uqff` (and additional shards `q4k-1.uqff`, ... if the model is large)
- `q8_0-0.uqff` (and additional shards if needed)
- `README.md` (auto-generated model card for Hugging Face)
- Shared files: `config.json`, `tokenizer.json`, `residual.safetensors`, etc.

> Note: Multiple `--isq` values require a directory output path (not a `.uqff` file path).

### Model card generation

When using directory output mode, the `quantize` command automatically generates a `README.md` model card in the output directory. This model card includes Hugging Face YAML frontmatter, a description, and an examples table with the appropriate `--from-uqff` commands for each quantization.

To skip model card generation, use `--no-readme`:
```bash
mistralrs quantize -m microsoft/Phi-3.5-mini-instruct --isq q4k -o phi3.5-uqff/ --no-readme
```

### Uploading to Hugging Face

After quantization completes in directory mode, the `quantize` command prints the `huggingface-cli` upload command you can use. The general form is:

```bash
huggingface-cli upload <YOUR_USERNAME>/<MODEL_NAME>-UQFF <output_dir> --repo-type model --private
```

Alternatively, you can upload with Git LFS:

1) Install [git-lfs](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing)
2) Run `git lfs install`
3) (If the files are larger than **5GB**) Run `huggingface-cli lfs-enable-largefiles .` (you will need to `pip install huggingface_hub`)

After this, you can use Git to track, commit, and push files.

## List of models

You can find a list of models in the [Hugging Face model collection](https://huggingface.co/collections/EricB/uqff-670e4a49d56ecdd3f7f0fd4c).

Have you created a UQFF model on Hugging Face? If so, please [create an issue](https://github.com/EricLBuehler/mistral.rs/issues/new).
