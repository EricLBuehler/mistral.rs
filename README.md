<h1 align="center">
  mistral.rs
</h1>

<h3 align="center">
Blazingly fast LLM inference.
</h3>

<p align="center">
| <a href="https://ericlbuehler.github.io/mistral.rs/mistralrs/"><b>Rust Documentation</b></a> | <a href="https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/API.md"><b>Python Documentation</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> | <a href="https://matrix.to/#/#mistral.rs:matrix.org"><b>Matrix</b></a> |
</p>

Please submit requests for new models [here](https://github.com/EricLBuehler/mistral.rs/issues/156).

## Get started fast 🚀

1) [Install](#installation-and-build)

2) [Get models](#getting-models)

3) Deploy with our easy to use APIs
    - [Python](examples/python)
    - [Rust](mistralrs/examples)
    - [OpenAI compatible HTTP server](docs/HTTP.md)

## Quick examples

*After following installation instructions*

- 🦙📷 Run the **Llama 3.2 Vision** Model: [documentation and guide here](docs/VLLAMA.md)

    <img src="https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg" alt="Mount Washington" width = "400" height = "267">
    <h6><a href = "https://www.nhmagazine.com/mount-washington/">Credit</a></h6>

    ```
    ./mistralrs-server -i vision-plain -m lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k -a vllama
    ```

- 🔥🧠 AnyMoE: Build a memory-efficient MoE model from anything, in seconds

    ```
    ./mistralrs-server -i toml -f toml-selectors/anymoe_lora.toml
    ```
- φ³ Run the new Phi 3.5/3.1/3 model with 128K context window

    ```
    ./mistralrs-server -i plain -m microsoft/Phi-3.5-mini-instruct -a phi3
    ```

- 🌀 Run the Phi 3.5 MoE model with 128K context window: [documentation and guide here](docs/PHI3.5MOE.md)

    ```
    ./mistralrs-server -i plain -m microsoft/Phi-3.5-MoE-instruct -a phi3.5moe
    ```

- φ³ 📷 Run the Phi 3 vision model: [documentation and guide here](docs/PHI3V.md)

    ```
    ./mistralrs-server --port 1234 vision-plain -m microsoft/Phi-3.5-vision-instruct -a phi3v
    ```

- 🌲📷 Run the FLUX.1 diffusion model: [documentation and guide here](docs/FLUX.md)

    <img src="https://github.com/user-attachments/assets/82bf5009-e3e9-402b-acf9-c48a52c7721b" width = "400" height = "267">

    ```
    ./mistralrs-server --port 1234 diffusion-plain -m black-forest-labs/FLUX.1-schnell -a flux
    ```

- Other models: [see a support matrix](#support-matrix) and [how to run them](#run-with-the-cli)

Mistal.rs supports several model categories:
- Text to Text
- Text+Image to Text: Vision (see [the docs](docs/VISION_MODELS.md))
- Text to Image: Image Generation (see [the docs](docs/IMAGEGEN_MODELS.md))

## Description
**Easy**:
- Lightweight OpenAI API compatible HTTP server
- Python API
- Grammar support with Regex and Yacc
- [ISQ](docs/ISQ.md) (In situ quantization): run `.safetensors` models directly from 🤗 Hugging Face by quantizing in-place

**Fast**:
- Apple silicon support: ARM NEON, Accelerate, Metal
- Accelerated CPU inference with MKL, AVX support
- CUDA support with flash attention and cuDNN.
- [Device mapping](docs/DEVICE_MAPPING.md): load and run some layers on the device and the rest on the CPU.

**Quantization**:
- [Details](docs/QUANTS.md)
- GGML: 2-bit, 3-bit, 4-bit, 5-bit, 6-bit and 8-bit, with ISQ support.
- GPTQ: 2-bit, 3-bit, 4-bit and 8-bit
- HQQ: 4-bit and 8 bit, with ISQ support

**Powerful**:
- LoRA support with weight merging
- First X-LoRA inference platform with first class support
- [AnyMoE](docs/ANYMOE.md): Build a memory-efficient MoE model from anything, in seconds
- Various [sampling and penalty](docs/SAMPLING.mds) methods
- Tool calling: [docs](docs/TOOL_CALLING.md)
- Prompt chunking: process large prompts in a more manageable way

**Advanced features**:
- [PagedAttention](docs/PAGED_ATTENTION.md) and continuous batching
- Prefix caching
- [Topology](docs/TOPOLOGY.md): Configure ISQ and device mapping easily
- [UQFF](docs/UQFF.md): Quantized file format for easy mixing of quants, see some [models](docs/UQFF.md#list-of-models) which have already been converted.
- Speculative Decoding: Mix supported models as the draft model or the target model
- Dynamic LoRA adapter activation with adapter preloading: [examples and docs](docs/ADAPTER_MODELS.md#adapter-model-dynamic-adapter-activation)

**Documentation for mistral.rs can be found [here](docs/README.md).**

This is a demo of interactive mode with streaming running Phi 3 128k mini with quantization via ISQ to Q4K.

<!-- Mistral GGUF demo, old API -->
<!-- https://github.com/EricLBuehler/mistral.rs/assets/65165915/3396abcd-8d44-4bf7-95e6-aa532db09415 -->

https://github.com/EricLBuehler/mistral.rs/assets/65165915/09d9a30f-1e22-4b9a-9006-4ec6ebc6473c

## Support matrix

> Note: See [supported models](#supported-models) for more information

|Model|Supports quantization|Supports adapters|Supports device mapping|Supported by AnyMoE|
|--|--|--|--|--|
|Mistral v0.1/v0.2/v0.3|✅|✅|✅|✅|
|Gemma|✅|✅|✅|✅|
|Llama 3.1/3.2|✅|✅|✅|✅|
|Mixtral|✅|✅|✅| |
|Phi 2|✅|✅|✅|✅|
|Phi 3|✅|✅|✅|✅|
|Phi 3.5 MoE|✅| |✅| |
|Qwen 2.5|✅| |✅|✅|
|Phi 3 Vision|✅| |✅|✅|
|Idefics 2|✅| |✅|✅|
|Gemma 2|✅|✅|✅|✅|
|Starcoder 2|✅|✅|✅|✅|
|LLaVa Next|✅| |✅|✅|
|LLaVa|✅| |✅|✅|
|Llama 3.2 Vision|✅| |✅| |

## APIs and Integrations

### Rust Crate

Rust multithreaded/async API for easy integration into any application.

- [Docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/)
- [Examples](mistralrs/examples/)
- To install: Add `mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git" }`

### Python API

Python API for mistral.rs.

- [Installation including PyPI](mistralrs-pyo3/README.md)
- [Docs](mistralrs-pyo3/API.md)
- [Examples](examples/python)
- [Cookbook](examples/python/cookbook.ipynb)


### HTTP Server

OpenAI API compatible API server

- [API Docs](docs/HTTP.md).
- [Running](README.md#run-with-the-cli)
- [Example](examples/server/chat.py)


### Llama Index integration (Python)

- Docs: https://docs.llamaindex.ai/en/stable/examples/llm/mistral_rs/

---

## Supported accelerators
- CUDA:
  - Compile with the `cuda` feature: `--features cuda`
  - FlashAttention support: compile with the `flash-attn` feature
  - cuDNN support: compile with the`cudnn` feature: `--features cudnn`
- Metal:
  - Compile with the `metal` feature: `--features metal`
- CPU:
  - Intel MKL: compile with the `mkl` feature: `--features mkl`
  - Apple Accelerate: compile with the `accelerate` feature: `--features accelerate`
  - ARM NEON and AVX are used automatically

Enabling features is done by passing `--features ...` to the build system. When using `cargo run` or `maturin develop`, pass the `--features` flag before the `--` separating build flags from runtime flags.

- To enable a single feature like `metal`: `cargo build --release --features metal`.
- To enable multiple features, specify them in quotes: `cargo build --release --features "cuda flash-attn cudnn"`.

## Installation and Build

> Note: You can use our [Docker containers here](https://github.com/EricLBuehler/mistral.rs/pkgs/container/mistral.rs).
> Learn more about running Docker containers: https://docs.docker.com/engine/reference/run/

> Note: You can use pre-built `mistralrs-server` binaries [here](https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.3.1)

- Install the [Python package here](mistralrs-pyo3/README.md).

1) Install required packages
    - `OpenSSL` (*Example on Ubuntu:* `sudo apt install libssl-dev`)
    - <b>*Linux only:*</b> `pkg-config` (*Example on Ubuntu:* `sudo apt install pkg-config`)

2) Install Rust: https://rustup.rs/

    *Example on Ubuntu:*
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

3) <b>*Optional:*</b> Set HF token correctly (skip if already set or your model is not gated, or if you want to use the `token_source` parameters in Python or the command line.)
    - Note: you can install `huggingface-cli` as documented [here](https://huggingface.co/docs/huggingface_hub/en/installation). 
    ```bash
    huggingface-cli login
    ```

4) Download the code
    ```bash
    git clone https://github.com/EricLBuehler/mistral.rs.git
    cd mistral.rs
    ```

5) Build or install
    - Base build command
        ```bash
        cargo build --release
        ```
    - Build with CUDA support
        ```bash
        cargo build --release --features cuda
        ```
    - Build with CUDA and Flash Attention V2 support
        ```bash
        cargo build --release --features "cuda flash-attn"
        ```
    - Build with Metal support
        ```bash
        cargo build --release --features metal
        ```
    - Build with Accelerate support
        ```bash
        cargo build --release --features accelerate
        ```
    - Build with MKL support
        ```bash
        cargo build --release --features mkl
        ```
    - Install with `cargo install` for easy command line usage

        Pass the same values to `--features` as you would for `cargo build`
        ```bash
        cargo install --path mistralrs-server --features cuda
        ```
6) The build process will output a binary `misralrs-server` at `./target/release/mistralrs-server` which may be copied into the working directory with the following command:
    
    *Example on Ubuntu:*
    ```
    cp ./target/release/mistralrs-server ./mistralrs-server
    ```

7) Use our APIs and integrations 
    
    [APIs and integrations list](#apis-and-integrations)

## Getting models

There are 2 ways to get models with mistral.rs:
- From Hugging Face Hub (easiest)
- From local files
    - Running a GGUF model
    - Specify local paths

### Getting models from Hugging Face Hub

Mistral.rs can automatically download models from HF Hub. To access gated models, you should provide a token source. They may be one of:
- `literal:<value>`: Load from a specified literal
- `env:<value>`: Load from a specified environment variable
- `path:<value>`: Load from a specified file
- `cache`: **default**: Load from the HF token at ~/.cache/huggingface/token or equivalent.
- `none`: Use no HF token

This is passed in the following ways:
- Command line:
```bash
./mistralrs-server --token-source none -i plain -m microsoft/Phi-3-mini-128k-instruct -a phi3
```
- Python:

[Here](examples/python/token_source.py) is an example of setting the token source.

If token cannot be loaded, no token will be used (i.e. effectively using `none`).

### Loading models from local files:

You can also instruct mistral.rs to load models fully locally by modifying the `*_model_id` arguments or options:
```bash
./mistralrs-server --port 1234 plain -m . -a mistral
```

Throughout mistral.rs, any model ID argument or option may be a local path and should contain the following files for each model ID option:
- `--model-id` (server) or `model_id` (python/rust) or `--tok-model-id` (server) or `tok_model_id` (python/rust): 
  - `config.json`
  - `tokenizer_config.json`
  - `tokenizer.json` (if not specified separately)
  - `.safetensors`/`.bin`/`.pth`/`.pt` files (defaults to `.safetensors`)
  - `preprocessor_config.json` (required for vision models).
  - `processor_config.json` (optional for vision models).
- `--quantized-model-id` (server) or `quantized_model_id` (python/rust):
  - Specified `.gguf` or `.ggml` file.
- `--x-lora-model-id` (server) or `xlora_model_id` (python/rust):
  - `xlora_classifier.safetensors`
  - `xlora_config.json`
  - Adapters `.safetensors` and `adapter_config.json` files in their respective directories
- `--adapters-model-id` (server) or `adapters_model_id` (python/rust):
  - Adapters `.safetensors` and `adapter_config.json` files in their respective directories

### Running GGUF models

To run GGUF models, the only mandatory arguments are the quantized model ID and the quantized filename. The quantized model ID can be a HF model ID.

GGUF models contain a tokenizer. However, mistral.rs allows you to run the model with a tokenizer from a specified model, typically the official one. This means there are two options:
1) [With a specified tokenizer](#with-a-specified-tokenizer)
1) [With the builtin tokenizer](#with-the-builtin-tokenizer)

#### With a specified tokenizer

Running with a tokenizer model ID enables you to specify the model ID to source the tokenizer from:

```bash
./mistralrs-server gguf -m bartowski/Phi-3.5-mini-instruct-GGUF -f Phi-3.5-mini-instruct-Q4_K_M.gguf -t microsoft/Phi-3.5-mini-instruct
```

If the specified tokenizer model ID contains a `tokenizer.json`, then it will be used over the GGUF tokenizer.

#### With the builtin tokenizer

Using the builtin tokenizer:

```bash
./mistralrs-server gguf -m bartowski/Phi-3.5-mini-instruct-GGUF -f Phi-3.5-mini-instruct-Q4_K_M.gguf
```

(or using a local file):

```bash
./mistralrs-server gguf -m path/to/files -f Phi-3.5-mini-instruct-Q4_K_M.gguf
```

There are a few more ways to configure:

**Chat template:**

The chat template can be automatically detected and loaded from the GGUF file if no other chat template source is specified including the tokenizer model ID.

If that does not work, you can either [provide a tokenizer](#with-a-specified-tokenizer) (recommended), or specify a custom chat template.

```bash
./mistralrs-server --chat-template <chat_template> gguf -m . -f Phi-3.5-mini-instruct-Q4_K_M.gguf
```

**Tokenizer**

The following tokenizer model types are currently supported. If you would like one to be added, please raise an issue. Otherwise,
please consider using the method demonstrated in examples below, where the tokenizer is sourced from Hugging Face.

**Supported GGUF tokenizer types**
- `llama` (sentencepiece)
- `gpt2` (BPE)

## Run with the CLI

Mistral.rs uses subcommands to control the model type. They are generally of format `<XLORA/LORA>-<QUANTIZATION>`. Please run `./mistralrs-server --help` to see the subcommands.

Additionally, for models without quantization, the model architecture should be provided as the `--arch` or `-a` argument in contrast to GGUF models which encode the architecture in the file. 

### Architecture for plain models

> Note: for plain models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device. This is specified in the `--dype`/`-d` parameter after the model architecture (`plain`).

If you do not specify the architecture, an attempt will be made to use the model's config. If this fails, please raise an issue.

- `mistral`
- `gemma`
- `mixtral`
- `llama`
- `phi2`
- `phi3`
- `phi3.5moe`
- `qwen2`
- `gemma2`
- `starcoder2`

### Architecture for vision models

> Note: for vision models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device. This is specified in the `--dype`/`-d` parameter after the model architecture (`vision-plain`).

- `phi3v`
- `idefics2`
- `llava_next`
- `llava`
- `vllama`

### Supported GGUF architectures

**Plain:**

- `llama`
- `phi2`
- `phi3`
- `starcoder2`

**With adapters:**

- `llama`
- `phi3`

### Interactive mode

You can launch interactive mode, a simple chat application running in the terminal, by passing `-i`:

```bash
./mistralrs-server -i plain -m microsoft/Phi-3-mini-128k-instruct -a phi3
```

Vision models work too:

```bash
./mistralrs-server -i vision-plain -m lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k -a vllama
```

And even diffusion models:

```bash
./mistralrs-server -i diffusion-plain -m black-forest-labs/FLUX.1-schnell -a flux
```

### OpenAI HTTP server

You can an HTTP server

```bash
./mistralrs-server --port 1234 plain -m microsoft/Phi-3.5-MoE-instruct -a phi3.5moe
```

### Structured selection with a `.toml` file

We provide a method to select models with a `.toml` file. The keys are the same as the command line, with `no_kv_cache` and `tokenizer_json` being "global" keys.

Example:
```bash
./mistralrs-server --port 1234 toml -f toml-selectors/gguf.toml
```

---

## Benchmarks
|Device|Mistral.rs Completion T/s|Llama.cpp Completion T/s|Model|Quant|
|-|-|-|-|-|
|A10 GPU, CUDA|86|83|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|4_K_M|
|Intel Xeon 8358 CPU, AVX|11|23|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|4_K_M|
|Raspberry Pi 5 (8GB), Neon|2|3|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|2_K|
|A100 GPU, CUDA|131|134|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|4_K_M|
|RTX 6000 GPU, CUDA|103|96|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|4_K_M|

> Note: All CUDA tests for mistral.rs conducted with PagedAttention enabled, block size = 32

Please submit more benchmarks via raising an issue!

## Supported models

**Quantization support**
|Model|GGUF|GGML|ISQ|
|--|--|--|--|
|Mistral|✅| |✅|
|Gemma| | |✅|
|Llama|✅|✅|✅|
|Mixtral|✅| |✅|
|Phi 2|✅| |✅|
|Phi 3|✅| |✅|
|Phi 3.5 MoE| | |✅|
|Qwen 2.5| | |✅|
|Phi 3 Vision| | |✅|
|Idefics 2| | |✅|
|Gemma 2| | |✅|
|Starcoder 2| |✅|✅|
|LLaVa Next| | |✅|
|LLaVa| | |✅|
|Llama 3.2 Vision| | |✅|

**Device mapping support**
|Model category|Supported|
|--|--|
|Plain|✅|
|GGUF|✅|
|GGML| |
|Vision Plain|✅|

**X-LoRA and LoRA support**
|Model|X-LoRA|X-LoRA+GGUF|X-LoRA+GGML|
|--|--|--|--|
|Mistral|✅|✅| |
|Gemma|✅| | |
|Llama|✅|✅|✅|
|Mixtral|✅|✅| |
|Phi 2|✅| | |
|Phi 3|✅|✅| |
|Phi 3.5 MoE| | | |
|Qwen 2.5| | | |
|Phi 3 Vision| | | |
|Idefics 2| | | |
|Gemma 2|✅| | |
|Starcoder 2|✅| | |
|LLaVa Next| | | |
|LLaVa| | | |
|Llama 3.2 Vision| | | |

**AnyMoE support**
|Model|AnyMoE|
|--|--|
|Mistral 7B|✅|
|Gemma|✅|
|Llama|✅|
|Mixtral| |
|Phi 2|✅|
|Phi 3|✅|
|Phi 3.5 MoE| |
|Qwen 2.5|✅|
|Phi 3 Vision| |
|Idefics 2| |
|Gemma 2|✅|
|Starcoder 2|✅|
|LLaVa Next|✅|
|LLaVa|✅|
|Llama 3.2 Vision| |


### Using derivative model

To use a derivative model, select the model architecture using the correct subcommand. To see what can be passed for the architecture, pass `--help` after the subcommand. For example, when using a different model than the default, specify the following for the following types of models:

- **Plain**: Model id
- **Quantized**: Quantized model id, quantized filename, and tokenizer id
- **X-LoRA**: Model id, X-LoRA ordering
- **X-LoRA quantized**: Quantized model id, quantized filename, tokenizer id, and X-LoRA ordering
- **LoRA**: Model id, LoRA ordering
- **LoRA quantized**: Quantized model id, quantized filename, tokenizer id, and LoRA ordering
- **Vision Plain**: Model id

See [this](#adapter-ordering-file) section to determine if it is necessary to prepare an X-LoRA/LoRA ordering file, it is always necessary if the target modules or architecture changed, or if the adapter order changed.

It is also important to check the chat template style of the model. If the HF hub repo has a `tokenizer_config.json` file, it is not necessary to specify. Otherwise, templates can be found in `chat_templates` and should be passed before the subcommand. If the model is not instruction tuned, no chat template will be found and the APIs will only accept a prompt, no messages.

For example, when using a Zephyr model:

`./mistralrs-server --port 1234 --log output.txt gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf`

### Adapter model support: X-LoRA and LoRA

An adapter model is a model with X-LoRA or LoRA. X-LoRA support is provided by selecting the `x-lora-*` architecture, and LoRA support by selecting the `lora-*` architecture. Please find docs for adapter models [here](docs/ADAPTER_MODELS.md). Examples may be found [here](docs/LORA_XLORA.md).

### Chat Templates and Tokenizer
Mistral.rs will attempt to automatically load a chat template and tokenizer. This enables high flexibility across models and ensures accurate and flexible chat templating. However, this behavior can be customized. Please find detailed documentation [here](docs/CHAT_TOK.md).

## Contributing

Thank you for contributing! If you have any problems or want to contribute something, please raise an issue or pull request.
If you want to add a new model, please contact us via an issue and we can coordinate how to do this.

## FAQ
- Debugging with the environment variable `MISTRALRS_DEBUG=1` causes the following things
    - If loading a GGUF or GGML model, this will output a file containing the names, shapes, and types of each tensor.
        - `mistralrs_gguf_tensors.txt` or `mistralrs_ggml_tensors.txt`
    - More logging.
- Setting the CUDA compiler path:
    - Set the `NVCC_CCBIN` environment variable during build.
- Error: `recompile with -fPIE`:
    - Some Linux distributions require compiling with `-fPIE`.
    - Set the `CUDA_NVCC_FLAGS` environment variable to `-fPIE` during build: `CUDA_NVCC_FLAGS=-fPIE`
- Error `CUDA_ERROR_NOT_FOUND` or symbol not found when using a normal or vison model:
    - For non-quantized models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device.

## Credits
This project would not be possible without the excellent work at [`candle`](https://github.com/huggingface/candle). Additionally, thank you to all contributors! Contributing can range from raising an issue or suggesting a feature to adding some new functionality.
