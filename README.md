<a name="top"></a>
<h1 align="center">
  mistral.rs
</h1>

<h3 align="center">
Blazingly fast LLM inference.
</h3>

<p align="center">
| <a href="https://ericlbuehler.github.io/mistral.rs/mistralrs/"><b>Rust Documentation</b></a> | <a href="https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/API.md"><b>Python Documentation</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> | <a href="https://matrix.to/#/#mistral.rs:matrix.org"><b>Matrix</b></a> |
</p>

Mistral.rs is a cross-platform, highly multimodal inference engine featuring support for **text**, **vision**, **image generation**, and **speech generation** models!

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

Minimal Llama chat example:
```
./mistralrs-server -i plain -m meta-llama/Llama-3.2-3B-Instruct
```

## Description
**Easy**:
- Lightweight OpenAI API compatible HTTP server
- Python API
- Grammar support with JSON Schema, Regex, Lark, and Guidance via [LLGuidance library](https://github.com/microsoft/llguidance)
- [ISQ](docs/ISQ.md) (In situ quantization): run `.safetensors` models directly from 🤗 Hugging Face by quantizing in-place
    - Enhance performance with an [imatrix](docs/IMATRIX.md)!
- Automatic [device mapping](docs/DEVICE_MAPPING.md) to easily load and run models across multiple GPUs and CPU.
- Specify custom chat templates easily: [chat templates](docs/CHAT_TOK.md)

**Fast**:
- Apple silicon support: ARM NEON, Accelerate, Metal
- Accelerated CPU inference with MKL, AVX support
- CUDA support with FlashAttention and cuDNN.
- Automatic tensor-parallelism support with NCCL: [distributed documentation](docs/DISTRIBUTED.md)

**Quantization**:
- [Details](docs/QUANTS.md)
- GGML: 2-bit, 3-bit, 4-bit, 5-bit, 6-bit and 8-bit, with imatrix support
- GPTQ: 2-bit, 3-bit, 4-bit and 8-bit, with [Marlin](https://github.com/IST-DASLab/marlin) kernel support in 4-bit and 8-bit.
- AWQ: 4-bit and 8-bit (convert using [script](scripts/convert_awq_marlin.py))
- AFQ: 🔥 2-bit, 3-bit, 4-bit, 6-bit and 8-bit, designed to be fast on Metal!
- HQQ: 4-bit and 8 bit, with ISQ support
- FP8
- BNB: bitsandbytes int8, fp4, nf4 support
- Easily run MLX prequantized models
- Automatic ISQ to select the fastest and most accurate quantization method.

**Powerful**:
- LoRA support with weight merging
- First X-LoRA inference platform with first class support
- [AnyMoE](docs/ANYMOE.md): Build a memory-efficient MoE model from anything, in seconds
- Various [sampling and penalty](docs/SAMPLING.mds) methods
- Native tool calling support for Llama, Mistral Small, Mistral Nemo, Hermes, and DeepSeek models: [docs](docs/TOOL_CALLING.md)
- Prompt chunking: process large prompts in a more manageable way

**Advanced features**:
- [PagedAttention](docs/PAGED_ATTENTION.md) and continuous batching (CUDA and Metal support)
- [FlashAttention](docs/FLASH_ATTENTION.md) V2/V3
- Prefix caching, including support for multimodal prefix caching
- [Topology](docs/TOPOLOGY.md): Configure ISQ and device mapping easily
- [UQFF](docs/UQFF.md): Quantized file format for easy mixing of quants, [collection here](https://huggingface.co/collections/EricB/uqff-670e4a49d56ecdd3f7f0fd4c).
- Speculative Decoding: Mix supported models as the draft model or the target model
- Dynamic LoRA adapter activation with adapter preloading: [examples and docs](docs/ADAPTER_MODELS.md#adapter-model-dynamic-adapter-activation)
- Integrated agentic web search capabilities, enabling models to easily access the internet.

**Documentation for mistral.rs can be found [here](docs/README.md).**

This is a demo of interactive mode with streaming running Phi 3 128k mini with quantization via ISQ to Q4K.

<!-- Mistral GGUF demo, old API -->
<!-- https://github.com/EricLBuehler/mistral.rs/assets/65165915/3396abcd-8d44-4bf7-95e6-aa532db09415 -->

https://github.com/EricLBuehler/mistral.rs/assets/65165915/09d9a30f-1e22-4b9a-9006-4ec6ebc6473c

<details>
<summary>Show architecture support matrix</summary>

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
|Qwen2-VL|✅| |✅| |
|Idefics 3|✅| |✅|✅|
|DeepseekV2|✅| |✅| |
|DeepseekV3|✅| |✅| |
|MinCPM-O 2.6|✅| |✅| |
|Phi 4 Multimodal|✅| |✅| |
|Qwen2.5-VL|✅| |✅| |
|Gemma 3|✅| |✅|✅|
|Mistral 3|✅| |✅|✅|
|Llama 4|✅| |✅| |
|Qwen 3|✅| |✅| |
</details>

## APIs and Integrations

### Rust Crate

Rust multithreaded/async API for easy integration into any application.

- [Docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/)
- [Examples](mistralrs/examples/)
- To install: Add `mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git" }`

### Python API

Python API for mistral.rs.

- [Installation including PyPI](mistralrs-pyo3/_README.md)
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
- NVIDIA GPUs (CUDA):
  - Compile with the `cuda` feature: `--features cuda`
  - FlashAttention support: compile with the `flash-attn` feature
  - cuDNN support: compile with the`cudnn` feature: `--features cudnn`
- Apple Silicon GPU (Metal):
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

- Install the [Python package here](mistralrs-pyo3/_README.md).
- The Python package has [wheels on PyPi](mistralrs-pyo3/_README.md#installation-from-pypi)!

1) Install required packages:
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

4) Download the code:
    ```bash
    git clone https://github.com/EricLBuehler/mistral.rs.git
    cd mistral.rs
    ```

5) Build or install:
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
6) The build process will output a binary `mistralrs-server` at `./target/release/mistralrs-server`. We can switch to that directory so that the binary can be accessed as `./mistralrs-server` with the following command:

    *Example on Ubuntu:*
    ```
    cd target/release
    ```

7) Use our APIs and integrations: 
    
    [APIs and integrations list](#apis-and-integrations)

## Getting models
<details>
<summary>Show: How to get models (Hub, local, GGUF, adapters, etc.)</summary>

### Getting models from Hugging Face Hub
- **Default:** Downloads from Hugging Face Hub.
- For gated models, you can optionally set token source:
    - CLI: `./mistralrs-server --token-source env:HF_TOKEN ...`
    - Python: See [examples/python/token_source.py](examples/python/token_source.py)
    - If no token is found, tries `~/.cache/huggingface/token` or runs with no token.

### Loading models from local files
- Pass a path to a downloaded model from Hugging Face hub:
    - Example:  
      ```
      ./mistralrs-server -i plain -m path/to/model
      ```

### Running GGUF models
- Minimal example:
  ```
  ./mistralrs-server gguf -m author/model-repo -f model-quant.gguf
  ```
- Specify tokenizer (if needed):
  ```
  ./mistralrs-server gguf -m author/model-repo -f file.gguf -t author/official-tokenizer
  ```
  (Or use the built-in GGUF tokenizer.)

### Adapters, X-LoRA, LoRA, Chat Templates
- Use the correct subcommand (`x-lora-*`, `lora-*`), pass model, adapter, or quant file as needed.
- See [docs/ADAPTER_MODELS.md](docs/ADAPTER_MODELS.md) for details.
- For chat templates: usually auto-detected, override with `--chat-template <file>`.  
  See [docs/CHAT_TOK.md](docs/CHAT_TOK.md).

### More model CLI examples
- See [Run with the CLI](#run-with-the-cli) below or [full documentation](docs/README.md).

</details>

## Run with the CLI

Mistral.rs uses subcommands to control the model type. Please run `./mistralrs-server --help` to see the subcommands which categorize the models by kind.

### Architecture for plain models

> Note: for plain models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device. This is specified in the `--dype`/`-d` parameter after the model architecture (`plain`). For quantized models (gguf/ggml), you may specify data type of `f32` or `bf16` (`f16` is not recommended due to its lower precision in quantized inference).

If you do not specify the architecture, an attempt will be made to use the model's config. If this fails, please raise an issue.

<details>
  <summary>Show plain architectures</summary>

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
- `deepseekv2`
- `deepseekv3`
- `qwen3`
- `qwen3moe`

</details>

### Architecture for vision models

> Note: for vision models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device. This is specified in the `--dype`/`-d` parameter after the model architecture (`vision-plain`).

<details>
  <summary>Show vision architectures</summary>

- `phi3v`
- `idefics2`
- `llava_next`
- `llava`
- `vllama`
- `qwen2vl`
- `idefics3`
- `minicpmo`
- `phi4mm`
- `qwen2_5vl`
- `gemma3`
- `mistral3`
- `llama4`

</details>

### Supported GGUF architectures

<details>
  <summary>Show supported GGUF architectures</summary>

**Plain:**
- llama
- phi2
- phi3
- starcoder2
- qwen2

**With adapters:**
- llama
- phi3

</details>

### Interactive mode

You can launch interactive mode, a simple chat application running in the terminal, by passing `-i`:

```bash
./mistralrs-server -i plain -m meta-llama/Llama-3.2-3B-Instruct
```

Vision models work seamlessly:

```bash
./mistralrs-server -i vision-plain -m lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k
```

Diffusion models can be run too:

```bash
./mistralrs-server -i diffusion-plain -m black-forest-labs/FLUX.1-schnell -a flux
```

And you can run speech generation in your terminal!

```bash
./mistralrs-server -i speech -m nari-labs/Dia-1.6B -a dia
```

### OpenAI HTTP server

You can an HTTP server

```bash
./mistralrs-server --port 1234 plain -m microsoft/Phi-3.5-MoE-instruct
```

### Structured selection with a `.toml` file

We provide a method to select models with a `.toml` file. The keys are the same as the command line, with `no_kv_cache` and `tokenizer_json` being "global" keys.

Example:
```bash
./mistralrs-server --port 1234 toml -f toml-selectors/gguf.toml
```

---

Please submit more benchmarks via raising an issue!

## Supported models

<details>
<summary>Show quantization support</summary>

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
|Qwen2-VL| | |✅|
|Idefics 3| | |✅|
|Deepseek V2| | |✅|
|Deepseek V3| | |✅|
|MiniCPM-O 2.6| | |✅|
|Qwen2.5-VL| | |✅|
|Gemma 3| | |✅|
|Mistral 3| | |✅|
|Llama 4| | |✅|
|Qwen 3| | |✅|
|Dia 1.6b| | |✅|
</details>

<details>
<summary>Show device mapping support</summary>

**Device mapping support**
|Model category|Supported|
|--|--|
|Plain|✅|
|GGUF|✅|
|GGML| |
|Vision Plain|✅|
</details>

<details>
<summary>Show X-LoRA and LoRA support</summary>

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
|Qwen2-VL| | | |
|Idefics 3| | | |
|Deepseek V2| | | |
|Deepseek V3| | | |
|MiniCPM-O 2.6| | | |
|Qwen2.5-VL| | | |
|Gemma 3| | | |
|Mistral 3| | | |
|Llama 4| | | |
|Qwen 3| | | |
</details>

<details>
<summary>Show AnyMoE support</summary>

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
|Qwen2-VL| |
|Idefics 3|✅|
|Deepseek V2| |
|Deepseek V3| |
|MiniCPM-O 2.6| |
|Qwen2.5-VL| |
|Gemma 3|✅|
|Mistral 3|✅|
|Llama 4| |
|Qwen 3| |
</details>

### Using derivative and adapter models

To use a derivative or adapter model (e.g., quantized, LoRA, X-LoRA, vision, etc.), select the correct architecture subcommand and pass the required arguments—typically model id, and for quantized/adapters, also the quantization filename, tokenizer, or adapter ordering if needed.

- **See all options:** Run `./mistralrs-server <subcommand> --help`  
- **Docs:** [Adapter models](docs/ADAPTER_MODELS.md), [Chat templates](docs/CHAT_TOK.md)

<details>
<summary>Arguments by model type</summary>

| Model Type          | Required Arguments                                                     |
|---------------------|-----------------------------------------------------------------------|
| Plain               | model id                                                              |
| Quantized           | model id, quantized filename, tokenizer id                            |
| X-LoRA              | model id, X-LoRA ordering (if not default)                            |
| X-LoRA quantized    | model id, quantized filename, tokenizer id, X-LoRA ordering           |
| LoRA                | model id, LoRA ordering (if not default)                              |
| LoRA quantized      | model id, quantized filename, tokenizer id, LoRA ordering             |
| Vision Plain        | model id                                                              |

</details>

<details>
<summary>Example: Zephyr GGUF model</summary>

```bash
./mistralrs-server --port 1234 --log output.txt gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf
```
</details>

Chat template and tokenizer are usually auto-detected.  
If you need to override, see the [chat templates doc](docs/CHAT_TOK.md).

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
- What is the minimum supported CUDA compute cap?
    - The minimum CUDA compute cap is **5.3**.
- Metal not found (error: unable to find utility "metal", not a developer tool or in PATH)
    1) Install Xcode: `xcode-select --install`
    2) Set the active developer directory: `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`
  
## Credits
This project would not be possible without the excellent work at [`candle`](https://github.com/huggingface/candle). Additionally, thank you to all contributors! Contributing can range from raising an issue or suggesting a feature to adding some new functionality.

<p align="right">
  <a href="#top">⬆️ Back to Top</a>
</p>
