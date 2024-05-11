<h1 align="center">
  mistral.rs
</h1>

<h3 align="center">
Blazingly fast LLM inference.
</h3>

<p align="center">
| <a href="https://ericlbuehler.github.io/mistral.rs/mistralrs/"><b>Rust Documentation</b></a> | <a href="https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/API.md"><b>Python Documentation</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> |

</p>

Mistral.rs is a fast LLM inference platform supporting inference on a variety of devices, quantization, and easy-to-use application with an Open-AI API compatible HTTP server and Python bindings. 

## Upcoming features
- More models: please submit requests [here](https://github.com/EricLBuehler/mistral.rs/issues/156).
- X-LoRA: Scalings `topk` and softmax `topk` ([#48](https://github.com/EricLBuehler/mistral.rs/issues/48)).
- Parallel linear layers (sharding) ([#50](https://github.com/EricLBuehler/mistral.rs/issues/50)).
- Speculative decoding: https://arxiv.org/pdf/2211.17192

**Running the new Llama 3 model**

`cargo run --release --features ... -- -i plain -m meta-llama/Meta-Llama-3-8B-Instruct -a llama`

**Running the new Phi 3 model with 128K context window**

`cargo run --release --features ... -- -i plain -m microsoft/Phi-3-mini-128k-instruct -a phi3`

## Description
**Fast**:
- Quantized model support: 2-bit, 3-bit, 4-bit, 5-bit, 6-bit and 8-bit for faster inference and optimized memory usage.
- Continuous batching.
- Prefix caching.
- Device mapping: load and run some layers on the device and the rest on the CPU.

**Accelerator support**:
- Apple silicon support with the Metal framework.
- CPU inference with `mkl`, `accelerate` support and optimized backend.
- CUDA support with flash attention and cuDNN.

**Easy**:
- Lightweight OpenAI API compatible HTTP server.
- Python API.
- Grammar support with Regex and Yacc.
- [ISQ](docs/ISQ.md) (In situ quantization): run `.safetensors` models directly from Hugging Face Hub by quantizing them after loading instead of creating a GGUF file. This loads the ISQ-able weights on CPU before quantizing with ISQ and then moving back to the device to avoid memory spikes.

**Powerful**:
- Fast LoRA support with weight merging.
- First X-LoRA inference platform with first class support.
- Speculative Decoding: Mix supported models as the draft model or the target model
- Dynamic LoRA adapter swapping at runtime with adapter preloading: [examples and docs](docs/ADAPTER_MODELS.md#adapter-model-dynamic-adapter-activation)


This is a demo of interactive mode with streaming running Mistral GGUF:

https://github.com/EricLBuehler/mistral.rs/assets/65165915/3396abcd-8d44-4bf7-95e6-aa532db09415


**Supported models:**
- Mistral 7B (v0.1 and v0.2)
- Gemma
- Llama, including Llama 3
- Mixtral 8x7B
- Phi 2
- Phi 3
- Qwen 2

Please see [this section](#supported-models) for details on quantization and LoRA support.

## APIs and Integrations
**Rust Library API**

Rust multithreaded API for easy integration into any application.

- [Docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/)
- [Examples](mistralrs/examples/)
- To install: Add `mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git" }`

**Python API**

Python API for mistral.rs.

- [Installation](mistralrs-pyo3/README.md)
- [Docs](mistralrs-pyo3/API.md)
- [Example](examples/python/python_api.py)
- [Cookbook](examples/python/cookbook.ipynb)

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
    )
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[{"role":"user", "content":"Tell me a story about the Rust type system."}],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

**HTTP Server**

OpenAI API compatible API server

- [API Docs](examples/http.md).
- [Running](README.md#run)
- [Example](examples/server/chat.py)

**Llama Index integration**

- Docs: https://docs.llamaindex.ai/en/stable/examples/llm/mistral_rs/

---

## Supported accelerators
- CUDA:
  - Enable with `cuda` feature: `--features cuda`
  - Flash attention support with `flash-attn` feature, only applicable to non-quantized models: `--features flash-attn`
  - cuDNNsupport with `cudnn` feature: `--features cudnn`
- Metal:
  - Enable with `metal` feature: `--features metal`
- CPU:
  - Intel MKL with `mkl` feature: `--features mkl`
  - Apple Accelerate with `accelerate` feature: `--features accelerate`

Enabling features is done by passing `--features ...` to the build system. When using `cargo run` or `maturin develop`, pass the `--features` flag before the `--` separating build flags from runtime flags.

- To enable a single feature like `metal`: `cargo build --release --features metal`.
- To enable multiple features, specify them in quotes: `cargo build --release --features "cuda flash-attn cudnn"`.

## Benchmarks
|Device|Mistral.rs Completion T/s|Llama.cpp Completion T/s|Model|Quant|
|-|-|-|-|-|
|A10 GPU, CUDA|78|78|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|4_K_M|
|Intel Xeon 8358 CPU, AVX|6|19|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|4_K_M|
|Raspberry Pi 5 (8GB), Neon|2|3|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|2_K|
|A100 GPU, CUDA|110|119|[mistral-7b](TheBloke/Mistral-7B-Instruct-v0.1-GGUF)|4_K_M|

Please submit more benchmarks via raising an issue!

## Usage
### Installation and Build
To install mistral.rs, one should ensure they have Rust installed by following [this](https://rustup.rs/) link. Additionally, the Hugging Face token should be provided in `~/.cache/huggingface/token` when using the server to enable automatic download of gated models.

1) Install required packages
    - `openssl` (ex., `sudo apt install libssl-dev`)
    - `pkg-config` (ex., `sudo apt install pkg-config`)

2) Install Rust: https://rustup.rs/
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

3) Set HF token correctly (skip if already set or your model is not gated, or if you want to use the `token_source` parameters in Python or the command line.)
    ```bash
    mkdir ~/.cache/huggingface
    touch ~/.cache/huggingface/token
    echo <HF_TOKEN_HERE> > ~/.cache/huggingface/token
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
    ```
    cp ./target/release/mistralrs-server ./mistralrs_server
    ```

7) Installing Python support

    You can install Python support by following the guide [here](/mistralrs-pyo3/README.md).

### Getting models
**Loading from HF Hub:**

Mistral.rs can automatically download models from HF Hub. To access gated models, you should provide a token source. They may be one of:
- `literal:<value>`: Load from a specified literal
- `env:<value>`: Load from a specified environment variable
- `path:<value>`: Load from a specified file
- `cache`: **default**: Load from the HF token at ~/.cache/huggingface/token or equivalent.
- `none`: Use no HF token

This is passed in the following ways:
- Command line:
```bash
./mistralrs_server --token-source none -i plain -m microsoft/Phi-3-mini-128k-instruct -a phi3
```
- Python:

[Here](examples/python/token_source.py) is an example of setting the token source.

If token cannot be loaded, no token will be used (i.e. effectively using `none`).

**Loading from local files:**

You can also instruct mistral.rs to load models locally by modifying the `*_model_id` arguments or options:
```bash
./mistralrs_server --port 1234 plain -m . -a mistral
```

The following files must be present in the paths for the options below:
- `--model-id` (server) or `model_id` (python) or `--tok-model-id` (server) or `tok_model_id` (python): 
  - `config.json`
  - `tokenizer_config.json`
  - `tokenizer.json` (if not specified separately)
  - `.safetensors` files.
- `--quantized-model-id` (server) or `quantized_model_id` (python):
  - Specified `.gguf` or `.ggml` file.
- `--x-lora-model-id` (server) or `xlora_model_id` (python):
  - `xlora_classifier.safetensors`
  - `xlora_config.json`
  - Adapters `.safetensors` and `adapter_config.json` files in their respective directories

### Run

To start a server serving Mistral GGUF on `localhost:1234`, 
```bash
./mistralrs_server --port 1234 --log output.log gguf -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -t mistralai/Mistral-7B-Instruct-v0.1 -f mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

Mistral.rs uses subcommands to control the model type. They are generally of format `<XLORA/LORA>-<QUANTIZATION>`. Please run `./mistralrs_server --help` to see the subcommands.

Additionally, for models without quantization, the model architecture should be provided as the `--arch` or `-a` argument in contrast to GGUF models which encode the architecture in the file. It should be one of the following:
- `mistral`
- `gemma`
- `mixtral`
- `llama`
- `phi2`
- `phi3`
- `qwen2`

**Interactive mode:**

You can launch interactive mode, a simple chat application running in the terminal, by passing `-i`:

```bash
./mistralrs_server -i gguf -t mistralai/Mistral-7B-Instruct-v0.1 -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -f mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### Quick examples:

- X-LoRA with no quantization

To start an X-LoRA server with the exactly as presented in [the paper](https://arxiv.org/abs/2402.07148):

```bash
./mistralrs_server --port 1234 x-lora-plain -o orderings/xlora-paper-ordering.json -x lamm-mit/x-lora
```
- LoRA with a model from GGUF

To start an LoRA server with adapters from the X-LoRA paper (you should modify the ordering file to use only one adapter, as the adapter static scalings are all 1 and so the signal will become distorted):

```bash
./mistralrs_server --port 1234 lora-gguf -o orderings/xlora-paper-ordering.json -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q8_0.gguf -a lamm-mit/x-lora
```

Normally with a LoRA model you would use a custom ordering file. However, for this example we use the ordering from the X-LoRA paper because we are using the adapters from the X-LoRA paper.

- With a model from GGUF

To start a server running Mistral from GGUF:

```bash
./mistralrs_server --port 1234 gguf -t mistralai/Mistral-7B-Instruct-v0.1 -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -f mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

- With a model from GGML

To start a server running Llama from GGML:

```bash
./mistralrs_server --port 1234 ggml -t meta-llama/Llama-2-13b-chat-hf -m TheBloke/Llama-2-13B-chat-GGML -f llama-2-13b-chat.ggmlv3.q4_K_M.bin
```

- Plain model from safetensors

To start a server running Mistral from safetensors.

```bash
./mistralrs_server --port 1234 gguf -m mistralai/Mistral-7B-Instruct-v0.1
```

### Structured selection with a `.toml` file

We provide a method to select models with a `.toml` file. The keys are the same as the command line, with `no_kv_cache` and `tokenizer_json` being "global" keys.

Example:
```bash
./mistralrs_server --port 1234 toml -f toml-selectors/gguf.toml
```

**Command line docs**

Command line docs [here](docs/CMD_LINE_DOCS.md)

---

## Supported models
**Quantization support**
|Model|GGUF|GGML|
|--|--|--|
|Mistral 7B |✅| |
|Gemma| | |
|Llama|✅|✅|
|Mixtral 8x7B|✅| |
|Phi 2|✅| |
|Phi 3|✅| |
|Qwen 2| | |

**Device mapping support**
|Model|Supported|
|--|--|
|Normal|✅|
|GGUF|✅|
|GGML| |

**X-LoRA and LoRA support**
|Model|X-LoRA|X-LoRA+GGUF|X-LoRA+GGML|
|--|--|--|--|
|Mistral 7B |✅|✅| |
|Gemma|✅| | |
|Llama|✅|✅|✅|
|Mixtral 8x7B|✅|✅| |
|Phi 2|✅| | |
|Phi 3|✅|✅| |
|Qwen 2| | | |

**Using derivative models**

To use a derivative model, select the model architecture using the correct subcommand. To see what can be passed for the architecture, pass `--help` after the subcommand. For example, when using a different model than the default, specify the following for the following types of models:

- **Normal**: Model id
- **Quantized**: Quantized model id, quantized filename, and tokenizer id
- **X-LoRA**: Model id, X-LoRA ordering
- **X-LoRA quantized**: Quantized model id, quantized filename, tokenizer id, and X-LoRA ordering
- **LoRA**: Model id, LoRA ordering
- **LoRA quantized**: Quantized model id, quantized filename, tokenizer id, and LoRA ordering

See [this](#adapter-ordering-file) section to determine if it is necessary to prepare an X-LoRA/LoRA ordering file, it is always necessary if the target modules or architecture changed, or if the adapter order changed.

It is also important to check the chat template style of the model. If the HF hub repo has a `tokenizer_config.json` file, it is not necessary to specify. Otherwise, templates can be found in `chat_templates` and should be passed before the subcommand. If the model is not instruction tuned, no chat template will be found and the APIs will only accept a prompt, no messages.

For example, when using a Zephyr model:

`./mistralrs_server --port 1234 --log output.txt gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf`

### Adapter model support: X-LoRA and LoRA

An adapter model is a model with X-LoRA or LoRA. X-LoRA support is provided by selecting the `x-lora-*` architecture, and LoRA support by selecting the `lora-*` architecture. Please find docs for adapter models [here](docs/ADAPTER_MODELS.md)

---

### Chat Templates and Tokenizer
Mistral.rs will attempt to automatically load a chat template and tokenizer. This enables high flexibility across models and ensures accurate and flexible chat templating. However, this behavior can be customized. Please find detailed documentation [here](docs/CHAT_TOK.md).

## Contributing
If you have any problems or want to contribute something, please raise an issue or pull request!

Consider enabling `RUST_LOG=debug` environment variable.

If you want to add a new model, please see [our guide](docs/ADDING_MODELS.md).

## CUDA FAQ

- Setting the compiler path:
    - Set the `NVCC_CCBIN` environment variable during build.
- Error: `recompile with -fPIE`:
    - Some Linux distributions require compiling with `-fPIE`.
    - Set the `CUDA_NVCC_FLAGS` environment variable to `-fPIE` during build: `CUDA_NVCC_FLAGS=-fPIE`

## Credits
This project would not be possible without the excellent work at [`candle`](https://github.com/huggingface/candle). Additionally, thank you to all contributors! Contributing can range from raising an issue or suggesting a feature to adding some new functionality.
