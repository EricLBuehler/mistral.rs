# mistral.rs
[![Documentation](https://github.com/EricLBuehler/mistral.rs/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/mistral.rs/mistralrs/)

Mistral.rs is a fast LLM inference platform written. We support inference on a variety of devices, quantization, and easy-to-use application with an Open-AI API compatible HTTP server and Python bindings. 

## Current work and upcoming features
- More models: please submit requests [here](https://github.com/EricLBuehler/mistral.rs/issues/156).
- X-LoRA: Scalings `topk` and softmax `topk` ([#48](https://github.com/EricLBuehler/mistral.rs/issues/48)).
- Parallel linear layers (sharding) ([#50](https://github.com/EricLBuehler/mistral.rs/issues/50)). ⭐ Top priority, active work
- Device offloading ([#157](https://github.com/EricLBuehler/mistral.rs/pull/157)). ⭐ Top priority, active work
- Completion streaming support (pending PR, this is holding back the Llama-index integration).

**Running the new Llama 3 model on CUDA**

`cargo run --release --features cuda -- -i llama -m meta-llama/Meta-Llama-3-8B-Instruct`

**Running the new Llama 3 model on Metal**

`cargo run --release --features metal -- -i llama -m meta-llama/Meta-Llama-3-8B-Instruct`

**Running the new Llama 3 model on CPU**

`cargo run --release -- -i llama -m meta-llama/Meta-Llama-3-8B-Instruct`

## Description
**Fast**:
- Quantized model support: 2-bit, 3-bit, 4-bit, 5-bit, 6-bit and 8-bit for faster inference and optimized memory usage.
- Continuous batching.
- Prefix caching.

**Accelerator support**:
- Apple silicon support with the Metal framework.
- CPU inference with `mkl`, `accelerate` support and optimized backend.
- CUDA support with flash attention and CUDNN.

**Easy**:
- Lightweight OpenAI API compatible HTTP server.
- Python API.
- Grammar support with Regex and Yacc.

**Powerful**:
- Fast LoRA support with weight merging.
- First X-LoRA inference platform with first class support.


This is a demo of interactive mode with streaming running Mistral GGUF:

https://github.com/EricLBuehler/mistral.rs/assets/65165915/3396abcd-8d44-4bf7-95e6-aa532db09415


**Supported models:**
- Mistral 7B (v0.1 and v0.2)
- Gemma
- Llama, including Llama 3
- Mixtral 8x7B
- Phi 2

Please see [this section](README#supported-models) for details on quantization and LoRA support.

## APIs and Integrations
**Rust Library API**

Rust multithreaded API for easy integration into any application.

- [Docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/)
- To install: Add `mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git" }`

**Python API**

Python API for mistral.rs.

- [Docs and installation](mistralrs-pyo3/README.md)
- [Example](examples/python/python_api.py)
- [Cookbook](examples/python/cookbook.ipynb)

```python
from mistralrs import Runner, Which, ChatCompletionRequest, Message, Role

runner = Runner(
    which=Which.MistralGGUF(
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
        messages=[Message(Role.User, "Tell me a story about the Rust type system.")],
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

- [Source](integrations/llama_index_integration.py).
- [Example](examples/llama_index/xlora_gguf.py)

---

## Supported accelerators
- CUDA:
  - Enable with `cuda` feature: `--features cuda`
  - Flash attention support with `flash-attn` feature, only applicable to non-quantized models: `--features flash-attn`
  - CUDNN support with `cudnn` feature: `--features cudnn`
- Metal:
  - Enable with `metal` feature: `--features metal`
- CPU:
  - Intel MKL with `mkl` feature: `--features mkl`
  - Apple Accelerate with `accelerate` feature: `--features accelerate`

Enabling features is done by passing `--features ...` to the build system. When using `cargo run` or `maturin develop`, pass the `--features` flag before the `--` separating build flags from runtime flags.

- To enable a single feature like `metal`: `cargo build --release --features metal`.
- To enable multiple features, specify them in quotes: `cargo build --release --features "cuda flash-attn cudnn"`.

## Benchmarks
|Device|Mistral.rs Completion T/s|Llama.cpp Completion T/s|
|-|-|-|
|A10 GPU, CUDA|76.11|79.24|
|Intel Xeon 8358 CPU|6.22|18.14|

More benchmarks coming!

## Usage
### Installation and Build
To install mistral.rs, one should ensure they have Rust installed by following [this](https://rustup.rs/) link. Additionally, the Huggingface token should be provided in `~/.cache/huggingface/token` when using the server to enable automatic download of gated models.

**Easy quickstart**
For an easy quickstart on a `*nix` system, the script below will 
download an setup Rust and then build mistral.rs to run with CUDA.
```bash
sudo apt update -y
sudo apt install libssl-dev -y
sudo apt install pkg-config -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
mkdir ~/.cache/huggingface
touch ~/.cache/huggingface/token
echo <HF_TOKEN_HERE> > ~/.cache/huggingface/token
cargo build --release --features cuda
```

**Detailed guide**
Please see the [detailed guide here](INSTALLATION.md).

### Run

To start a server serving Mistral on `localhost:1234`, 
```bash
./mistralrs-server --port 1234 --log output.log mistral
```

Mistral.rs uses subcommands to control the model type. They are of format `<XLORA/LORA>-<ARCHITECTURE>-<QUANTIZATION>`. Please run `./mistralrs-server --help` to see the subcommands.

**Interactive mode:**

You can launch interactive mode, a simple chat application running in the terminal, by passing `-i`:

```bash
./mistralrs-server -i mistral-gguf
```

**Quick examples:**

- X-LoRA with no quantization

To start an X-LoRA server with the exactly as presented in [the paper](https://arxiv.org/abs/2402.07148):

`./mistralrs-server --port 1234 x-lora-mistral -o orderings/xlora-paper-ordering.json -m HuggingFaceH4/zephyr-7b-beta -x lamm-mit/x-lora`

- LoRA with a model from GGUF

To start an LoRA server with adapters from the X-LoRA paper (you should modify the ordering file to use only one adapter, as the adapter static scalings are all 1 and so the signal will become distorted):

`./mistralrs-server --port 1234 lora-mistral-gguf -o orderings/xlora-paper-ordering.json -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q8_0.gguf -a lamm-mit/x-lora`

Normally with a LoRA model you would use a custom ordering file. However, for this example we use the ordering from the X-LoRA paper because we are using the adapters from the X-LoRA paper.

- With a model from GGUF

To start a server running Llama from GGUF:

`./mistralrs-server --port 1234 llama-gguf -t meta-llama/Llama-2-13b-chat-hf -m TheBloke/Llama-2-13B-chat-GGUF -f llama-2-13b-chat.Q4_K_M.gguf`

- With a model from GGML

To start a server running Llama from GGML:

`./mistralrs-server --port 1234 llama-ggml -t meta-llama/Llama-2-13b-chat-hf -m TheBloke/Llama-2-13B-chat-GGML -f llama-2-13b-chat.ggmlv3.q4_K_M.bin`

- Single prompt inference

To run a single prompt and then shut down:

`./mistralrs-server --prompt "Hello!" mistral-gguf -t mistralai/Mistral-7B-Instruct-v0.1 -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -f mistral-7b-instruct-v0.1.Q4_K_M.gguf`

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
|Phi 2| | |

**X-LoRA and LoRA support**
|Model|X-LoRA|X-LoRA+GGUF|X-LoRA+GGML|
|--|--|--|--|
|Mistral 7B |✅|✅| |
|Gemma|✅| | |
|Llama|✅|✅|✅|
|Mixtral 8x7B|✅|✅| |
|Phi 2|✅| | |

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

`./mistralrs-server --port 1234 --log output.txt mistral-gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf`

### Adapter model support: X-LoRA and LoRA

An adapter model is a model with X-LoRA or LoRA. X-LoRA support is provided by selecting the `x-lora-*` architecture, and LoRA support by selecting the `lora-*` architecture. Please find docs for adapter models [here](docs/ADAPTER_MODELS.md)

---

### Chat Templates and Tokenizer
Mistral.rs will attempt to automatically load a chat template and tokenizer. This enables high flexibility across models and ensures accurate and flexible chat templating. However, this behavior can be customized. Please find detailed documentation [here](docs/CHAT_TOK.md).

## Contributing
If you have any problems or want to contribute something, please raise an issue or pull request!

Consider enabling `RUST_LOG=debug` environment variable.

If you want to add a new model, please see [our guide](docs/ADDING_MODELS.md).

## Credits
This project would not be possible without the excellent work at [`candle`](https://github.com/huggingface/candle). Additionally, thank you to all contributors! Contributing can range from raising an issue or suggesting a feature to adding some new functionality.