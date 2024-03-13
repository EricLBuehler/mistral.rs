# mistral.rs
[![Documentation](https://github.com/EricLBuehler/mistral.rs/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/mistral.rs/mistralrs/)

Mistral.rs is a LLM inference platform written in pure, safe Rust.

## Upcoming features
- Falcon model.
- X-LoRA: Scalings `topk`.
- X-LoRA: Softmax `topk`.
- Fused kernels ([#13](https://github.com/EricLBuehler/mistral.rs/pull/13))
    - Softmax.
    - Rotary Embedding.

## Description
- Fast performance with per-sequence and catch-up KV cache management technique.
- 2-bit, 3-bit, 4-bit, 5-bit, 6-bit and 8-bit quantization for faster inference and optimized memory usage.
- First X-LoRA inference platform with first class support.
- Continuous batching.
- Lightweight OpenAI API compatible HTTP server.
- Python API.
- Apple silicon support with the Metal framework.

**Supported models:**
- Mistral 7B
- Gemma
- Llama
- Mixtral 8x7B

|Model|GGUF|GGML|X-LoRA|X-LoRA+GGUF|X-LoRA+GGML|
|--|--|--|--|--|--|
|Mistral 7B |✅| |✅|✅| |
|Gemma| | |✅| | |
|Llama|✅|✅|✅|✅|✅|
|Mixtral 8x7B|✅| |✅|✅|✅|

**Note when using quantized derivative models**

Please note that when using a derivative model with a quantized architecture, it is important to specify the corresponding model ID for the tokenizer with `-t`.

`./mistralrs-server --port 1234 --log output.txt mistral-gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf`

**Rust Library API**

Rust multithreaded API for easy integration into any application.

- [Docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/)
- To install: Add `mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git" }`

**Python API**

Python API for mistral.rs.

- [Docs](mistralrs-pyo3/README.md)
- [Example](examples/python/python_api.py)

**HTTP Server**

OpenAI API compatible API server

- [Docs](examples/http.md).
- [Running](README.md#run)
- [Example](examples/server/chat.py)

## Benchmarks
**A6000** X-LoRA Mistral GGUF + CUDA (8-bit quantization, prompt tokens = 27, completion tokens = 64)
- 3.13 tok/s

**A10** Mistral + CUDA (prompt tokens = 37, completion tokens = 96)
- 32.16 tok/s

**A10** Mistral GGUF + CUDA (prompt tokens = 37, completion tokens = 105)
- 42.3 tok/s

## Usage
### Installation and Build
To install mistral.rs, one should ensure they have Rust installed by following [this](https://rustup.rs/) link. Additionally, the Huggingface token should be provided in `~/.cache/huggingface/token` when using the server to enable automatic download of gated models.

**Easy quickstart**
For an easy quickstart, the script below will 
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

1) Install required packages
    - `libssl` (ex., `sudo apt install libssl-dev`)
    - `pkg-config` (ex., `sudo apt install pkg-config`)

2) Install Rust
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

3) Set HF token correctly (skip if already done)
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
    - Build
        ```bash
        cargo build --release --features cuda
        ```
    - Install with `cargo install` for easy command line usage
        ```bash
        cargo build --path mistralrs-server --features cuda
        ```


The build process will output a binary `misralrs-server` at `./target/release/mistralrs-server` which may be copied into the working directory with `cp ./target/release/mistralrs-server .`.

### Run

To start a server serving Mistral on `localhost:1234`, 
```bash
./mistralrs-server --port 1234 --log output.log mistral
```

Mistral.rs uses subcommands to control the model type. They are of format `<XLORA>-<ARCHITECTURE>-<QUANTIZATION>`. Please run `./mistralrs-server --help` to see the subcommands.

To start an X-LoRA server with the default weights and ordering (exactly as presented in [the paper](https://arxiv.org/abs/2402.07148)):

`./mistralrs-server --port 1234 x-lora-mistral -o x-lora-orderings/default-ordering.json`


### Building for GPU, Metal or enabling other features
Rust uses a feature flag system during build to implement compile-time build options. As such, the following is a list of features
which may be specified using the `--features` command.
1) `cuda`
2) `cudnn` (if installed, to be used with `cuda`)
2) `metal` (mutally excl. to `cuda`)
3) `flash-attn` (mutally excl. to `metal`, only has an affect on non-quantized models)

### X-LoRA
**Preparing the X-LoRA Ordering File**
The X-LoRA ordering file is necessary to prepare before inference with an X-LoRA model. However, it is easy with a provided [`script`](scripts/create_ordering.py)!

The X-LoRA ordering JSON file contains 2 parts. The first is the order of the adapters and the second, the layer ordering. The layer ordering has been automatically generated and should not be manipulated as it controls the application of scalings. However the order of adapter should be an array of strings which are the adapter names corresponding to the order the adapters were specified during training. For example, if the adapters were specified as a dictionary:

```python
adapters = {
    "math": ...,
    "reasoning": ...,
    "biology": ...
}
```

The specified order would be `["math", "reasoning", "biology"]`.

There are 2 scripts to prepare the ordering file. The ordering file is specific to each architecture and set of target modules. Therefore, if either are changed, it is necessary to create a new ordering file using the first option. If only the adapter order or adapters changed, then it the second option should be used.

1) From scratch: No ordering file for the architecture and target modules

    A script [`create_ordering.py`](scripts/create_ordering.py) is provided which prompts the user for the model ID, target modules, and adapter names. The user is prompted for an output file location, relative to the working directory.

2) Create a new ordering file from an existing ordering file for an architecture and target modules

    A script [`modify_names.py`](scripts/modify_names.py) is provided which prompts the user for the adapter names and the old ordering file. The user is prompted for an output file location, relative to the working directory.

A provide a [default ordering file](scripts/default-ordering.json) which contains the ordering for the X-LoRA model associated with [the paper](https://arxiv.org/abs/2402.07148) and the Huggingface repository: https://huggingface.co/lamm-mit/x-lora.

---

**Quantized X-LoRA models**

Mistral.rs supports running quantized models with X-LoRA. The X-LoRA layers will not be quantized, only the base model. Please note that
using a high quantization level (eg., 4-bit) can distort the signal and prevent the classifier from acting properly. Therefore, it is better to use slightly higher levels such as 8-bit.

**Supported X-LoRA quantized layers**
- model.layers.{layer_idx}.self_attn.q_proj
- model.layers.{layer_idx}.self_attn.k_proj
- model.layers.{layer_idx}.self_attn.v_proj
- model.layers.{layer_idx}.self_attn.o_proj
- model.layers.{layer_idx}.mlp.up_proj
- model.layers.{layer_idx}.mlp.down_proj
- model.layers.{layer_idx}.mlp.gate_proj

### Chat Templates and Tokenizer
**Chat Templates**

Mistral.rs attempts to automatically load a chat template from the `tokenizer_config.json` file. This enables high flexibility across instruction-tuned models and ensures accurate chat templating. However, if the `chat_template` field is missing, then a JINJA chat template should be provided. The JINJA chat template may use `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs. Some chat templates are provided [here](chat_templates), and it is easy to modify or create others.

For example, to use the `chatml` template, `--chat-template` is specified *before* the model architecture. For example:

```bash
./mitralrs-server --port 1234 --log output.log --chat-template ./chat_templates/chatml.json llama
```

If no JINJA chat template is provided, then the default chat template located [here](default.json) will be loaded. It is recommended to copy this file to the working directory where `./mistralrs-server` will be run.

**Tokenizer**

Some models do not provide a `tokenizer.json` file although mistral.rs expects one. To solve this, please run [this](scripts/get_tokenizers_json.py) script. It will output the `tokenizer.json` file for your specific model. This may be used by passing the `--tokenizer-json` flag *after* the model architecture. For example:

```bash
$ python3 scripts/get_tokenizers_json.py
Enter model ID: microsoft/Orca-2-13b
$ ./mistralrs-server --port 1234 --log output.log llama --tokenizer-json tokenizer.json
```

Putting it all together, to run, for example, an [Orca](https://huggingface.co/microsoft/Orca-2-13b) model (which does not come with a `tokenizer.json` or chat template):
1) Generate the `tokenizer.json` by running the script at `scripts/get_tokenizers_json.py`. This will output some files including `tokenizer.json` in the working directory.
2) Find and copy the correct chat template from `chat-templates` to the working directory (eg., `cp chat_templates/chatml.json .`)
3) Run `mistralrs-server`, specifying the tokenizer and chat template: `cargo run --release --features cuda -- --port 1234 --log output.txt --chat-template chatml.json llama -m microsoft/Orca-2-13b -t tokenizer.json`
