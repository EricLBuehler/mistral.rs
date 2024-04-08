# mistral.rs
[![Documentation](https://github.com/EricLBuehler/mistral.rs/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/mistral.rs/mistralrs/)

Mistral.rs is a LLM inference platform written in pure, safe Rust.

## Current work
- More models: please submit requests [here](https://github.com/EricLBuehler/mistral.rs/issues/49).
- X-LoRA: Scalings `topk` ([#48](https://github.com/EricLBuehler/mistral.rs/issues/48)).
- X-LoRA: Softmax `topk` ([#48](https://github.com/EricLBuehler/mistral.rs/issues/48)).
- PagedAttention ([#47](https://github.com/EricLBuehler/mistral.rs/pull/47)) ⭐ Active work.
- Model grammar support via BNF ([#59](https://github.com/EricLBuehler/mistral.rs/issues/59)). ⭐ Active work.
- Parallel linear layers (sharding) ([#50](https://github.com/EricLBuehler/mistral.rs/issues/50)).
- LoRA/multi-LoRA adapters support ([#61](https://github.com/EricLBuehler/mistral.rs/issues/61)). ⭐ Active work.
- Prefix caching ([#91](https://github.com/EricLBuehler/mistral.rs/issues/91)). ⭐ Active work.

## Description
- Fast performance with per-sequence and catch-up KV cache management technique.
- 2-bit, 3-bit, 4-bit, 5-bit, 6-bit and 8-bit quantization for faster inference and optimized memory usage.
- First X-LoRA inference platform with first class support.
- Continuous batching.
- Lightweight OpenAI API compatible HTTP server.
- Python API.
- Apple silicon support with the Metal framework.
- CPU inference with `mkl`, `accelerate` support and optimized backend.

**Supported models:**
- Mistral 7B (v0.1 and v0.2)
- Gemma
- Llama
- Mixtral 8x7B
- Phi 2

|Model|GGUF|GGML|X-LoRA|X-LoRA+GGUF|X-LoRA+GGML|
|--|--|--|--|--|--|
|Mistral 7B |✅| |✅|✅| |
|Gemma| | |✅| | |
|Llama|✅|✅|✅|✅|✅|
|Mixtral 8x7B|✅| |✅|✅|✅|
|Phi 2| | |✅| | |

**Using derivative models**
To use a derivative model, select the model architecture using the correct subcommand. To see what can be passed for the architecture, pass `--help` after the subcommand. For example, when using a different model than the default, specify the following for the following types of models:

- **Normal**: Model id
- **Quantized**: Quantized model id, quantized filename, and tokenizer id
- **X-LoRA**: Model id, X-LoRA ordering
- **X-LoRA quantized**: Quantized model id, quantized filename, tokenizer id, and X-LoRA ordering

See [this](#x-lora) section to determine if it is necessary to prepare an X-LoRA ordering file, it is always necessary if the target modules or architecture changed, or if the adapter order changed.

It is also important to check the chat template style of the model. If the HF hub repo has a `tokenizer_config.json` file, it is not necessary to specify. Otherwise, templates can be found in `chat_templates` and should be passed before the subcommand.

For example, when using a Zephyr model:

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

## Supported accelerators
- CUDA:
  - Enable with `cuda` feature.
  - Flash attention support with `flash-attn` feature, only applicable to non-quantized models.
  - CUDNN support with `cudnn` feature.
- Metal:
  - Enable with `metal` feature.
- CPU:
  - Intel MKL with `mkl` feature.
  - Apple Accelerate with `accelerate` feature.

## Benchmarks
Coming soon!

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
        cargo install --path mistralrs-server --features cuda
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

---

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

**Quantized X-LoRA models**

Mistral.rs supports running quantized models with X-LoRA. The X-LoRA layers will not be quantized, only the base model. Please note that
using a high quantization level (eg., 4-bit) can distort the signal and prevent the classifier from acting properly. Therefore, it is better to use slightly lower levels such as 8-bit.

**Supported X-LoRA quantized layers**
- model.layers.{layer_idx}.self_attn.q_proj
- model.layers.{layer_idx}.self_attn.k_proj
- model.layers.{layer_idx}.self_attn.v_proj
- model.layers.{layer_idx}.self_attn.o_proj
- model.layers.{layer_idx}.mlp.up_proj
- model.layers.{layer_idx}.mlp.down_proj
- model.layers.{layer_idx}.mlp.gate_proj


**Avoiding the scaling pass with non-granular scalings**

The X-LoRA implementation supports non-granular scalings. This caches the scalings after `k` completion tokens are generated and they will be used for the remaining passes avoiding the scaling pass. The number of tokens to generate before caching is defined by setting `tgt_non_granular_index`. Setting `tgt_non_granular_index` will restrict the maximum running sequences to 1.

---

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

## Contributing
If you have any problems or want to contribute something, please raise an issue or pull request! If you want to add a new model, please see [our guide](ADDING_MODELS.md).
