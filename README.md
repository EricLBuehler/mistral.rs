# mistral.rs
[![Documentation](https://github.com/EricLBuehler/mistral.rs/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/mistral.rs/mistralrs/)

Mistral.rs is a LLM inference platform written in pure, safe Rust.

## Upcoming features
- Falcon

## Description
- Lightweight OpenAI API compatible HTTP server.
- Python API.
- Fast performance with per-sequence and catch-up KV cache management technique.
- Continuous batching.
- First X-LoRA inference platform with first class support.
- 2-bit, 3-bit, 4-bit, 5-bit, 6-bit and 8-bit quantization for faster inference and optimized memory usage.
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

Please note that when using a derivative model with a quantized architecture, it is important to specify the corresponding model ID for the tokenizer with `-t` as failing to do so will result in a incorrect prompt templating:

`./mistralrs-server --port 1234 --log output.txt mistral-gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf`

**Rust Library API**

Rust multithreaded API for easy integration into any application: [docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/). To use, add `mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git" }` to the Cargo.toml.

**Python API**

A Python API is provided. Please see [these docs](mistralrs-pyo3/README.md) for getting started, and [this file](examples/python_api.py) for a use case.

**HTTP Server**

Mistral.rs provides an OpenAI API compatible API server, documentation [here](examples/http.md).

To get started see [this](README.md#run) section, and [this file](examples/chat.py) for an example of a simple chat program.

## Benchmarks
**A6000** X-LoRA Mistral GGUF + CUDA (8-bit quantization, prompt tokens = 27, completion tokens = 64)
- 3.13 tok/s

**A10** Mistral + CUDA (prompt tokens = 37, completion tokens = 96)
- 32.16 tok/s

**A10** Mistral GGUF + CUDA (prompt tokens = 37, completion tokens = 105)
- 42.3 tok/s

## Usage
### Build
To build mistral.rs, one should ensure they have Rust installed by following [this](https://rustup.rs/) link.
The Huggingface token should be provided in `~/.cache/huggingface/token`. 
- Using a script

    For an easy quickstart, the script below will 
    download an setup Rust and then build mistral.rs to run on the CPU.
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
    cargo build --release
    ```
- Manual build

    If Rust is installed and the Huggingface token is set, then one may build mistral.rs by executing the build command.
    `cargo build --release`.
The build process will output a binary `misralrs-server` at `./target/release/mistralrs-server`.
### Building for GPU, Metal or enabling other features
Rust uses a feature flag system during build to implement compile-time build options. As such, the following is a list of features
which may be specified using the `--features` command.
1) `cuda`
2) `metal`
3) `flash-attn`

### X-LoRA
**Preparing the X-LoRA Ordering File**
The X-LoRA ordering file is only necessary if the X-LoRA config JSON file does not contain a mapping of adapters. If it does, it is not necessary to modify the ordering file.

The X-LoRA ordering JSON file contains 2 parts. The first is the order of the adapters and the second, the layer ordering. The layer ordering has been automatically generated and should not be manipulated as it controls the application of scalings. However the order of adapter should be an array of strings which are the adapter names corresponding to the order the adapters were specified during training. For example, if the adapters were specified as a dictionary:

```python
adapters = {
    "math": ...,
    "reasoning": ...,
    "biology": ...
}
```

The specified order would be `["math", "reasoning", "biology"]`

For convenience, a script `generate.py` in `xlora-orderings` is provided which prompts the user for a comma delimited list of numbers and then writes the file. If the X-LoRA config contains a mapping of adapters defining the order, it is not necessary to use this script. However, for both cases, it is recommended to copy the modified ordering file from `xlora-orderings.json` to the directory where the server will be run for convenience. For example:

`cp ./xlora-orderings/mistral-ordering.json ordering.json`

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
Mistral.rs attempts to automatically load a chat template from the `tokenizer_config.json` file. This enables high flexibility across instruction-tuned models and ensures accurate chat templating. However, if the `chat_template` field is missing, then a manual JINJA template will be used:

```bash
./mitralrs-server --port 1234 --log output.log --chat-template ... llama
```

The JINJA chat template may use `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs. Some default chat templates are provided [here](chat_templates). For example, if one wanted to use an Orca model where the first message is always the system prompt and the second is always the user's message, the following may be used: 

```bash
./mitralrs-server --port 1234 --log output.log --chat-template ./chat_templates/chatml.json llama
```

If no JINJA chat template is provided, then the default chat template located [here](default.json) will be loaded. It is recommended to copy this file to the working directory where `./mistralrs-server` will be run.

Some models do not provide a `tokenizer.json` file although mistral.rs expects one. To solve this, please run [this](examples/get_tokenizers_json.py) script. It will output the `tokenizer.json` file for your specific model. This may be uploaded to HF hub or a manually specified tokenizer path can be specified by passing the `--tokenizer-json` flag after the model architecture has been selected. For example:

```bash
$ python3 examples/get_tokenizers_json.py
Enter model ID: microsoft/Orca-2-13b
$ ./mistralrs-server --port 1234 --log output.log llama --tokenizer-json tokenizer.json
```

## Run

To start a server serving Mistral on `localhost:1234`, 
```bash
./mistralrs-server --port 1234 --log output.log mistral
```

Mistral.rs uses subcommands to control the model type. They are of format `<XLORA>-<ARCHITECTURE>-<QUANTIZATION>`. Please run `./mistralrs-server --help` to see the subcommands.

To start an X-LoRA server with the default weights, run the following after modifying or copying the ordering file as described [here](README.md#x-lora).

`./mistralrs-server --port 1234 x-lora-mistral -o ordering.json`
