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
- Llama
- Mixtral 8x7B
- Phi 2

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

**Rust Library API**

Rust multithreaded API for easy integration into any application.

- [Docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/)
- To install: Add `mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git" }`

**Python API**

Python API for mistral.rs.

- [Docs and installation](mistralrs-pyo3/README.md)
- [Example](examples/python/python_api.py)
- [Cookbook](examples/python/cookbook.ipynb)

**HTTP Server**

OpenAI API compatible API server

- [API Docs](examples/http.md).
- [Running](README.md#run)
- [Example](examples/server/chat.py)

**Llama Index integration**

- [Source](integrations/llama_index_integration.py).
- [Example](examples/llama_index/xlora_gguf.py)

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

Enabling features is done by passing `--features ...` to the build system. When using `cargo run`, pass the `--features` flag before the `--` separating build flags from runtime flags.

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
`./mistralrs-server --help`

```bash
Fast and easy LLM serving.

Usage: mistralrs-server [OPTIONS] <COMMAND>

Commands:
  mistral              Select the mistral model
  mistral-gguf         Select the quantized mistral model with gguf
  x-lora-mistral       Select the mistral model, with X-LoRA
  gemma                Select the gemma model
  x-lora-gemma         Select the gemma model, with X-LoRA
  llama                Select the llama model
  llama-gguf           Select the quantized llama model with gguf
  llama-ggml           Select the quantized llama model with gguf
  x-lora-llama         Select the llama model, with X-LoRA
  mixtral              Select the mixtral model
  mixtral-gguf         Select the quantized mixtral model with gguf
  x-lora-mixtral       Select the mixtral model, with X-LoRA
  x-lora-mistral-gguf  Select the quantized mistral model with gguf and X-LoRA
  x-lora-llama-gguf    Select the quantized mistral model with gguf and X-LoRA
  x-lora-llama-ggml    Select the quantized mistral model with gguf and X-LoRA
  x-lora-mixtral-gguf  Select the quantized mistral model with gguf and X-LoRA
  phi2                 Select the phi2 model
  x-lora-phi2          Select the phi2 model, with X-LoRA
  lora-mistral-gguf    Select the mistral model, with LoRA and gguf
  lora-mistral         Select the mistral model, with LoRA
  lora-mixtral         Select the mixtral model, with LoRA
  lora-llama           Select the llama model, with LoRA
  lora-llama-gguf      Select the quantized mistral model with gguf and LoRA
  lora-llama-ggml      Select the quantized mistral model with gguf and LoRA
  lora-mixtral-gguf    Select the quantized mistral model with gguf and LoRA
  help                 Print this message or the help of the given subcommand(s)

Options:
      --serve-ip <SERVE_IP>
          IP to serve on. Defaults to "0.0.0.0"
  -p, --port <PORT>
          Port to serve on
  -l, --log <LOG>
          Log all responses and requests to this file
  -t, --truncate-sequence
          If a sequence is larger than the maximum model length, truncate the number of tokens such that the sequence will fit at most the maximum length. If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead
      --max-seqs <MAX_SEQS>
          Maximum running sequences at any time. If the `tgt_non_granular_index` flag is set for X-LoRA models, this will be set to 1 [default: 16]
      --no-kv-cache
          Use no KV cache
  -c, --chat-template <CHAT_TEMPLATE>
          JINJA chat template with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs. Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded
      --token-source <TOKEN_SOURCE>
          Source of the token for authentication. Can be in the formats: "literal:<value>", "env:<value>", "path:<value>", "cache" to use a cached token or "none" to use no token. Defaults to using a cached token [default: cache]
  -i, --interactive-mode
          Enter interactive mode instead of serving a chat server
      --prefix-cache-n <PREFIX_CACHE_N>
          Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy [default: 16]
      --prompt <PROMPT>
          Run a single prompt. This cannot be used with interactive mode
      --prompt-concurrency <PROMPT_CONCURRENCY>
          Requires --prompt. Number of prompt completions to run concurrently in prompt mode [default: 1]
      --prompt-max-tokens <PROMPT_MAX_TOKENS>
          Requires --prompt. Number of prompt tokens to generate [default: 128]
  -h, --help
          Print help
  -V, --version
          Print version
```

Docs for quantized models, specifically for: `./mistralrs mistral-gguf --help`

```bash
Select the quantized mistral model with gguf

Usage: mistralrs-server mistral-gguf [OPTIONS]

Options:
  -t, --tok-model-id <TOK_MODEL_ID>
          Model ID to load the tokenizer from [default: mistralai/Mistral-7B-Instruct-v0.1]
      --tokenizer-json <TOKENIZER_JSON>
          Path to local tokenizer.json file. If this is specified it is used over any remote file
  -m, --quantized-model-id <QUANTIZED_MODEL_ID>
          Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set. If it is set to an empty string then the quantized filename will be used as a path to the GGUF file [default: TheBloke/Mistral-7B-Instruct-v0.1-GGUF]
  -f, --quantized-filename <QUANTIZED_FILENAME>
          Quantized filename, only applicable if `quantized` is set [default: mistral-7b-instruct-v0.1.Q4_K_M.gguf]
      --repeat-last-n <REPEAT_LAST_N>
          Control the application of repeat penalty for the last n tokens [default: 64]
  -h, --help
          Print help
```

Docs for X-LoRA and quantized models, specifically for: `./mistralrs-server x-lora-mistral-gguf --help`

```bash
Select the quantized mistral model with gguf and X-LoRA

Usage: mistralrs-server x-lora-mistral-gguf [OPTIONS] --order <ORDER>

Options:
  -t, --tok-model-id <TOK_MODEL_ID>
          Model ID to load the tokenizer from [default: HuggingFaceH4/zephyr-7b-beta]
      --tokenizer-json <TOKENIZER_JSON>
          Path to local tokenizer.json file. If this is specified it is used over any remote file
  -m, --quantized-model-id <QUANTIZED_MODEL_ID>
          Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set. If it is set to an empty string then the quantized filename will be used as a path to the GGUF file [default: TheBloke/zephyr-7B-beta-GGUF]
  -f, --quantized-filename <QUANTIZED_FILENAME>
          Quantized filename, only applicable if `quantized` is set [default: zephyr-7b-beta.Q8_0.gguf]
      --repeat-last-n <REPEAT_LAST_N>
          Control the application of repeat penalty for the last n tokens [default: 64]
  -x, --xlora-model-id <XLORA_MODEL_ID>
          Model ID to load X-LoRA from [default: lamm-mit/x-lora]
  -o, --order <ORDER>
          Ordering JSON file
      --tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>
          Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached. This makes the maximum running sequences 1
  -h, --help
          Print help
```

---

### Adapter model support: X-LoRA and LoRA

An adapter model is a model with X-LoRA or LoRA. X-LoRA support is provided by selecting the `x-lora-*` architecture, and LoRA support by selecting the `lora-*` architecture. For both X-LoRA and LoRA, an ordering file (see [this section](#adapter-ordering-file) for preparing the ordering file) must be provided. The ordering file describes the ordering of layers and which adapters to use (and what order to use them in for X-LoRA).

When using an adapter model with a quantized base model, if the ordering file specifies unsupported layers you will receive an error.

**Supported X-LoRA or LoRA quantized layers**
- model.layers.{layer_idx}.self_attn.q_proj
- model.layers.{layer_idx}.self_attn.k_proj
- model.layers.{layer_idx}.self_attn.v_proj
- model.layers.{layer_idx}.self_attn.o_proj
- model.layers.{layer_idx}.mlp.up_proj
- model.layers.{layer_idx}.mlp.down_proj
- model.layers.{layer_idx}.mlp.gate_proj

### Adapter ordering file
**Preparing the X-LoRA/LoRA Ordering File**
The X-LoRA/LoRA ordering file is necessary to prepare before inference with an X-LoRA model. However, it is easy with a provided [`script`](scripts/create_ordering.py)!

The X-LoRA/LoRA ordering JSON file contains 2 parts. The first is the order of the adapters and the second, the layer ordering. The layer ordering has been automatically generated and should not be manipulated as it controls the application of scalings. However the order of adapter should be an array of strings which are the adapter names corresponding to the order the adapters were specified during training. For example, if the adapters were specified as a dictionary:

```python
adapters = {
    "math": ...,
    "reasoning": ...,
    "biology": ...
}
```

The specified order would be `["math", "reasoning", "biology"]`.

For LoRA models, the order of the adapters does not matter. You can reorder them or remove some to control which adapters will be used. However, for an X-LoRA model, the order of the adapters in the ordering file is important.

There are 2 scripts to prepare the ordering file. The ordering file is specific to each architecture and set of target modules. Therefore, if either are changed, it is necessary to create a new ordering file using the first option. If only the adapter order or adapters changed, then it the second option should be used.

1) From scratch: No ordering file for the architecture and target modules

    A script [`create_ordering.py`](scripts/create_ordering.py) is provided which prompts the user for the model ID, target modules, and adapter names. The user is prompted for an output file location, relative to the working directory.

2) Create a new ordering file from an existing ordering file for an architecture and target modules

    A script [`modify_names.py`](scripts/modify_names.py) is provided which prompts the user for the adapter names and the old ordering file. The user is prompted for an output file location, relative to the working directory.

A provide a [ordering file](scripts/xlora-paper-ordering.json) which contains the ordering for the X-LoRA model associated with [the paper](https://arxiv.org/abs/2402.07148) and the Huggingface repository: https://huggingface.co/lamm-mit/x-lora.

**Quantized X-LoRA or LoRA models**

Mistral.rs supports running quantized models with X-LoRA or LoRA. The X-LoRA or LoRA adapter layers will not be quantized, only the base model. Please note that using a high quantization level (eg., 4-bit) can distort the signal and prevent the classifier from acting properly. Therefore, it is better to use slightly lower levels such as 8-bit.


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
If you have any problems or want to contribute something, please raise an issue or pull request!

Consider enabling `RUST_LOG=debug` environment variable.

If you want to add a new model, please see [our guide](ADDING_MODELS.md).

## Credits
This project would not be possible without the excellent work at [`candle`](https://github.com/huggingface/candle). Additionally, thank you to all contributors! Contributing can range from raising an issue or suggesting a feature to adding some new functionality.