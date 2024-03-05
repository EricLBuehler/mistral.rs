# mistral.rs
[![Documentation](https://github.com/EricLBuehler/mistral.rs/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/mistral.rs/mistralrs_core/)

Mistral.rs is a LLM serving platform written in pure, safe Rust. Please see docs [here](https://ericlbuehler.github.io/mistral.rs/mistralrs_core/).

## Features
- **OpenAI compatible API** server.
- **Fast** performance with per-sequence and catch-up KV cache management technique.
- **Arbitrary derivative and GGUF models** allowing specification of any weight file.
- **First X-LoRA inference platform** with first class support.

## Quickstart
To build mistral.rs,
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

It will output a binary called `mistralrs` to `./target/release/mistralrs`.

To start a server serving Mistral on `localhost:1234`, 
```bash
./target/release/mistralrs --port 1234 --log output.log mistral
```

Mistral.rs uses subcommands for organization. For example, to start a normal Mistral server one may use `./target/release/mistralrs  --port 1234 mistral` but to start a GGUF mistral server one may use `./target/release/mistralrs --port 1234 gguf-mistral`. For help with the global program, one should pass `--help` before the model subcommand, but for help with the subcommand the flag should be passed after.

To start an X-LoRA server with the default weights, run the following after modifying or copying the ordering file as descibed [here](README.md#x-lora-ordering-file).

`./target/release/mistralrs --port 1234 x-lora-mistral -m HuggingFaceH4/zephyr-7b-beta -o ordering.json`

## X-LoRA Ordering File
The X-LoRA ordering JSON file contains 2 parts. The first is the order of the adapters. In the template, this is an empty array but should be filled in with strings. The second part is the layer ordering. This has been generated and should not be manipulated as it controls the application of scalings.