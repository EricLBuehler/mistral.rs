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

To start a server serving Mistral on `localhost:1234`, 
```bash
./target/release/mistralrs --port 1234 --log output.log mistral
```

## X-LoRA Ordering File
The X-LoRA ordering JSON file contains 2 parts. The first is the order of the adapters. In the template, this is an empty array but should be filled in with strings. The second part is the layer ordering. This has been generated and should not be manipulated as it controls the application of scalings.