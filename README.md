# mistral.rs
[![Documentation](https://github.com/EricLBuehler/mistral.rs/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/mistral.rs/mistralrs_core/)

Mistral.rs is a LLM serving platform written in pure, safe Rust. Please see docs [here](https://ericlbuehler.github.io/mistral.rs/mistralrs_core/).

## Quickstart
For Ubuntu-like systems:
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
cargo run --release --features cuda -- --port 1234 --log output.log mistral-gguf
```