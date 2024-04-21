# Installing Mistral.rs

1) Install required packages
    - `libssl` (ex., `sudo apt install libssl-dev`)
    - `pkg-config` (ex., `sudo apt install pkg-config`)

2) Install Rust: https://rustup.rs/
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

3) Set HF token correctly (skip if already set or your model is not gated)
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
6) The build process will output a binary `misralrs-server` at `./target/release/mistralrs-server` which may be copied into the working directory with the following command:
    ```
    cp ./target/release/mistralrs-server .
    ```