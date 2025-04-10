# CLI installation

> Note: You can use our [Docker containers here](https://github.com/EricLBuehler/mistral.rs/pkgs/container/mistral.rs).
> 
> Learn more about running Docker containers: https://docs.docker.com/engine/reference/run/


1) Install required packages:
    - `OpenSSL` (*Example on Ubuntu:* `sudo apt install libssl-dev`)
    - <b>*Linux only:*</b> `pkg-config` (*Example on Ubuntu:* `sudo apt install pkg-config`)

2) Install Rust: https://rustup.rs/

    *Example on Ubuntu:*
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

3) <b>*Optional:*</b> Set HF token correctly (skip if already set or your model is not gated, or if you want to use the `token_source` parameters in Python or the command line.)
    - Note: you can install `huggingface-cli` as documented [here](https://huggingface.co/docs/huggingface_hub/en/installation#install-with-pip). 
    ```bash
    huggingface-cli login
    ```

4) Download the code:
    ```bash
    git clone https://github.com/EricLBuehler/mistral.rs.git
    cd mistral.rs
    ```

5) Build or install:

    mistral.rs controls compilation of hardware accelerator-specific and optimized components with feature flags.
    You can select feature flags to use using [this guide](features.md).
    
    - Build
        ```bash
        cargo build --release --features ...
        ```
    - Install with `cargo install` for easy command line usage

        Pass the same values to `--features` as you would for `cargo build`
        ```bash
        cargo install --path mistralrs-server --features ...
        ```

6) The build process will output a binary `mistralrs-server` at `./target/release/mistralrs-server`. We can switch to that directory so that the binary can be accessed as `./mistralrs-server` with the following command:

    *Example on Ubuntu:*
    ```
    cd target/release
    ```

7) Use our APIs and integrations: 
    
    [APIs and integrations list](#apis-and-integrations)
