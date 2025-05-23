# Python Installation from scratch

After you install the Python package, you can find API documentation [here](../../python_docs/api.md).

1) Install required packages
    - `openssl` (*Example on Ubuntu:* `sudo apt install libssl-dev`)
    - `pkg-config` (*Example on Ubuntu:* `sudo apt install pkg-config`)

2) Install Rust: https://rustup.rs/
    
    *Example on Ubuntu:*
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

3) Set HF token correctly (skip if already set or your model is not gated, or if you want to use the `token_source` parameters in Python or the command line.)
    
    *Example on Ubuntu:*
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

5) `cd` into the correct directory for building `mistralrs`:
    `cd mistralrs-pyo3`

6) Install `maturin`, our Rust + Python build system:
    Maturin requires a Python virtual environment such as `venv` or `conda` to be active. The `mistralrs` package will be installed into that
    environment.
    ```
    pip install maturin[patchelf]
    ```

7) Install `mistralrs`

    mistral.rs controls compilation of hardware accelerator-specific and optimized components with feature flags.
    You can select feature flags to use using [this guide](features.md).

    ```bash
    maturin develop -r --features ...
    ```
