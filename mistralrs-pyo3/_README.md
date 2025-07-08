# mistral.rs PyO3 Bindings: `mistralrs`

`mistralrs` is a Python package which provides an easy to use API for `mistral.rs`. 

## Example
More examples can be found [here](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/python)!

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

## Installation from PyPi
With the PyPi installation method, you can take advantage of prebuild wheels to easily install.

- CUDA

  `pip install mistralrs-cuda -v`
- Metal

  `pip install mistralrs-metal -v`
- Apple Accelerate

  `pip install mistralrs-accelerate -v`
- Intel MKL

  `pip install mistralrs-mkl -v`
- Without accelerators

  `pip install mistralrs -v`

All installations will install the `mistralrs` package. The suffix on the package installed by `pip` only controls the feature activation.

## Installation from source
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
    Install `mistralrs` by executing the following in this directory where [features](../README.md#supported-accelerators) such as `cuda` or `flash-attn` may be specified with the `--features` argument just like they would be for `cargo run`.

    
    ```bash
    maturin develop -r --features <specify feature(s) here>
    ```
    
Please find [API docs here](API.md) and the type stubs [here](mistralrs.pyi), which are another great form of documentation.

We also provide [a cookbook here](../examples/python/cookbook.ipynb)!
