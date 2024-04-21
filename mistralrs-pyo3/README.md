# mistral.rs PyO3 Bindings: `mistralrs`

`mistralrs` is a Python package which provides an API for `mistral.rs`. We build `mistralrs` with the `maturin` build manager.

## Installation
1) `cd` into mistralrs-pyo3.

2) Activate a Python environment if it is not already. For example:

    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

3) Install `maturin` with `pip install maturin[patchelf]`.

4) Install `mistralrs`
    Install `mistralrs` by executing the following in this directory where [features](../README.md#supported-accelerators) such as `cuda` or `flash-attn` may be specified with the `--features` argument just like they would be for `cargo run`.

    ```bash
    maturin develop -r --features ...
    ```

Please find [API docs here](API.md) and the type stubs [here](mistralrs.pyi), which are another great form of documentation.

We also provide [a cookbook here](../examples/python/cookbook.ipynb)!

## Example
```python
from mistralrs import ModelKind, MistralLoader, ChatCompletionRequest

kind = ModelKind.QuantizedGGUF
loader = MistralLoader(
    model_id="mistralai/Mistral-7B-Instruct-v0.1",
    kind=kind,
    no_kv_cache=False,
    repeat_last_n=64,
    quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
)
runner = loader.load()
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res)
```