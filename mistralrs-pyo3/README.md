# Mistral.rs PyO3 Bindings: `mistralrs`

`mistralrs` is a Python package which provides an API for `mistral.rs`.

## Installation
1) `cd` into mistralrs-pyo3.

2) Activate a Python environment. For example:

    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    pip install maturin
    ```

3) Install `maturin` with `pip install maturin[patchelf]`.

4) Install `mistralrs`
    Install `mistralrs` by executing the following in this directory where [features](../README.md#building-for-gpu-metal-or-enabling-other-features) such as `cuda` or `flash-attn` may be specified with the `--features` argument.

    ```bash
    maturin develop -r --features ...
    ```

Please find [API docs here](API.md).

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
        repetition_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res)
```

The API consists of the following loader classes:
- `MistralLoader`
- `MixtralLoader`
- `GemmaLoader`
- `LlamaLoader`
- `NormalLoader`
- `XLoraLoader`
- `QuantizedLoader`
- `XLoraQuantizedLoader`

These should be instantiatd to begin setup process. Calling the `load` method will load the model, attempting to download the model.