# Mistral.rs PyO3 Bindings

To use, activate a Python virtual environment and ensure that `maturin` is installed, for example:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install maturin
```

And then install `mistralrs` by executing the following in this directory.

```bash
maturin develop -r --features ...
```

Features such as `cuda` or `flash-attn` may be specified with the `--features` argument.

For an example of how to use the bindings, see below.

```python
from mistralrs import ModelKind, MistralLoader, Request

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
res = runner.add_request(
    Request(
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