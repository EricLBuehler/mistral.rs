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
res = runner.send_chat_completion_request(
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

## Supported Models
The API consists of the following loader classes:
- `MistralLoader`
    - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    - `kind`: Model kind
    - `no_kv_cache=False`: Disable kv cache.
    - `use_flash_attn=<feature>`: Use flash attn, only used if feature is enabled.
    - `repeat_last_n=64`: Repeat last n context window.
    - `gqa=None`: GQA, irrelevant.
    - `order_file=None`: Ordering JSON file.
    - `quantized_model_id=None`: Quantized model ID.
    - `quantized_filename=None`: Quantized filename (gguf/ggml),
    - `xlora_model_id=None`: X-LoRA model
    - `chat_template=None`: Chat template literal or file.
    - `tokenizer_json=None`: Tokenizer json file.
- `MixtralLoader`
    - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    - `kind`: Model kind
    - `no_kv_cache=False`: Disable kv cache.
    - `use_flash_attn=<feature>`: Use flash attn, only used if feature is enabled.
    - `repeat_last_n=64`: Repeat last n context window.
    - `gqa=1`: GQA, irrelevant.
    - `order_file=None`: Ordering JSON file.
    - `quantized_model_id=None`: Quantized model ID.
    - `quantized_filename=None`: Quantized filename (gguf/ggml),
    - `xlora_model_id=None`: X-LoRA model
    - `chat_template=None`: Chat template literal or file.
    - `tokenizer_json=None`: Tokenizer json file.
- `GemmaLoader`
    - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    - `kind`: Model kind
    - `no_kv_cache=False`: Disable kv cache.
    - `use_flash_attn=None`: Use flash attn, irrelevant.
    - `repeat_last_n=64`: Repeat last n context window.
    - `gqa=None`: GQA, irrelevant.
    - `order_file=None`: Ordering JSON file.
    - `quantized_model_id=None`: Quantized model ID.
    - `quantized_filename=None`: Quantized filename (gguf/ggml),
    - `xlora_model_id=None`: X-LoRA model
    - `chat_template=None`: Chat template literal or file.
    - `tokenizer_json=None`: Tokenizer json file.
- `LlamaLoader`
    - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    - `kind`: Model kind
    - `no_kv_cache=False`: Disable kv cache.
    - `use_flash_attn=<feature>`: Use flash attn, only used if feature is enabled.
    - `repeat_last_n=64`: Repeat last n context window.
    - `gqa=1`: GQA, irrelevant if non quantized model type.
    - `order_file=None`: Ordering JSON file.
    - `quantized_model_id=None`: Quantized model ID.
    - `quantized_filename=None`: Quantized filename (gguf/ggml),
    - `xlora_model_id=None`: X-LoRA model
    - `chat_template=None`: Chat template literal or file.
    - `tokenizer_json=None`: Tokenizer json file.

Additionally, the following four ergonomic classes provide a more streamlined method which take one of the above loader classes (without instantiation):
- `NormalLoader`
    - `class`: Loader class.
    - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    - `no_kv_cache=False`: Disable kv cache.
    - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
    - `repeat_last_n=64`: Repeat last n context window.
    - `gqa=None`: GQA, irrelevant if non quantized model type.
    - `chat_template=None`: Chat template literal or file.
    - `tokenizer_json=None`: Tokenizer json file.
- `XLoraLoader`
    - `class`: Loader class.
    - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    - `no_kv_cache=False`: Disable kv cache.
    - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
    - `repeat_last_n=64`: Repeat last n context window.
    - `gqa=None`: GQA, irrelevant if non quantized model type.
    - `order_file=None`: Ordering JSON file.
    - `xlora_model_id=None`: X-LoRA model
    - `chat_template=None`: Chat template literal or file.
    - `tokenizer_json=None`: Tokenizer json file.
- `QuantizedLoader`
    - `class`: Loader class.
    - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    - `is_gguf`: Loading gguf or ggml.
    - `no_kv_cache=False`: Disable kv cache.
    - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
    - `repeat_last_n=64`: Repeat last n context window.
    - `gqa=None`: GQA, irrelevant if non quantized model type.
    - `quantized_model_id=None`: Quantized model ID.
    - `quantized_filename=None`: Quantized filename (gguf/ggml),
    - `chat_template=None`: Chat template literal or file.
    - `tokenizer_json=None`: Tokenizer json file.
- `XLoraQuantizedLoader`
    - `class`: Loader class.
    - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    - `is_gguf`: Loading gguf or ggml.
    - `no_kv_cache=False`: Disable kv cache.
    - `use_flash_attn=<feature>`: Use flash attn, only used if feature is enabled.
    - `repeat_last_n=64`: Repeat last n context window.
    - `gqa=1`: GQA, irrelevant if non quantized model type.
    - `order_file=None`: Ordering JSON file.
    - `quantized_model_id=None`: Quantized model ID.
    - `quantized_filename=None`: Quantized filename (gguf/ggml),
    - `xlora_model_id=None`: X-LoRA model
    - `chat_template=None`: Chat template literal or file.
    - `tokenizer_json=None`: Tokenizer json file.

These should be instantiatd to begin setup process. Calling the `.load(token_source = "cache", max_seqs = 2, truncate_sequence = false, logfile = None, revision = None, token_source_value = None)` method will load the model, attempting to download the model.