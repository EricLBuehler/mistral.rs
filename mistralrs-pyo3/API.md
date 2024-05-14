# `mistralrs` API

These are API docs for the `mistralrs` package.

**Table of contents**
- Full API docs: [here](https://ericlbuehler.github.io/mistral.rs/pyo3/mistralrs.html)
- Docs for the `Which` enum: [here](#which)
- Example: [here](#example)

## `Which`

Each `*_model_id` may be a HF hub repo or a local path.

Additionally, for models without quantization, the model architecture should be provided as the `arch` parameter in contrast to GGUF models which encode the architecture in the file. It should be one of the following:
- `mistral`
- `gemma`
- `mixtral`
- `llama`
- `phi2`
- `phi3`
- `qwen2`

```py
class Which(Enum):
    class Plain:
        model_id: str
        arch: Architecture
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    class XLora:
        arch: Architecture
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    class Lora:
        arch: Architecture
        adapters_model_id: str
        order: str
        model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    class GGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    class XLoraGGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    class LoraGGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    class GGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    class XLoraGGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    class LoraGGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
```


## Example
```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
    )
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[{"role":"user", "content":"Tell me a story about the Rust type system."}],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```