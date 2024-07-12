# `mistralrs` API

These are API docs for the `mistralrs` package.

**Table of contents**
- Full API docs: [here](https://ericlbuehler.github.io/mistral.rs/pyo3/mistralrs.html)
- Docs for the `Which` enum: [here](#which)
- Example: [here](#example)

## `Which`

Each `*_model_id` may be a HF hub repo or a local path. For quantized GGUF models, a list is accepted if multiples files must be specified.

Additionally, for models without quantization, the model architecture should be provided as the `arch` parameter in contrast to GGUF models which encode the architecture in the file. It should be one of the following:

### Architecture for plain models
- `Mistral`
- `Gemma`
- `Mixtral`
- `Llama`
- `Phi2`
- `Phi3`
- `Qwen2`
- `Gemma2`
- `Starcoder2`

### Architecture for vision models
- `Phi3V`
- `Idefics2`
- `LLaVaNext`
- `LLaVa`

```py
class Which(Enum):
    """
    Which model to select. See the docs for the `Which` enum in API.md for more details.
    Usage:
    >>> Which.Plain(...)
    """
    @dataclass
    class Plain:
        model_id: str
        arch: Architecture
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class XLora:
        xlora_model_id: str
        order: str
        arch: Architecture
        model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
        tgt_non_granular_index: int | None = None

    @dataclass
    class Lora:
        adapters_model_id: str
        order: str
        arch: Architecture
        model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class GGUF:
        quantized_model_id: str
        quantized_filename: str
        tok_model_id: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class XLoraGGUF:
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tok_model_id: str | None = None
        repeat_last_n: int = 64
        tgt_non_granular_index: int | None = None

    @dataclass
    class LoraGGUF:
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class GGML:
        quantized_model_id: str
        quantized_filename: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
        gqa: int | None = None

    @dataclass
    class XLoraGGML:
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tok_model_id: str | None = None
        tgt_non_granular_index: int | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
        gqa: int | None = None

    @dataclass
    class LoraGGML:
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class VisionPlain:
        model_id: str
        arch: VisionArchitecture
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