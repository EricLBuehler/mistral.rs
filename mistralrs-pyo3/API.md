# `mistralrs` API
## Loaders
The following classes provide a more bare-bones method to load a model.
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

Additionally, the following ergonomic classes provide a more streamlined method which take one of the above loader classes (without instantiation):
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

Each class has one method:
### `load(self, token_source: str = "cache", max_seqs: int = 16, truncate_sequence: bool = false, logfile: str | None = None, revision: str | None = None, token_source_value: str | None = None) -> Runner`
Load a model.

- `token_source`
Specify token source and token source value as the following pairing:
"cache" -> None
"literal" -> str
"envvar" -> str
"path" -> str

- `max_seqs`: Maximum running sequences at any time.

- `truncate_sequence`:
If a sequence is larger than the maximum model length, truncate the number
of tokens such that the sequence will fit at most the maximum length.
If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.

- `logfile`: Log all responses and requests to this file.

- `revision`: HF revision.

- `token_source_value`: Value of token source value for `token_source`

- `dtype=None`: Datatype to load the model into, only applicable for non-quantized models.

## `Runner`

Runner has no constructor and is created by calling `load` on a loader class.

### `send_chat_completion_request(self, request: ChatCompletionRequest) -> str`
Send an OpenAI compatible request, returning JSON.

## `ChatCompletionRequest`
Request is a class with a constructor which accepts the following arguments. It is used to create a chat completion request.

- `messages: list[dict[String, String]]`
- `model: str`
- `logit_bias: dict[int, float]`
- `logprobs: bool`
- `top_logprobs: usize | None`
- `max_tokens: usize | None`
- `n_choices: usize`
- `presence_penalty: float | None`
- `repetition_penalty: float | None`
- `stop_token_ids: list[int] | None`
- `temperature: float | None`
- `top_p: float | None`
- `top_k: usize | None`

`ChatCompletionRequest(messages, model, logprobs = false, n_choices = 1, logit_bias = None, top_logprobs = None, max_tokens = None, presence_penalty = None, repetition_penalty = None, stop_token_ids = None, temperature = None, top_p = None, top_k = None)`

## `ModelKind`
- Normal
- XLoraNormal
- XLoraGGUF
- XLoraGGML
- QuantizedGGUF
- QuantizedGGML

## `DType`
- U8
- U32
- I64
- BF16
- F16
- F32
- F64