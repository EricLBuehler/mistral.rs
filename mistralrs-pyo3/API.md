# `mistralrs` API
## Loaders
There are several ways to load different architectures of model:
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
    - `tgt_non_granular_index=None`: Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached. If this is set then the max running sequences will be set to 1.
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
    - `tgt_non_granular_index=None`: Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached. If this is set then the max running sequences will be set to 1.

The base loader classes listed below are passed to the wrapper loader classes above during construction:
- `MistralLoader`
- `GemmaLoader`
- `LlamaLoader`
- `MixtralLoader`
- `Phi2Loader`

Each class has one method:
### `load(self, token_source: str = "cache", max_seqs: int = 16, truncate_sequence: bool = false, logfile: str | None = None, revision: str | None = None, token_source_value: str | None = None) -> Runner`
Load a model.

- `token_source`
Specify token source and token source value as the following pairing:
"cache" -> None
"literal" -> str
"envvar" -> str
"path" -> str
"none" -> None

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

### `send_chat_completion_request(self, request: ChatCompletionRequest) -> str | ChatCompletionStreamer`
Send an OpenAI compatible request, returning OpenAI compatible JSON or a streamer which returns OpenAI compatible JSON chunks.

### `send_completion_request(self, request: CompletionRequest) -> str`
Send an OpenAI compatible request, returning OpenAI compatible JSON.

## `ChatCompletionRequest`
Request is a class with a constructor which accepts the following arguments. It is used to create a chat completion request to pass to `send_chat_completion_request`.

- `messages: list[dict[str, str]]`
- `model: str`
- `logit_bias: dict[int, float]`
- `logprobs: bool`
- `top_logprobs: usize | None`
- `max_tokens: usize | None`
- `n_choices: usize`
- `presence_penalty: float | None`
- `frequency_penalty: float | None`
- `stop_token_ids: list[int] | None`
- `temperature: float | None`
- `top_p: float | None`
- `top_k: usize | None`
- `stream: bool = False`

`ChatCompletionRequest(messages, model, logprobs = false, n_choices = 1, logit_bias = None, top_logprobs = None, max_tokens = None, presence_penalty = None, frequency_penalty = None, stop_token_ids = None, temperature = None, top_p = None, top_k = None, stream = False)`

## `CompletionRequest`
Request is a class with a constructor which accepts the following arguments. It is used to create a chat completion request to pass to `send_completion_request`.

- `prompt: str`
- `model: str`
- `best_of: int`
- `echo_prompt: bool = False`
- `logit_bias: dict[int, float] | None = None`
- `max_tokens: int | None = None`
- `n_choices: int = 1`
- `best_of: int = 1`
- `presence_penalty: float | None = None`
- `frequency_penalty: float | None = None`
- `stop_seqs: list[str] | None = None`
- `temperature: float | None = None`
- `top_p: float | None = None`
- `top_k: int | None = None`
- `suffix: str | None = None`
- `grammar: str | None = None`
- `grammar_type: str | None = None`

`CompletionRequest(prompt, model, best_of, echo_prompt = False, logit_bias = None, max_tokens: None, n_choices = 1, best_of = 1, presence_penalty = None, frequency_penalty = None, stop_seqs = None, temperature = None, top_p = None, top_k = None, suffix = None, grammar = None, grammar_type = None)`

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