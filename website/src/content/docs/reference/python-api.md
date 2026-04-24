---
title: Python API reference
description: Public surface of the mistralrs Python package.
sidebar:
  order: 6
---

For usage examples, see [Tutorial 3](/mistral.rs/tutorials/03-python-sdk/) and the [Python guides](/mistral.rs/guides/python/).

The import name is always `mistralrs`, regardless of installed wheel (`mistralrs`, `mistralrs-cuda`, `mistralrs-metal`, etc.).

The canonical type definitions live in `mistralrs.pyi` in the source repository.

## Runner

```python
Runner(
    which: Which,
    max_seqs: int = 16,
    no_kv_cache: bool = False,
    prefix_cache_n: int = 16,
    token_source: str = "cache",
    speculative_gamma: int = 32,
    which_draft: Which | None = None,
    chat_template: str | None = None,
    jinja_explicit: str | None = None,
    num_device_layers: list[str] | None = None,
    in_situ_quant: str | None = None,
    anymoe_config: AnyMoeConfig | None = None,
    pa_gpu_mem: int | float | None = None,
    pa_blk_size: int | None = None,
    pa_cache_type: PagedCacheType | None = None,
    no_paged_attn: bool = False,
    paged_attn: bool = False,
    seed: int | None = None,
    enable_search: bool = False,
    search_embedding_model: str | None = None,
    search_callback: Callable[[str], list[dict[str, str]]] | None = None,
    tool_callbacks: Mapping[str, Callable[[str, dict], str]] | None = None,
)
```

### Request methods

```python
send_chat_completion_request(
    request: ChatCompletionRequest,
    model_id: str | None = None,
) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]

send_completion_request(request: CompletionRequest, model_id=None)
send_embedding_request(request: EmbeddingRequest, model_id=None)
generate_image(prompt: str, response_format: ImageGenerationResponseFormat, model_id=None)
generate_audio(prompt: str, model_id=None)
```

### Tokenization

```python
tokenize_text(text: str, add_special_tokens: bool = True, model_id=None) -> list[int]
detokenize_text(tokens: list[int], skip_special_tokens: bool = True, model_id=None) -> str
```

### Model management

```python
list_models() -> list[str]
list_models_with_status() -> list[tuple[str, str]]
get_default_model_id() -> str
set_default_model_id(model_id: str)
is_model_loaded(model_id: str) -> bool
unload_model(model_id: str)
reload_model(model_id: str)
```

## Which

```python
Which.Plain(model_id: str, arch: Architecture | None = None)
Which.MultimodalPlain(model_id: str, arch: MultimodalArchitecture | None = None)
Which.GGUF(tok_model_id: str, quantized_model_id: str, quantized_filename: str)
Which.GGML(...)
Which.Lora(model_id: str, adapters_model_id: str, order: str)
Which.XLora(...)
Which.XLoraGGUF(...)
Which.LoraGGUF(...)
Which.XLoraGGML(...)
Which.LoraGGML(...)
Which.Speech(model_id: str, arch: DiffusionArchitecture)
Which.DiffusionPlain(model_id: str, arch: DiffusionArchitecture)
Which.Embedding(model_id: str)
```

Architecture enums (`Architecture`, `MultimodalArchitecture`, `DiffusionArchitecture`) enumerate supported model families. Auto-detection covers all supported models.

## ChatCompletionRequest

```python
@dataclass
class ChatCompletionRequest:
    messages: list[dict] | str
    model: str
    logit_bias: dict[int, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n_choices: int = 1
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop_seqs: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    top_k: int | None = None
    grammar: str | None = None
    grammar_type: str | None = None
    min_p: float | None = None
    tool_schemas: list[str] | None = None
    tool_choice: ToolChoice | None = None
    web_search_options: WebSearchOptions | None = None
    enable_thinking: bool | None = None
    truncate_sequence: bool = False
```

`CompletionRequest`, `EmbeddingRequest`, similar shape. See `mistralrs.pyi` for full fields.

## Enums

```python
class ToolChoice(Enum):
    NoTools = "None"
    Auto = "Auto"

class SearchContextSize(Enum):
    Low = "low"
    Medium = "medium"
    High = "high"
```

## Sessions

`Runner` exposes session management methods:

```python
runner.export_session(session_id: str, model_id: str | None = None) -> str | None
runner.import_session(session_id: str, session_json: str, model_id: str | None = None) -> None
runner.delete_session(session_id: str, model_id: str | None = None) -> bool
runner.list_session_ids(model_id: str | None = None) -> list[str]
```

`export_session` returns a JSON string (or `None` if the session does not exist). `import_session` takes the same JSON. `ChatCompletionRequest` does not accept a `session_id` field; for session-scoped requests, call the HTTP server.
