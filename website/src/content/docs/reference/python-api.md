---
title: Python API reference
description: Public surface of the mistralrs Python package.
sidebar:
  order: 6
---

This page lists the types and methods that the Python SDK exposes. For usage examples see [Tutorial 3](/mistral.rs/tutorials/03-python-sdk/) and the [Python guides](/mistral.rs/guides/python/).

The package name for imports is always `mistralrs`, regardless of which PyPI wheel you installed (`mistralrs`, `mistralrs-cuda`, `mistralrs-metal`, etc.).

## Runner

The central class. Owns a loaded model and exposes methods for sending requests.

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
    search_callback: Callable[[str], list[dict]] | None = None,
    tool_callbacks: Mapping[str, Callable[[str, dict], str]] | None = None,
)
```

### Request methods

```python
send_chat_completion_request(
    request: ChatCompletionRequest,
    model_id: str | None = None,
) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]
```

Returns a `ChatCompletionResponse` if `request.stream=False`, or an iterator of chunks if `request.stream=True`.

```python
send_completion_request(request: CompletionRequest, model_id=None)
send_embedding_request(request: EmbeddingRequest, model_id=None)
generate_image(prompt: str, response_format: ImageGenerationResponseFormat, model_id=None)
generate_audio(prompt: str, model_id=None)
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

### Tokenization

```python
tokenize_text(text: str, add_special_tokens: bool = True, model_id=None) -> list[int]
detokenize_text(tokens: list[int], skip_special_tokens: bool = True, model_id=None) -> str
```

### Session management

```python
export_session(session_id: str) -> dict
import_session(session_id: str, serialized: dict)
delete_session(session_id: str)
list_session_ids() -> list[str]
```

See [agentic sessions guide](/mistral.rs/guides/python/agentic-session/) for usage.

### Code execution (feature-gated)

These methods are available only when mistralrs was built with the `code-execution` feature (the default):

```python
exec_in_session(session_id: str, code: str) -> CodeExecResult
reset_session_python(session_id: str)
```

## Which

Specifies which kind of model to load.

```python
Which.Plain(model_id: str, arch: Architecture | None = None)
Which.MultimodalPlain(model_id: str, arch: MultimodalArchitecture | None = None)
Which.GGUF(tok_model_id: str, quantized_model_id: str, quantized_filename: str)
Which.GGML(...)  # same shape as GGUF
Which.Lora(model_id: str, adapters_model_id: str, order: str)
Which.XLora(...)
Which.Speech(model_id: str, arch: SpeechArchitecture)
Which.DiffusionPlain(model_id: str, arch: DiffusionArchitecture)
Which.Embedding(model_id: str)
```

Architecture enums (`Architecture`, `MultimodalArchitecture`, `SpeechArchitecture`, `DiffusionArchitecture`) enumerate the model families we support. You rarely need to pass them explicitly; auto-detection works for all supported models.

## Request types

`ChatCompletionRequest`:

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
    session_id: str | None = None
```

`CompletionRequest`, `EmbeddingRequest`: similar shape; see `mistralrs.pyi` in the repository for the full fields.

## Response types

`ChatCompletionResponse`:

```python
@dataclass
class ChatCompletionResponse:
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    system_fingerprint: str | None
    choices: list[Choice]
    usage: Usage
    session_id: str | None
    agentic_tool_calls: list[AgenticToolCallRecord] | None
```

`ChatCompletionChunkResponse`: the streaming chunk. Same shape with `choices[i].delta` instead of `choices[i].message`.

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

## Full type stubs

The canonical type definitions live in `mistralrs.pyi` in the source repository. IDEs that respect stub files will give you full autocomplete from that.
