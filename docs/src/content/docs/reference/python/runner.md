---
title: Runner
description: "The main entry point. Load a model and send requests."
sidebar:
  order: 2
---
### `Runner`

#### `Runner.__init__`

```python
__init__(
    which: Which,
    max_seqs: int = 16,
    no_kv_cache: bool = False,
    prefix_cache_n: int = 16,
    token_source: str = 'cache',
    speculative_gamma: int = 32,
    which_draft: Which | None = None,
    chat_template: str | None = None,
    jinja_explicit: str | None = None,
    num_device_layers: list[str] | None = None,
    in_situ_quant: str | None = None,
    anymoe_config: AnyMoeConfig | None = None,
    pa_gpu_mem: int | float | None = None,
    pa_gpu_mem_usage: float | None = None,
    pa_ctxt_len: int | None = None,
    pa_blk_size: int | None = None,
    pa_cache_type: PagedCacheType | None = None,
    no_paged_attn: bool = False,
    paged_attn: bool = False,
    seed: int | None = None,
    enable_search: bool = False,
    search_embedding_model: str | None = None,
    search_callback: Callable[[str], list[dict[str, str]]] | None = None,
    tool_callbacks: Mapping[str, Callable[[str, dict], str]] | None = None,
    code_execution_config: CodeExecutionConfig | None = None,
    mcp_client_config: McpClientConfigPy | None = None,
) -> None
```

Load a model.

- `which` specifies which model to load or the target model to load in the case of speculative decoding.
- `max_seqs` specifies how many sequences may be running at any time.
- `no_kv_cache` disables the KV cache.
- `prefix_cache_n` sets the number of sequences to hold in the device prefix cache, others will be evicted to CPU.
- `token_source` specifies where to load the HF token from.
    The token source follows the following format: "literal:<value>", "env:<value>", "path:<value>", "cache" to use a cached token or "none" to use no token.
- `speculative_gamma` specifies the `gamma` parameter for specuative decoding, the ratio of draft tokens to generate before calling
    the target model. If `which_draft` is not specified, this is ignored.
- `which_draft` specifies which draft model to load. Setting this parameter will cause a speculative decoding model to be loaded,
    with `which` as the target (higher quality) model and `which_draft` as the draft (lower quality) model.
- `chat_template` specifies an optional JINJA chat template as a JSON file.
    This chat template should have `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
    It is used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
- `jinja_explicit` allows an explicit JINJA chat template file to be used. If specified, this overrides all other chat templates.
- `num_device_layers` sets the number of layers to load and run on each device.
    Each element follows the format ORD:NUM where ORD is the device ordinal and NUM is
    the corresponding number of layers. Note: this is deprecated in favor of automatic device mapping.
- `in_situ_quant` sets the optional in-situ quantization for a model.
- `anymoe_config` specifies the AnyMoE config. If this is set, then the model will be loaded as an AnyMoE model.
- `pa_gpu_mem`: GPU memory to allocate for KV cache with PagedAttention in MBs.
    PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
- `pa_gpu_mem_usage`: Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    If this is not set and the device is CUDA, it will default to `0.9`.
    PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
- `pa_ctxt_len`: Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
- `pa_blk_size` sets the block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA,
    it will default to 32. PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
- `pa_cache_type` sets the PagedAttention KV cache type (auto or f8e4m3). Defaults to `auto`.
- `no_paged_attn` disables PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
- `paged_attn` enables PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
- `seed`, used to ensure reproducible random number generation.
- `enable_search`: Enable searching compatible with the OpenAI `web_search_options` setting. This loads the selected search embedding reranker (EmbeddingGemma by default).
- `search_embedding_model`: select which built-in search embedding model to load (currently `"embedding_gemma"`).
- `search_callback`: Custom Python callable to perform web searches. Should accept a query string and return a list of dicts with keys "title", "description", "url", and "content".
- `tool_callbacks`: Mapping from tool name to Python callable invoked for generic tool calls. Each callable receives the tool name and a dict of arguments and should return the tool output as a string.
- `code_execution_config`: enables the built-in Python code execution tool. Pass a `CodeExecutionConfig` to configure the interpreter, per-call timeout, and working directory. Per-request, set `ChatCompletionRequest.enable_code_execution=True`.

#### `Runner.send_chat_completion_request`

```python
send_chat_completion_request(
    request: ChatCompletionRequest,
    model_id: str | None = None,
) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]
```

Send a chat completion request to the mistral.rs engine, returning the response object or a generator
over chunk objects.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `ChatCompletionRequest` | required | The chat completion request. |
| `model_id` | `str \| None` | `None` | Optional model ID to send the request to. If None, uses the default model. |

#### `Runner.send_completion_request`

```python
send_completion_request(
    request: CompletionRequest,
    model_id: str | None = None,
) -> CompletionResponse
```

Send a completion request to the mistral.rs engine, returning the response object.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `CompletionRequest` | required | The completion request. |
| `model_id` | `str \| None` | `None` | Optional model ID to send the request to. If None, uses the default model. |

#### `Runner.send_embedding_request`

```python
send_embedding_request(
    request: EmbeddingRequest,
    model_id: str | None = None,
) -> list[list[float]]
```

Generate embeddings for the supplied inputs and return one embedding vector per input.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `EmbeddingRequest` | required | The embedding request. |
| `model_id` | `str \| None` | `None` | Optional model ID to send the request to. If None, uses the default model. |

#### `Runner.generate_image`

```python
generate_image(
    prompt: str,
    response_format: ImageGenerationResponseFormat,
    height: int = 720,
    width: int = 1280,
    model_id: str | None = None,
) -> ImageGenerationResponse
```

Generate an image.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | `str` | required | The image generation prompt. |
| `response_format` | `ImageGenerationResponseFormat` | required | The response format (url or b64_json). |
| `height` | `int` | `720` | Image height in pixels. |
| `width` | `int` | `1280` | Image width in pixels. |
| `model_id` | `str \| None` | `None` | Optional model ID to send the request to. If None, uses the default model. |

#### `Runner.generate_audio`

```python
generate_audio(
    prompt: str,
    model_id: str | None = None,
) -> SpeechGenerationResponse
```

Generate audio given a (model specific) prompt. PCM and sampling rate as well as the number of channels is returned.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | `str` | required | The audio generation prompt. |
| `model_id` | `str \| None` | `None` | Optional model ID to send the request to. If None, uses the default model. |

#### `Runner.send_re_isq`

```python
send_re_isq(dtype: str, model_id: str | None = None) -> None
```

Send a request to re-ISQ the model. If the model was loaded as GGUF or GGML then nothing will happen.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `dtype` | `str` | required | The ISQ dtype (e.g., "Q4K", "Q8_0"). |
| `model_id` | `str \| None` | `None` | Optional model ID to re-ISQ. If None, uses the default model. |

#### `Runner.tokenize_text`

```python
tokenize_text(
    text: str,
    add_special_tokens: bool,
    enable_thinking: bool | None = None,
    model_id: str | None = None,
) -> list[int]
```

Tokenize some text, returning raw tokens.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` | required | The text to tokenize. |
| `add_special_tokens` | `bool` | required | Whether to add special tokens. |
| `enable_thinking` | `bool \| None` | `None` | Enables thinking for models that support this configuration. |
| `model_id` | `str \| None` | `None` | Optional model ID to use for tokenization. If None, uses the default model. |

#### `Runner.detokenize_text`

```python
detokenize_text(
    tokens: list[int],
    skip_special_tokens: bool,
    model_id: str | None = None,
) -> str
```

Detokenize some tokens, returning text.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `tokens` | `list[int]` | required | The tokens to detokenize. |
| `skip_special_tokens` | `bool` | required | Whether to skip special tokens. |
| `model_id` | `str \| None` | `None` | Optional model ID to use for detokenization. If None, uses the default model. |

#### `Runner.max_sequence_length`

```python
max_sequence_length(model_id: str | None = None) -> int | None
```

Return the maximum supported sequence length for the current or specified model, or None when
the concept does not apply (such as diffusion or speech models).

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str \| None` | `None` | Optional model ID to query. If None, uses the default model. |

#### `Runner.list_models`

```python
list_models() -> list[str]
```

List all available model IDs (aliases if configured).

**Returns:** A list of model ID strings.

#### `Runner.get_default_model_id`

```python
get_default_model_id() -> str | None
```

Get the current default model ID.

**Returns:** The default model ID, or None if no default is set.

#### `Runner.set_default_model_id`

```python
set_default_model_id(model_id: str) -> None
```

Set the default model ID. The model must already be loaded.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | The model ID to set as default. |

**Raises:** ValueError: If the model ID is not found.

#### `Runner.is_model_loaded`

```python
is_model_loaded(model_id: str) -> bool
```

Check if a model is currently loaded in memory.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | The model ID to check. |

**Returns:** True if the model is loaded, False otherwise.

#### `Runner.unload_model`

```python
unload_model(model_id: str) -> None
```

Unload a model from memory while preserving its configuration for later reload.
The model can be reloaded manually with reload_model() or automatically when
a request is sent to it.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | The model ID to unload. |

#### `Runner.reload_model`

```python
reload_model(model_id: str) -> None
```

Manually reload a previously unloaded model.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | The model ID to reload. |

#### `Runner.list_models_with_status`

```python
list_models_with_status() -> list[tuple[str, str]]
```

List all models with their current status.

**Returns:** A list of (model_id, status) tuples where status is one of:
- "loaded": Model is loaded and ready
- "unloaded": Model is unloaded but can be reloaded
- "reloading": Model is currently being reloaded

#### `Runner.list_unloaded_models`

```python
list_unloaded_models() -> list[str]
```

List model IDs that are currently unloaded (but can be reloaded).

#### `Runner.get_model_status`

```python
get_model_status(model_id: str) -> str | None
```

Get the status of a model: "loaded", "unloaded", "reloading", or None if not found.

#### `Runner.remove_model`

```python
remove_model(model_id: str) -> None
```

Remove a model by ID in multi-model mode.

#### `Runner.send_chat_completion_request_to_model`

```python
send_chat_completion_request_to_model(
    request: ChatCompletionRequest,
    model_id: str,
) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]
```

Send a chat completion request to a specific model, returning the response
object or a generator over chunk objects.

#### `Runner.send_completion_request_to_model`

```python
send_completion_request_to_model(
    request: CompletionRequest,
    model_id: str,
) -> CompletionResponse
```

Send a completion request to a specific model.

#### `Runner.export_session`

```python
export_session(
    session_id: str,
    model_id: str | None = None,
) -> str | None
```

Export an agentic session by ID as a JSON string.

Returns None if the session does not exist.

#### `Runner.import_session`

```python
import_session(
    session_id: str,
    session_json: str,
    model_id: str | None = None,
) -> None
```

Import an agentic session from a JSON string.

Replaces any existing session with the same ID.

#### `Runner.delete_session`

```python
delete_session(session_id: str, model_id: str | None = None) -> bool
```

Delete an agentic session. Returns whether the session existed.

#### `Runner.list_session_ids`

```python
list_session_ids(model_id: str | None = None) -> list[str]
```

List all stored agentic session IDs.

#### `Runner.find_file`

```python
find_file(file_id: str) -> File | None
```

Look up a produced file by id. Returns the full body even if the
file was wire-truncated in the response payload.

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
