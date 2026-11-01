---
title: Runner
description: "The main entry point. Load a model and send requests."
sidebar:
  order: 2
---
## `Runner`

### `Runner.__init__`

```text
__init__(
    which: Which,
    max_seqs: int = 32,
    no_kv_cache: bool = False,
    prefix_cache_n: int | None = None,
    token_source: str = 'cache',
    mtp_model: str | None = None,
    mtp_n_predict: int | None = None,
    chat_template: str | None = None,
    jinja_explicit: str | None = None,
    num_device_layers: list[str] | None = None,
    in_situ_quant: str | None = None,
    anymoe_config: AnyMoeConfig | None = None,
    memory_goal: str | float | None = None,
    pa_blk_size: int | None = None,
    pa_cache_type: PagedCacheType | None = None,
    no_paged_attn: bool = False,
    paged_attn: bool = False,
    seed: int | None = None,
    enable_search: bool = False,
    search_embedding_model: str | None = None,
    search_callback: Callable[[str], list[dict[str, str]]] | None = None,
    tool_callbacks: Mapping[str, Callable[[str, dict], str]] | None = None,
    mcp_client_config: McpClientConfigPy | None = None,
    code_execution_config: CodeExecutionConfig | None = None,
    shell_config: ShellConfig | None = None,
) -> None
```

Load a model.

- `which`: the model to load. See [Which](/reference/python/which/).
- `max_seqs`: how many requests can run at once.
- `no_kv_cache`: turn off the KV cache, and paged attention with it.
- `prefix_cache_n`: how many finished requests' caches to keep, so a later request that starts
  with the same prompt can skip recomputing it.
    - `None` (default): keep up to 16. Models with recurrent layers may keep fewer to fit memory.
    - A number: keep exactly that many.
    - `0`: turn prefix caching off.
- `token_source`: where to read the Hugging Face token. One of `"cache"`, `"none"`,
  `"literal:<token>"`, `"env:<VAR>"`, or `"path:<file>"`.
- `mtp_model`: a separate drafter model, by ID or path, for
  [speculative decoding](/guides/perf/speculative-decoding/).
- `mtp_n_predict`: how many tokens the drafter proposes per step. If unset, mistral.rs picks the
  depth as it runs. See [draft depth](/guides/perf/speculative-decoding/#draft-depth).
- `chat_template`: fallback chat template, used only when the model's files don't include one. A JSON
  file with a `chat_template` key, plus optional `bos_token`, `eos_token`, and `unk_token`.
- `jinja_explicit`: a `.jinja` chat template file. It overrides every other template source.
- `num_device_layers`: place layers by hand instead of mapping them automatically. Each entry is
  `"ORD:NUM"`, putting `NUM` layers on device `ORD`. A single number puts that many on device 0.
- `in_situ_quant`: quantize the weights while loading. Takes a format such as `"q4k"`, or a bit width
  such as `"4"` that picks a format for the device. See
  [quantize a model](/guides/quantization/quantize-a-model/).
- `anymoe_config`: load the model as an AnyMoE model. See `AnyMoeConfig`.
- `memory_goal`: how much memory to plan for. A fraction or a size requires paged attention.
  See [paged attention](/guides/perf/paged-attention/#memory-budget).
    - `None` (default): size the KV cache from the memory left after loading.
    - A fraction such as `0.9`: keep weights, working memory, and cache under that share of GPU memory.
    - A size such as `"8GiB"` or `"65536tokens"`: give the KV cache exactly that much.
- `pa_blk_size`: tokens per paged attention block: 8, 16, or 32. Defaults to 32.
- `pa_cache_type`: KV cache storage type. See `PagedCacheType`.
- `no_paged_attn`: turn paged attention off. It's on by default on CUDA and Metal.
- `paged_attn`: require paged attention, failing if the device doesn't support it. Can't be combined
  with `no_paged_attn`.
- `seed`: random seed, for reproducible sampling.
- `enable_search`: load a reranker for web search results. The default is EmbeddingGemma.
- `search_embedding_model`: which reranker to load. The only option is `"embedding_gemma"`.
- `search_callback`: a function to run web searches yourself. It takes the query string and returns
  a list of dicts with `"title"`, `"description"`, `"url"`, and `"content"`.
- `tool_callbacks`: map of tool name to a Python function that runs that tool. The function gets the
  tool name and a dict of arguments, and returns the result as a string.
- `mcp_client_config`: connect to MCP servers and give the model their tools. See [MCP](/reference/python/mcp/).
- `code_execution_config`: turn on the built-in Python tool. See `CodeExecutionConfig`.
- `shell_config`: turn on the built-in shell tool. See `ShellConfig`.

### `Runner.send_chat_completion_request`

```text
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

### `Runner.send_completion_request`

```text
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

### `Runner.send_decision_request`

```text
send_decision_request(
    request: DecisionRequest,
    model_id: str | None = None,
) -> DecisionResponse
```

Evaluate named decision questions with the selected model, without text generation.

### `Runner.send_embedding_request`

```text
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

### `Runner.generate_image`

```text
generate_image(
    prompt: str,
    response_format: ImageGenerationResponseFormat,
    height: int = 720,
    width: int = 1280,
    model_id: str | None = None,
    save_file: str | None = None,
) -> ImageGenerationResponse
```

Generate an image.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | `str` | required | The image generation prompt. |
| `response_format` | `ImageGenerationResponseFormat` | required | The response format (Url or B64Json). |
| `height` | `int` | `720` | Image height in pixels. |
| `width` | `int` | `1280` | Image width in pixels. |
| `model_id` | `str \| None` | `None` | Optional model ID to send the request to. If None, uses the default model. |
| `save_file` | `str \| None` | `None` | Optional path where the PNG is written when response_format is Url. Defaults to an auto-generated filename. |

### `Runner.generate_audio`

```text
generate_audio(
    prompt: str,
    model_id: str | None = None,
) -> SpeechGenerationResponse
```

Generate speech with a dedicated speech model such as Dia or a chat speech
model such as LFM3-Audio or LFM3-Omni. Returns PCM samples, sample rate, and
channel count. LFM3 uses audio-only synthesis with its default voice.

Use send_chat_completion_request with modalities and audio for custom chat
prompts, mixed text/speech output, encoded formats, or PCM16 streaming.
HTTP instructions, speed, and audio.id history are not options on this helper.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | `str` | required | The audio generation prompt. |
| `model_id` | `str \| None` | `None` | Optional model ID to send the request to. If None, uses the default model. |

### `Runner.send_re_isq`

```text
send_re_isq(dtype: str, model_id: str | None = None) -> None
```

Re-ISQ a model that was loaded with `in_situ_quant`.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `dtype` | `str` | required | The ISQ dtype (e.g., "Q4K", "Q8_0"). |
| `model_id` | `str \| None` | `None` | Optional model ID to re-ISQ. If None, uses the default model. |

### `Runner.begin_calibration`

```text
begin_calibration(model_id: str | None = None) -> CalibrationStatus
```

Begin online calibration: collect activation statistics from live traffic on every
ISQ-tracked layer. The model must have been loaded with ISQ.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str \| None` | `None` | Optional model ID. If None, uses the default model. |

### `Runner.calibration_status`

```text
calibration_status(model_id: str | None = None) -> CalibrationStatus
```

Report per-layer calibration collection progress.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str \| None` | `None` | Optional model ID. If None, uses the default model. |

### `Runner.apply_calibration`

```text
apply_calibration(
    save_cimatrix: str | None = None,
    model_id: str | None = None,
) -> CalibrationStatus
```

Requantize from the source weights with the collected statistics and hot-swap the
layers into the live model. Returns the pre-apply status.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `save_cimatrix` | `str \| None` | `None` | Optional `.cimatrix` path to save the collected importance matrix. |
| `model_id` | `str \| None` | `None` | Optional model ID. If None, uses the default model. |

### `Runner.tokenize_text`

```text
tokenize_text(
    text: str,
    add_special_tokens: bool,
    enable_thinking: bool | None,
    model_id: str | None = None,
) -> list[int]
```

Tokenize some text, returning raw tokens.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` | required | The text to tokenize. |
| `add_special_tokens` | `bool` | required | Whether to add special tokens. |
| `enable_thinking` | `bool \| None` | required | Compatibility argument; raw text tokenization does not render a chat template. |
| `model_id` | `str \| None` | `None` | Optional model ID to use for tokenization. If None, uses the default model. |

### `Runner.detokenize_text`

```text
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

### `Runner.max_sequence_length`

```text
max_sequence_length(model_id: str | None = None) -> int | None
```

Return the maximum supported sequence length for the current or specified model, or None when
the concept does not apply (such as diffusion or speech models).

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str \| None` | `None` | Optional model ID to query. If None, uses the default model. |

### `Runner.list_models`

```text
list_models() -> list[str]
```

List all available model IDs (aliases if configured).

**Returns:** A list of model ID strings.

### `Runner.get_default_model_id`

```text
get_default_model_id() -> str | None
```

Get the current default model ID.

**Returns:** The default model ID, or None if no default is set.

### `Runner.set_default_model_id`

```text
set_default_model_id(model_id: str) -> None
```

Set the default model ID. The model must already be loaded.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | The model ID to set as default. |

**Raises:** ValueError: If the model ID is not found.

### `Runner.is_model_loaded`

```text
is_model_loaded(model_id: str) -> bool
```

Check if a model is currently loaded in memory.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | The model ID to check. |

**Returns:** True if the model is loaded, False otherwise.

### `Runner.unload_model`

```text
unload_model(model_id: str) -> None
```

Unload a model from memory while preserving its configuration for later reload.
The model can be reloaded manually with reload_model() or automatically when
a request is sent to it.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | The model ID to unload. |

### `Runner.reload_model`

```text
reload_model(model_id: str) -> None
```

Manually reload a previously unloaded model.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | The model ID to reload. |

### `Runner.load_lora_adapter`

```text
load_lora_adapter(
    alias: str,
    adapter_dir: str | PathLike[str],
    model_id: str | None = None,
    load_inplace: bool = False,
    expected_generation: str | None = None,
) -> LoraAdapterInfo
```

Load a LoRA adapter from a local directory under the name `alias`.

By default this fails if `alias` is already loaded. With `load_inplace=True`, it replaces the
alias instead. Requests already running keep the adapter version they started with.

To avoid overwriting someone else's change, also pass `expected_generation`, the generation you
last saw for the alias. The replace then happens only if the alias still points at it. If it was
changed first, the call fails with code `lora_generation_mismatch` and nothing changes.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `alias` | `str` | required | Name that requests use to select the adapter. |
| `adapter_dir` | `str \| PathLike[str]` | required | Directory with `adapter_config.json` and `adapter_model.safetensors`. |
| `model_id` | `str \| None` | `None` | Model to load it into. If None, uses the default model. |
| `load_inplace` | `bool` | `False` | Replace `alias` if it's already loaded. |
| `expected_generation` | `str \| None` | `None` | Replace only if the alias still points at this generation. Requires `load_inplace=True`. |

**Returns:** The alias and its new generation.

**Raises:** LoraAdapterError: The load failed. `code` says why.

### `Runner.unload_lora_adapter`

```text
unload_lora_adapter(
    alias: str,
    model_id: str | None = None,
    expected_generation: str | None = None,
) -> LoraAdapterInfo
```

Remove `alias`. Requests already running keep using its adapter until they finish.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `alias` | `str` | required | Alias to remove. |
| `model_id` | `str \| None` | `None` | Model to remove it from. If None, uses the default model. |
| `expected_generation` | `str \| None` | `None` | Remove only if the alias still points at this generation. If it was changed first, the call fails with code `lora_generation_mismatch`. |

**Returns:** The removed alias.

**Raises:** LoraAdapterError: The unload failed. `code` says why.

### `Runner.list_lora_adapters`

```text
list_lora_adapters(
    model_id: str | None = None,
) -> list[LoraAdapterInfo]
```

List the loaded adapter aliases for a model.

**Raises:** LoraAdapterError: The call failed, for example because the model has no LoRA runtime. `code` says why.

### `Runner.lora_adapter_status`

```text
lora_adapter_status(model_id: str | None = None) -> LoraRuntimeStatus
```

Report loaded aliases, adapter generations in memory, and the adapter limits for a model.

**Raises:** LoraAdapterError: The call failed, for example because the model has no LoRA runtime. `code` says why.

### `Runner.list_models_with_status`

```text
list_models_with_status() -> list[tuple[str, str]]
```

List all models with their current status.

**Returns:** A list of (model_id, status) tuples where status is one of:
- "loaded": Model is loaded and ready
- "unloaded": Model is unloaded but can be reloaded
- "reloading": Model is currently being reloaded

### `Runner.list_unloaded_models`

```text
list_unloaded_models() -> list[str]
```

List model IDs that are currently unloaded (but can be reloaded).

### `Runner.get_model_status`

```text
get_model_status(model_id: str) -> str | None
```

Get the status of a model: "loaded", "unloaded", "reloading", or None if not found.

### `Runner.remove_model`

```text
remove_model(model_id: str) -> None
```

Remove a model by ID in multi-model mode.

### `Runner.send_chat_completion_request_to_model`

```text
send_chat_completion_request_to_model(
    request: ChatCompletionRequest,
    model_id: str,
) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]
```

Send a chat completion request to a specific model, returning the response
object or a generator over chunk objects.

### `Runner.send_completion_request_to_model`

```text
send_completion_request_to_model(
    request: CompletionRequest,
    model_id: str,
) -> CompletionResponse
```

Send a completion request to a specific model.

### `Runner.export_session`

```text
export_session(
    session_id: str,
    model_id: str | None = None,
) -> str | None
```

Export an agentic session by ID as a JSON string.

Returns None if the session does not exist.

### `Runner.import_session`

```text
import_session(
    session_id: str,
    session_json: str,
    model_id: str | None = None,
) -> None
```

Import an agentic session from a JSON string.

Replaces any existing session with the same ID.

### `Runner.delete_session`

```text
delete_session(session_id: str, model_id: str | None = None) -> bool
```

Delete an agentic session. Returns whether the session existed.

### `Runner.list_session_ids`

```text
list_session_ids(model_id: str | None = None) -> list[str]
```

List all stored agentic session IDs.

### `Runner.find_file`

```text
find_file(file_id: str) -> File | None
```

Look up a file by ID and return it with its full body, even if the response left the body out.
Returns None if no file has that ID. Files are kept for 30 minutes.


## `CalibrationStatus`

| Field | Type |
| --- | --- |
| `collecting` | `bool` |
| `layers` | `int` |
| `layers_tracking` | `int` |
| `total_rows` | `int` |
| `min_rows` | `int` |
| `max_rows` | `int` |


## `LoraAdapterError`

Raised when a dynamic LoRA operation fails. See the
[error codes](/guides/customize/lora-adapters/#troubleshooting).

| Field | Type | Description |
| --- | --- | --- |
| `code` | `str` | Why it failed, such as `lora_generation_mismatch` or `lora_adapter_not_found`. |


## `LoraAdapterInfo`

A loaded adapter alias and the generation it points to now. A generation is one exact set of
adapter files, identified by a hash of its config and weights. See `LoraAdapterGeneration`.

| Field | Type | Description |
| --- | --- | --- |
| `alias` | `str` | Name that requests use to select the adapter. |
| `source` | `str` | Repo or directory the adapter was loaded from. |
| `revision` | `str \| None` | Hub revision requested at load, if any. |
| `generation` | `str` | 64-character ID of the generation the alias points to now. |
| `rank` | `int` | Highest LoRA rank in the adapter's config. |
| `bytes` | `int` | Memory used by this generation's adapter tensors, in bytes. |

### `LoraAdapterInfo.exact`

```text
exact() -> LoraAdapterGeneration
```

Return a selector for this exact generation, to pass as a request's `adapter`.


## `LoraResidentGenerationInfo`

One adapter generation held in memory. When an alias is replaced or unloaded, its old generation
is retired: no alias points to it, but it stays in memory until the requests using it finish.

| Field | Type | Description |
| --- | --- | --- |
| `generation` | `str` | 64-character generation ID. |
| `aliases` | `list[str]` | Aliases that point to this generation now. |
| `rank` | `int` | Highest LoRA rank in the adapter's config. |
| `bytes` | `int` | Memory used by this generation's adapter tensors, in bytes. |
| `retired` | `bool` | True if no alias points to it and it stays only for requests still using it. |
| `active_leases` | `int` | How many in-flight requests are using it. |


## `LoraRuntimeStatus`

Adapter memory use and limits for one model.

| Field | Type | Description |
| --- | --- | --- |
| `adapters` | `list[LoraAdapterInfo]` | Loaded aliases. |
| `generations` | `list[LoraResidentGenerationInfo]` | Every generation in memory, including retired ones. |
| `resident_generations` | `int` | Number of generations in memory. |
| `retired_generations` | `int` | How many of those are retired and kept only for running requests. |
| `resident_bytes` | `int` | Memory used by all adapter tensors, in bytes. |
| `max_adapters` | `int` | `max_adapters` limit set on `Which.Lora` or `Which.GGUF`. |
| `max_rank` | `int` | `max_rank` limit set on `Which.Lora` or `Which.GGUF`. |
| `max_bytes` | `int` | `max_bytes` limit set on `Which.Lora` or `Which.GGUF`. |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
