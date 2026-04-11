# mistralrs Python SDK

Documentation for the `mistralrs` Python package.

> **Installation:** See [Python Installation](PYTHON_INSTALLATION.md) for installation instructions.

**Table of contents**
- [Quick Start](#quick-start)
- [Model Configuration](#model-configuration)
- [Streaming](#streaming)
- [Structured Output](#structured-output)
- [Tool Calling](#tool-calling)
- [Multimodal Input](#multimodal-input)
- [Embeddings](#embeddings)
- [Multi-Model Support](#multi-model-support)
- [MCP Client](#mcp-client)
- [Configuration Reference](#configuration-reference)

Full API reference: [here](https://ericlbuehler.github.io/mistral.rs/pyo3/mistralrs.html)

## Quick Start

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

## Model Configuration

Models are configured through the `Which` enum. The most common variants:

| Variant | Use case | Key parameters |
|---|---|---|
| `Which.Plain` | HuggingFace models | `model_id`, `arch` (optional) |
| `Which.GGUF` | GGUF quantized models | `quantized_model_id`, `quantized_filename`, `tok_model_id` |
| `Which.MultimodalPlain` | Vision/audio models | `model_id`, `arch` |
| `Which.Embedding` | Embedding models | `model_id`, `arch` |
| `Which.DiffusionPlain` | Image generation | `model_id`, `arch` |
| `Which.Speech` | Speech synthesis | `model_id`, `arch` |

Architecture is auto-detected for most models. Specify `arch` only if auto-detection fails.

**Common Runner options:**

```python
runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",          # ISQ level: "2"-"8", "q4k", "q8_0", etc.
    # pa_gpu_mem=4096,          # PagedAttention GPU memory in MB
    # pa_blk_size=32,           # PagedAttention block size
    # enable_search=True,       # Enable web search
    # mcp_client_config=...,    # MCP client configuration
)
```

**Loading GGUF models:**

```python
runner = Runner(
    which=Which.GGUF(
        tok_model_id="Qwen/Qwen3-0.6B",
        quantized_model_id="unsloth/Qwen3-0.6B-GGUF",
        quantized_filename="Qwen3-0.6B-Q4_K_M.gguf",
    )
)
```

**Loading UQFF models:**

```python
runner = Runner(
    which=Which.Plain(
        model_id="EricB/Phi-3.5-mini-instruct-UQFF",
        from_uqff="phi3.5-mini-instruct-q4k-0.uqff",
    )
)
```

## Streaming

Set `stream=True` on the request and iterate over chunks:

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Tell me a story."}],
        max_tokens=256,
        stream=True,
    )
)
for chunk in res:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

## Structured Output

Constrain the model to produce valid JSON matching a schema. Works with Pydantic models or raw JSON schemas:

```python
import json
from pydantic import BaseModel, Field
from mistralrs import Runner, Which, ChatCompletionRequest

class City(BaseModel):
    name: str = Field(..., description="City name")
    country: str = Field(..., description="Country")
    population: int = Field(..., description="Population")

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Give me info about Paris."}],
        max_tokens=256,
        grammar_type="json_schema",
        grammar=json.dumps(City.model_json_schema()),
    )
)
city = City.model_validate_json(res.choices[0].message.content)
print(f"{city.name}, {city.country}: pop. {city.population}")
```

You can also use `grammar_type="regex"` or `grammar_type="lark"` for other constrained generation patterns.

## Tool Calling

Define tools as JSON schemas and let the model call them:

```python
import json
from mistralrs import Runner, ToolChoice, Which, ChatCompletionRequest

tools = [
    json.dumps({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
            "strict": True,
        },
    })
]

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

# Model generates a tool call
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        max_tokens=256,
        tool_schemas=tools,
        tool_choice=ToolChoice.Auto,
    )
)

tool_called = res.choices[0].message.tool_calls[0].function
print(f"Tool: {tool_called.name}, Args: {tool_called.arguments}")
```

For server-side tool execution (agentic loop), use `tool_callbacks` or `max_tool_rounds`. See the [Agentic Features Guide](AGENTS.md).

## Multimodal Input

Send images, video, or audio using the OpenAI content format:

```python
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="google/gemma-4-E4B-it",
        arch=MultimodalArchitecture.Gemma4,
    ),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/photo.jpg"},
                    },
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

Image URLs can be web URLs, local file paths, or base64-encoded data URIs. Video uses `video_url` and audio uses `audio_url`.

## Embeddings

```python
from mistralrs import EmbeddingArchitecture, EmbeddingRequest, Runner, Which

runner = Runner(
    which=Which.Embedding(
        model_id="google/embeddinggemma-300m",
        arch=EmbeddingArchitecture.EmbeddingGemma,
    )
)

embeddings = runner.send_embedding_request(
    EmbeddingRequest(
        input=["superconductors", "graphene conductivity"],
        truncate_sequence=True,
    )
)
print(len(embeddings), len(embeddings[0]))
```

## Multi-Model Support

Load and manage multiple models with a single `Runner`. All request methods accept an optional `model_id` parameter:

```python
# List available models
models = runner.list_models()

# Send request to a specific model
response = runner.send_chat_completion_request(request, model_id="model-id")

# Model management
runner.unload_model("model-id")     # Free memory
runner.reload_model("model-id")     # Reload on demand
# Or just send a request. Unloaded models auto-reload.
```

For server-based multi-model deployment, see [Multi-Model Support](multi_model/overview.md).

## MCP Client

Connect to external tool servers via the Model Context Protocol:

```python
import mistralrs

mcp_config = mistralrs.McpClientConfigPy(
    servers=[
        mistralrs.McpServerConfigPy(
            name="Filesystem Tools",
            source=mistralrs.McpServerSourcePy.Process(
                command="npx",
                args=["@modelcontextprotocol/server-filesystem", "."],
            ),
        )
    ],
    auto_register_tools=True,
)

runner = mistralrs.Runner(
    which=mistralrs.Which.Plain(model_id="Qwen/Qwen3-4B"),
    mcp_client_config=mcp_config,
)
```

MCP tools are automatically discovered and available to the model. Supports Process, HTTP, and WebSocket transports. See [MCP Client](MCP/client.md) for details.

## Configuration Reference

<details>
<summary>Which enum: full type definitions</summary>

### Architecture for plain models

If you do not specify the architecture, an attempt will be made to use the model's config. If this fails, please raise an issue.

- `Mistral`, `Gemma`, `Mixtral`, `Llama`, `Phi2`, `Phi3`, `Qwen2`, `Gemma2`, `GLM4`, `GLM4MoeLite`, `GLM4Moe`, `Starcoder2`, `Phi3_5MoE`, `DeepseekV2`, `DeepseekV3`, `Qwen3`, `Qwen3Moe`, `SmolLm3`, `GraniteMoeHybrid`, `GptOss`, `Qwen3Next`

### Architecture for multimodal models

- `Phi3V`, `Idefics2`, `LLaVaNext`, `LLaVa`, `VLlama`, `Qwen2VL`, `Idefics3`, `MiniCpmO`, `Phi4MM`, `Qwen2_5VL`, `Gemma3`, `Gemma4`, `Mistral3`, `Llama4`, `Gemma3n`, `Qwen3VL`, `Qwen3VLMoE`, `Qwen3_5`, `Qwen3_5Moe`, `Voxtral`

### Architecture for diffusion models

- `Flux`, `FluxOffloaded`

### Architecture for speech models

- `Dia`

### Architecture for embedding models

- `EmbeddingGemma`, `Qwen3Embedding`

### ISQ Organization

- `Default`
- `MoQE`: if applicable, only quantize MoE experts. [arxiv.org/abs/2310.02410](https://arxiv.org/abs/2310.02410)

### Which variants

```python
class Which(Enum):
    @dataclass
    class Plain:
        model_id: str
        arch: Architecture | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        organization: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None
        calibration_file: str | None = None
        imatrix: str | None = None
        hf_cache_path: str | None = None

    @dataclass
    class GGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None

    @dataclass
    class MultimodalPlain:
        model_id: str
        arch: MultimodalArchitecture
        tokenizer_json: str | None = None
        topology: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        max_edge: int | None = None
        auto_map_params: MultimodalAutoMapParams | None = None
        calibration_file: str | None = None
        imatrix: str | None = None
        hf_cache_path: str | None = None

    @dataclass
    class Embedding:
        model_id: str
        arch: EmbeddingArchitecture | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        hf_cache_path: str | None = None

    @dataclass
    class DiffusionPlain:
        model_id: str
        arch: DiffusionArchitecture
        dtype: ModelDType = ModelDType.Auto

    @dataclass
    class Speech:
        model_id: str
        arch: DiffusionArchitecture
        dac_model_id: str | None = None
        dtype: ModelDType = ModelDType.Auto
```

> Note: `from_uqff` specifies a UQFF path to load from. For sharded models, you only need to specify the first shard (e.g., `q4k-0.uqff`). The remaining shards are auto-discovered.

> Note: `enable_thinking` enables thinking for models that support it.

> Note: `truncate_sequence=True` trims prompts that exceed the model's maximum context length.

Adapter variants (`XLora`, `Lora`, `XLoraGGUF`, `LoraGGUF`, `GGML`, `XLoraGGML`, `LoraGGML`) are also available. See the [full API reference](https://ericlbuehler.github.io/mistral.rs/pyo3/mistralrs.html) for details.

</details>

## Examples

| Category | Examples |
|---|---|
| **Getting Started** | [plain.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/plain.py), [streaming.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/streaming.py), [gguf.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gguf.py) |
| **Structured Output** | [pydantic_schema.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/pydantic_schema.py), [json_schema.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/json_schema.py), [regex.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/regex.py), [lark.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/lark.py) |
| **Tool Calling** | [tool_call.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/tool_call.py), [custom_tool_call.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/custom_tool_call.py), [agentic_tools.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/agentic_tools.py) |
| **Search** | [web_search.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/web_search.py), [custom_search.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/custom_search.py) |
| **MCP** | [mcp_client.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/mcp_client.py) |
| **Quantization** | [isq.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/isq.py), [imatrix.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/imatrix.py), [topology.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/topology.py) |
| **Adapters** | [lora_zephyr.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/lora_zephyr.py), [xlora_zephyr.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/xlora_zephyr.py), [anymoe.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/anymoe.py) |
| **Advanced** | [paged_attention.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/paged_attention.py), [speculative.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/speculative.py), [multi_model_example.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/multi_model_example.py) |
| **Cookbook** | [cookbook.ipynb](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/cookbook.ipynb), [tool_calling.ipynb](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/tool_calling.ipynb) |

Browse all examples: [examples/python/](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/python)
