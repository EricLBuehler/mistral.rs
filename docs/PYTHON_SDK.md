# mistralrs Python SDK

Documentation for the `mistralrs` Python package.

> **Installation:** See [PYTHON_INSTALLATION.md](PYTHON_INSTALLATION.md) for installation instructions.

**Table of contents**
- Full API reference: [here](https://ericlbuehler.github.io/mistral.rs/pyo3/mistralrs.html)
- Model configuration (`Which` enum): [here](#which)
- Multi-model support: [here](#multi-model-support)
- MCP Client Configuration: [here](#mcp-client)
- Example: [here](#example)
- Embeddings example: [here](#embeddings-example)

## `Which`

Each `*_model_id` may be a HF hub repo or a local path. For quantized GGUF models, a list is accepted if multiple files must be specified.

### Architecture for plain models
If you do not specify the architecture, an attempt will be made to use the model's config. If this fails, please raise an issue.

- `Mistral`
- `Gemma`
- `Mixtral`
- `Llama`
- `Phi2`
- `Phi3`
- `Qwen2`
- `Gemma2`
- `GLM4`
- `Starcoder2`
- `Phi3_5MoE`
- `DeepseekV2`
- `DeepseekV3`
- `Qwen3`
- `Qwen3Moe`
- `SmolLm3`
- `GraniteMoeHybrid`
- `GptOss`

### ISQ Organization
- `Default`
- `MoQE`: if applicable, only quantize MoE experts. https://arxiv.org/abs/2310.02410

### Architecture for vision models
- `Phi3V`
- `Idefics2`
- `LLaVaNext`
- `LLaVa`
- `VLlama`
- `Qwen2VL`
- `Idefics3`
- `MiniCpmO`
- `Phi4MM`
- `Qwen2_5VL`
- `Gemma3`
- `Mistral3`
- `Llama4`
- `Gemma3n`
- `Qwen3VL`

### Architecture for diffusion models
- `Flux`
- `FluxOffloaded`

### Architecture for speech models
- `Dia`

### Architecture for embedding models
- `EmbeddingGemma`
- `Qwen3Embedding`

### ISQ Organization
- `Default`
- `MoQE`: if applicable, only quantize MoE experts. https://arxiv.org/abs/2310.02410

> Note: `from_uqff` specifies a UQFF path to load from. If provided, this takes precedence over applying ISQ. For sharded models, you only need to specify the first shard (e.g., `q4k-0.uqff`) -- the remaining shards are auto-discovered. For multiple different quantizations, use a semicolon delimiter (;).

> Note: `enable_thinking` enables thinking for models that support the configuration.
> Note: `truncate_sequence=True` trims prompts that would otherwise exceed the model's maximum context length. Leave it `False` to receive a validation error instead.

```py
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
        auto_map_params: TextAutoMapParams | None = (None,)
        calibration_file: str | None = None
        imatrix: str | None = None
        hf_cache_path: str | None = None

    @dataclass
    class XLora:
        xlora_model_id: str
        order: str
        arch: Architecture | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        tgt_non_granular_index: int | None = None
        topology: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)
        hf_cache_path: str | None = None

    @dataclass
    class Lora:
        adapter_model_id: str
        arch: Architecture | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)
        hf_cache_path: str | None = None

    @dataclass
    class GGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class XLoraGGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        xlora_model_id: str
        order: str
        tok_model_id: str | None = None
        tgt_non_granular_index: int | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class LoraGGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class GGML:
        quantized_model_id: str
        quantized_filename: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        gqa: int | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class XLoraGGML:
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tok_model_id: str | None = None
        tgt_non_granular_index: int | None = None
        tokenizer_json: str | None = None
        gqa: int | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class LoraGGML:
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

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
    class VisionPlain:
        model_id: str
        arch: VisionArchitecture
        tokenizer_json: str | None = None
        topology: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        max_edge: int | None = None
        auto_map_params: VisionAutoMapParams | None = (None,)
        calibration_file: str | None = None
        imatrix: str | None = None
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

## Multi-model Support

The `mistralrs` Python SDK supports running multiple models using the `Runner` class with the `model_id` parameter. All request methods accept an optional `model_id` to target a specific model. When `model_id` is `None` or omitted, the default model is used. If aliases are configured (for example via the server config or Rust `MultiModelBuilder`), `list_models()` will return those aliases and you can pass them in requests; canonical pipeline names remain accepted.

### Basic Usage with model_id

```python
import mistralrs

# Create a Runner with a vision model (Gemma 3 4B)
runner = mistralrs.Runner(
    which=mistralrs.Which.VisionPlain(
        model_id="google/gemma-3-4b-it",
        arch=mistralrs.VisionArchitecture.Gemma3,
    ),
    in_situ_quant="Q4K",
)

# List available models (model IDs are registered IDs, aliases if configured)
models = runner.list_models()
print(f"Available models: {models}")  # ["google/gemma-3-4b-it"]

# Send request to specific model using model_id parameter
response = runner.send_chat_completion_request(
    mistralrs.ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100
    ),
    model_id="google/gemma-3-4b-it"  # Target specific model
)

# Send request without model_id (uses default model)
response = runner.send_chat_completion_request(
    mistralrs.ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100
    )
)
```

### Multi-model Management

```python
# List available models
models = runner.list_models()
print(f"Available models: {models}")

# Get/set default model
default_model = runner.get_default_model_id()
print(f"Default model: {default_model}")

# Change default model (model must be loaded)
runner.set_default_model_id("google/gemma-3-4b-it")

# List models with their status
models_with_status = runner.list_models_with_status()
for model_id, status in models_with_status:
    print(f"{model_id}: {status}")  # status is "loaded", "unloaded", or "reloading"
```

### Model Unloading and Reloading

You can unload models to free memory and reload them on demand:

```python
model_id = "google/gemma-3-4b-it"

# Check if model is loaded
is_loaded = runner.is_model_loaded(model_id)
print(f"Model loaded: {is_loaded}")

# List models with their status
models_with_status = runner.list_models_with_status()
for mid, status in models_with_status:
    print(f"{mid}: {status}")

# Unload a model to free memory (preserves configuration for reload)
runner.unload_model(model_id)

# Check status after unload
is_loaded = runner.is_model_loaded(model_id)
print(f"Model loaded after unload: {is_loaded}")  # False

# Manually reload a model
runner.reload_model(model_id)

# Auto-reload: sending a request to an unloaded model will reload it automatically
response = runner.send_chat_completion_request(
    mistralrs.ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello!"}]
    ),
    model_id=model_id  # Will auto-reload if unloaded
)
```

### Request Methods with model_id

All request methods accept an optional `model_id` parameter:

```python
# Chat completion
response = runner.send_chat_completion_request(request, model_id="model-id")

# Completion
response = runner.send_completion_request(request, model_id="model-id")

# Embeddings
embeddings = runner.send_embedding_request(request, model_id="model-id")

# Image generation
image = runner.generate_image(prompt, response_format, model_id="model-id")

# Audio generation
audio = runner.generate_audio(prompt, model_id="model-id")

# Tokenization
tokens = runner.tokenize_text(text, add_special_tokens=True, model_id="model-id")
text = runner.detokenize_text(tokens, skip_special_tokens=True, model_id="model-id")
```

When `model_id` is `None` or omitted, the default model is used.

### Server Configuration
For server-based multi-model deployment, see the [multi-model documentation](multi_model/overview.md).

## MCP Client

The `mistralrs` Python SDK now supports Model Context Protocol (MCP) clients, enabling AI assistants to connect to and interact with external tools and resources through standardized server interfaces.

### MCP Server Configuration

Configure MCP servers using `McpServerConfigPy`:

```python
# HTTP-based MCP server with Bearer token authentication
http_server = mistralrs.McpServerConfigPy(
    id="web_search",
    name="Web Search MCP",
    source=mistralrs.McpServerSourcePy.Http(
        url="https://api.example.com/mcp",
        timeout_secs=30,
        headers={"X-API-Version": "v1"}  # Optional additional headers
    ),
    enabled=True,
    tool_prefix="web",  # Prefixes tool names to avoid conflicts
    resources=None,
    bearer_token="your-api-token"  # Automatically added as Authorization header
)

# Process-based MCP server for local tools
process_server = mistralrs.McpServerConfigPy(
    id="filesystem",
    name="Filesystem MCP",
    source=mistralrs.McpServerSourcePy.Process(
        command="mcp-server-filesystem",
        args=["--root", "/tmp"],
        work_dir=None,
        env={"MCP_LOG_LEVEL": "debug"}  # Optional environment variables
    ),
    enabled=True,
    tool_prefix="fs",
    resources=["file://**"],  # Resource patterns this client is interested in
    bearer_token=None  # Process servers typically don't need authentication
)

# WebSocket-based MCP server for real-time communication
websocket_server = mistralrs.McpServerConfigPy(
    id="realtime_data",
    name="Real-time Data MCP",
    source=mistralrs.McpServerSourcePy.WebSocket(
        url="wss://realtime.example.com/mcp",
        timeout_secs=60,
        headers=None
    ),
    enabled=True,
    tool_prefix="rt",
    resources=None,
    bearer_token="websocket-token"  # WebSocket Bearer token support
)
```

### MCP Client Configuration

Configure the MCP client using `McpClientConfigPy`:

```python
mcp_config = mistralrs.McpClientConfigPy(
    servers=[http_server, process_server, websocket_server],
    auto_register_tools=True,  # Automatically discover and register tools
    tool_timeout_secs=30,      # Timeout for individual tool calls
    max_concurrent_calls=5     # Maximum concurrent tool calls across all servers
)
```

### Integration with Runner

Pass the MCP client configuration to the `Runner`:

```python
runner = mistralrs.Runner(
    which=mistralrs.Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    ),
    mcp_client_config=mcp_config  # MCP tools automatically registered
)
```

When `auto_register_tools=True`, the MCP client will:
1. Connect to all enabled MCP servers
2. Discover available tools from each server
3. Register them for automatic tool calling with appropriate prefixes
4. Make them available during model conversations

### MCP Transport Types

- **HTTP Transport**: Best for public APIs, RESTful services, servers behind load balancers. Supports SSE (Server-Sent Events) and standard HTTP semantics.

- **Process Transport**: Best for local tools, development servers, sandboxed environments. Provides process isolation with no network overhead.

- **WebSocket Transport**: Best for interactive applications, real-time data, low-latency requirements. Supports persistent connections and server-initiated notifications.

### Authentication

- **Bearer Tokens**: Automatically added as `Authorization: Bearer <token>` header for HTTP and WebSocket connections
- **Custom Headers**: Additional headers can be specified for API keys, versioning, etc.
- **Process Servers**: Typically don't require authentication as they run locally

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
        model="default",
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

## Embeddings example

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
        input=[
            "task: query | text: superconductors",
            "task: query | text: graphene",
        ],
        truncate_sequence=True,
    )
)

print(len(embeddings), len(embeddings[0]))

# Swap the model_id and arch below to load Qwen/Qwen3-Embedding-0.6B instead:
# Runner(
#     which=Which.Embedding(
#         model_id="Qwen/Qwen3-Embedding-0.6B",
#         arch=EmbeddingArchitecture.Qwen3Embedding,
#     )
# )
```
