# `mistralrs` API

These are API docs for the `mistralrs` package.

**Table of contents**
- Full API docs: [here](https://ericlbuehler.github.io/mistral.rs/pyo3/mistralrs.html)
- Docs for the `Which` enum: [here](#which)
- Multi-model support: [here](#multi-model-support)
- MCP Client Configuration: [here](#mcp-client)
- Example: [here](#example)

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

### Architecture for diffusion models
- `Flux`
- `FluxOffloaded`

### Architecture for speech models
- `Dia`

### ISQ Organization
- `Default`
- `MoQE`: if applicable, only quantize MoE experts. https://arxiv.org/abs/2310.02410

> Note: `from_uqff` specified a UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).

> Note: `enable_thinking` enables thinking for models that support the configuration.

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

The `mistralrs` Python API supports running multiple models simultaneously using the `MultiModelRunner` class, enabling you to serve different models and switch between them dynamically.

### Basic Multi-model Usage

```python
import mistralrs

# Create a MultiModelRunner instead of Runner
runner = mistralrs.MultiModelRunner(
    models=[
        {
            "model_id": "llama3-3b",
            "which": mistralrs.Which.Plain(
                model_id="meta-llama/Llama-3.2-3B-Instruct"
            )
        },
        {
            "model_id": "qwen3-4b", 
            "which": mistralrs.Which.Plain(
                model_id="Qwen/Qwen3-4B"
            )
        }
    ],
    default_model_id="meta-llama/Llama-3.2-3B-Instruct"
)

# Send requests to specific models
response_llama = runner.send_chat_completion_request(
    mistralrs.ChatCompletionRequest(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify which model to use
        messages=[{"role": "user", "content": "Hello from Llama!"}],
        max_tokens=100
    )
)

response_qwen = runner.send_chat_completion_request(
    mistralrs.ChatCompletionRequest(
        model="Qwen/Qwen3-4B",  # Use a different model
        messages=[{"role": "user", "content": "Hello from Qwen!"}],
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
runner.set_default_model_id("Qwen/Qwen3-4B")

# Remove a model
runner.remove_model("meta-llama/Llama-3.2-3B-Instruct")
```

### Server Configuration
For server-based multi-model deployment, see the [multi-model documentation](../docs/multi_model/README.md).

## MCP Client

The `mistralrs` Python API now supports Model Context Protocol (MCP) clients, enabling AI assistants to connect to and interact with external tools and resources through standardized server interfaces.

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
        model="ignore",
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