# Multi-Model Support

The `mistralrs` CLI supports loading and serving multiple models simultaneously, allowing you to switch between different models in the same server instance.

- Each model runs in its own engine thread
- Models can have different configurations (quantization, device layers, etc.)
- Memory usage scales with the number of loaded models
- All models share the same server configuration (port, logging, etc.)
- Interactive mode uses the default model or the first model if no default is set
- You can unload all models (including the last one) - they will auto-reload when accessed

## Usage

### Single-Model Mode (Default)
```bash
# Traditional usage - loads one model
mistralrs serve -p 1234 -m meta-llama/Llama-3.2-3B-Instruct
```

### Multi-Model Mode
```bash
# Load multiple models from configuration file
mistralrs from-config --file config.toml
```

## Configuration File Format

Create a JSON file with model configurations as object keys:

```json
{
  "llama3-3b": {
    "alias": "llama3-3b",
    "Plain": {
      "model_id": "meta-llama/Llama-3.2-3B-Instruct"
    }
  },
  "qwen3-4b": {
    "alias": "qwen3-4b",
    "Plain": {
      "model_id": "Qwen/Qwen3-4B"
    },
    "in_situ_quant": "Q4K"
  }
}
```

### Configuration Structure

- **Object keys** (e.g., `"llama3-3b"`, `"qwen3-4b"`): Organizational labels (for human readability)
- **API identifiers**: By default the pipeline name (usually the `model_id` inside the model spec). You can override this with `alias`.
- **Model specification**: The model type and configuration (same format as CLI subcommands)
- **Optional fields**:
  - `alias`: Custom model ID (nickname) used in API requests
  - `chat_template`: Custom chat template
  - `jinja_explicit`: JINJA template file
  - `num_device_layers`: Device layer configuration  
  - `in_situ_quant`: In-situ quantization setting

**How API identifiers work:**
- ✅ Object keys are **organizational only** (for config readability)
- ✅ If `alias` is set, it becomes the API model ID
- ✅ Otherwise, the pipeline name (usually the `model_id` field) is used
- ✅ The canonical pipeline name remains accepted as an alias for compatibility

## API Usage

### Selecting Models in Requests

Use the `model` field in your requests to specify which model to use:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-3b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Default Model Behavior

- **Explicit model**: Use the alias if configured (e.g., `"llama3-3b"`), otherwise the full pipeline name (e.g., `"meta-llama/Llama-3.2-3B-Instruct"`)
- **Default model**: Use `"default"` to explicitly request the default model
- **Auto-fallback**: If the `model` field is omitted entirely, the default model will be used

```bash
# Use default model explicitly
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The default model is either:
1. The model specified with `--default-model-id` when starting the server
2. The first model loaded (if no default is explicitly set)

### List Available Models

```bash
curl http://localhost:1234/v1/models
```

Returns:
```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    },
    {
      "id": "llama3-3b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    },
    {
      "id": "qwen3-4b", 
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    }
  ]
}
```

**Note**: The `"default"` model is always listed first and represents the server's default model. If aliases are configured, they will appear in the list while the canonical pipeline names remain accepted.

## CLI Arguments

Use the `multi-model` subcommand with these options:

- `--config <PATH>` (required): Path to the JSON configuration file
- `--default-model-id <ID>` (optional): Default model ID for requests that don't specify a model (alias or pipeline name)

**New syntax:**
```bash
mistralrs from-config --file <CONFIG>
```

## Examples

### Example 1: Text Models
```json
{
  "llama3-3b": {
    "Plain": {
      "model_id": "meta-llama/Llama-3.2-3B-Instruct"
    }
  },
  "qwen3-4b": {
    "Plain": {
      "model_id": "Qwen/Qwen3-4B"
    },
    "in_situ_quant": "Q4K"
  }
}
```

### Example 2: Mixed Model Types
```json
{
  "text-model": {
    "Plain": {
      "model_id": "meta-llama/Llama-3.2-3B-Instruct"
    }
  },
  "vision-model": {
    "VisionPlain": {
      "model_id": "google/gemma-3-4b-it"
    }
  }
}
```

### Example 3: GGUF Models
```json
{
  "llama-gguf": {
    "GGUF": {
      "tok_model_id": "meta-llama/Llama-3.2-3B-Instruct",
      "quantized_model_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
      "quantized_filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    }
  }
}
```

## Model Unloading and Reloading

You can dynamically unload models to free memory and reload them on demand. This is useful for managing GPU memory when working with multiple large models.

### Unload a Model

Unload a model from memory while preserving its configuration for later reload:

```bash
curl -X POST http://localhost:1234/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.2-3B-Instruct"}'
```

Response:
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "status": "unloaded"
}
```

### Reload a Model

Manually reload a previously unloaded model:

```bash
curl -X POST http://localhost:1234/v1/models/reload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.2-3B-Instruct"}'
```

Response:
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "status": "loaded"
}
```

### Check Model Status

Get the current status of a specific model:

```bash
curl -X POST http://localhost:1234/v1/models/status \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.2-3B-Instruct"}'
```

Response:
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "status": "loaded"
}
```

Possible status values:
- `loaded`: Model is loaded and ready
- `unloaded`: Model is unloaded but can be reloaded
- `reloading`: Model is currently being reloaded
- `not_found`: Model ID not recognized
- `no_loader_config`: Model cannot be reloaded (missing loader configuration)
- `internal_error`: An internal error occurred

### Auto-Reload

When a request is sent to an unloaded model, it will automatically reload before processing the request. This enables a "lazy loading" pattern where models are only loaded when needed.

### List Models with Status

The `/v1/models` endpoint now includes status information:

```bash
curl http://localhost:1234/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    },
    {
      "id": "meta-llama/Llama-3.2-3B-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local",
      "status": "loaded"
    },
    {
      "id": "Qwen/Qwen3-4B",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local",
      "status": "unloaded"
    }
  ]
}
```

## Rust SDK Usage

The `mistralrs` crate provides `MultiModelBuilder` for loading multiple models and `Model` methods for multi-model management.

### Loading Multiple Models

By default, model IDs are the pipeline names (usually the HuggingFace model path, e.g., `"google/gemma-3-4b-it"`). You can provide custom aliases with `add_model_with_alias` for shorter IDs.

```rust
use mistralrs::{IsqType, MultiModelBuilder, TextModelBuilder, VisionModelBuilder, TextMessages, TextMessageRole};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build a multi-model instance with a vision model and a text model
    // Use aliases for shorter model IDs in requests
    let model = MultiModelBuilder::new()
        .add_model_with_alias(
            "gemma-vision",
            VisionModelBuilder::new("google/gemma-3-4b-it")  // Vision model
                .with_isq(IsqType::Q4K)
                .with_logging(),
        )
        .add_model_with_alias(
            "qwen-text",
            TextModelBuilder::new("Qwen/Qwen3-4B")  // Text model
                .with_isq(IsqType::Q4K),
        )
        .with_default_model("gemma-vision")
        .build()
        .await?;

    // Send request to default model
    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Hello!");
    let response = model.send_chat_request(messages).await?;

    // Send request to specific model using its alias
    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Hello from Qwen!");
    let response = model.send_chat_request_with_model(messages, Some("qwen-text")).await?;

    Ok(())
}
```

### Model Management Methods

```rust
// List all models (returns aliases if configured, otherwise pipeline names)
let models = model.list_models()?;

// Get/set default model
let default = model.get_default_model_id()?;
model.set_default_model_id("qwen-text")?;

// List models with status
let status = model.list_models_with_status()?;
// Returns Vec<(String, ModelStatus)> where ModelStatus is Loaded, Unloaded, or Reloading

// Check if a model is loaded
let is_loaded = model.is_model_loaded("gemma-vision")?;

// Unload a model to free memory
model.unload_model("gemma-vision")?;

// Reload when needed
model.reload_model("gemma-vision").await?;
```

### Available `_with_model` Methods

All request methods have `_with_model` variants that accept an optional model ID:

- `send_chat_request_with_model(request, model_id: Option<&str>)`
- `stream_chat_request_with_model(request, model_id: Option<&str>)`
- `generate_image_with_model(..., model_id: Option<&str>)`
- `generate_speech_with_model(prompt, model_id: Option<&str>)`
- `generate_embeddings_with_model(request, model_id: Option<&str>)`
- `tokenize_with_model(..., model_id: Option<&str>)`
- `detokenize_with_model(..., model_id: Option<&str>)`
- `config_with_model(model_id: Option<&str>)`
- `max_sequence_length_with_model(model_id: Option<&str>)`
- `re_isq_model_with_model(isq_type, model_id: Option<&str>)`

When `model_id` is `None`, the default model is used. If aliases are configured, you can pass either the alias or the canonical pipeline name.

## Python SDK Usage

The Python `Runner` class supports multi-model operations directly.

### Basic Usage

```python
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture, Architecture

# Create a runner with a vision model (Gemma 3 4B)
runner = Runner(
    which=Which.VisionPlain(
        model_id="google/gemma-3-4b-it",
        arch=VisionArchitecture.Gemma3,
    ),
    in_situ_quant="Q4K",
)

# Or create a runner with a text model (Qwen3 4B)
# runner = Runner(
#     which=Which.Plain(
#         model_id="Qwen/Qwen3-4B",
#         arch=Architecture.Qwen3,
#     ),
#     in_situ_quant="Q4K",
# )

# List models
models = runner.list_models()
print(f"Available models: {models}")

# Get/set default model
default = runner.get_default_model_id()
runner.set_default_model_id("google/gemma-3-4b-it")

# Send request with specific model_id
request = ChatCompletionRequest(
    messages=[{"role": "user", "content": "Hello!"}]
)
response = runner.send_chat_completion_request(request, model_id=models[0])
```

If aliases are configured (for example via the server config or Rust `MultiModelBuilder`), `list_models()` will return those aliases and you can pass them in `model_id`. The canonical pipeline names remain accepted.

### Model Management

```python
# List models with their status
status = runner.list_models_with_status()
# Returns list of (model_id, status) tuples

# Check if a model is loaded
is_loaded = runner.is_model_loaded("google/gemma-3-4b-it")

# Unload a model to free memory
runner.unload_model("google/gemma-3-4b-it")

# Reload when needed
runner.reload_model("google/gemma-3-4b-it")
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

# Speech generation
audio = runner.generate_audio(prompt, model_id="model-id")

# Tokenization
tokens = runner.tokenize_text(text, add_special_tokens=True, model_id="model-id")
text = runner.detokenize_text(tokens, skip_special_tokens=True, model_id="model-id")
```

When `model_id` is `None` or omitted, the default model is used.

## Migration Guide

### From `MultiModel` (Rust)

The `MultiModel` struct has been removed. Use `Model` directly with `MultiModelBuilder`:

```rust
// Old (deprecated)
let multi = MultiModel::new(...);
multi.send_chat_request_to_model(request, "model-id").await?;

// New - model IDs are pipeline names by default (aliases optional)
let model = MultiModelBuilder::new()
    .add_model(VisionModelBuilder::new("google/gemma-3-4b-it"))
    .add_model(TextModelBuilder::new("Qwen/Qwen3-4B"))
    .build()
    .await?;
model.send_chat_request_with_model(request, Some("Qwen/Qwen3-4B")).await?;
```

### From `MultiModelRunner` (Python)

The `MultiModelRunner` class has been removed. Use `Runner` directly:

```python
# Old (deprecated)
multi_runner = MultiModelRunner(runner)
multi_runner.send_chat_completion_request_to_model(request, "model-id")

# New - model IDs are the registered IDs (aliases if configured)
runner = Runner(which=Which.Plain(model_id="google/gemma-3-4b-it", ...))
runner.send_chat_completion_request(request, model_id="google/gemma-3-4b-it")
```
