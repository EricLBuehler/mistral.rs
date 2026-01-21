# Multi-Model Support in mistralrs-server

The mistralrs-server supports loading and serving multiple models simultaneously, allowing you to switch between different models in the same server instance.

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
mistralrs-server --port 1234 plain -m meta-llama/Llama-3.2-3B-Instruct
```

### Multi-Model Mode
```bash
# Load multiple models from configuration file
mistralrs-server --port 1234 multi-model --config config.json --default-model-id meta-llama/Llama-3.2-3B-Instruct
```

## Configuration File Format

Create a JSON file with model configurations as object keys:

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

### Configuration Structure

- **Object keys** (e.g., `"llama3-3b"`, `"qwen3-4b"`): Organizational labels (for human readability)
- **Actual API identifiers**: Derived automatically from the model path (e.g., `"meta-llama/Llama-3.2-3B-Instruct"`)
- **Model specification**: The model type and configuration (same format as CLI subcommands)
- **Optional fields**:
  - `chat_template`: Custom chat template
  - `jinja_explicit`: JINJA template file
  - `num_device_layers`: Device layer configuration  
  - `in_situ_quant`: In-situ quantization setting

**How API identifiers work:**
- ✅ Object keys are **organizational only** (for config readability)
- ✅ **Real API identifiers** are derived from `model_id` field inside the model spec
- ✅ Use the full model path in API requests (e.g., `"meta-llama/Llama-3.2-3B-Instruct"`)
- ✅ No naming conflicts - each model has its canonical identifier

## API Usage

### Selecting Models in Requests

Use the `model` field in your requests to specify which model to use:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Default Model Behavior

- **Explicit model**: Use the full pipeline name (e.g., `"meta-llama/Llama-3.2-3B-Instruct"`)
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
      "id": "meta-llama/Llama-3.2-3B-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    },
    {
      "id": "Qwen/Qwen3-4B", 
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    }
  ]
}
```

**Note**: The `"default"` model is always listed first and represents the server's default model.

## CLI Arguments

Use the `multi-model` subcommand with these options:

- `--config <PATH>` (required): Path to the JSON configuration file
- `--default-model-id <ID>` (optional): Default model ID for requests that don't specify a model

**New syntax:**
```bash
mistralrs-server [GLOBAL_OPTIONS] multi-model --config <CONFIG> [--default-model-id <ID>]
```

**Legacy syntax (deprecated):**
```bash
mistralrs-server [OPTIONS] --multi-model --multi-model-config <CONFIG> [--default-model-id <ID>]
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
