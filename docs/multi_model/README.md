# Multi-Model Support in mistralrs-server

The mistralrs-server supports loading and serving multiple models simultaneously, allowing you to switch between different models in the same server instance.

## Usage

### Single-Model Mode (Default)
```bash
# Traditional usage - loads one model
mistralrs-server --port 1234 plain -m meta-llama/Llama-3.2-3B-Instruct
```

### Multi-Model Mode
```bash
# Load multiple models from configuration file
mistralrs-server --port 1234 multi-model --config config.json --default-model-id "meta-llama/Llama-3.2-3B-Instruct"
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

If no model is specified, the default model (set with `--default-model-id`) will be used.

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
    "Gguf": {
      "tok_model_id": "meta-llama/Llama-3.2-3B-Instruct",
      "quantized_model_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
      "quantized_filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    }
  }
}
```

## Notes

- Each model runs in its own engine thread
- Models can have different configurations (quantization, device layers, etc.)
- Memory usage scales with the number of loaded models
- All models share the same server configuration (port, logging, etc.)
- Interactive mode uses the default model or the first model if no default is set