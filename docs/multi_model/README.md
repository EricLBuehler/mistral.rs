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
mistralrs-server --port 1234 multi-model --config config.json --default-model-id llama3-3b
```

## Configuration File Format

Create a JSON file with an array of model configurations:

```json
[
  {
    "model_id": "llama3-3b",
    "model": {
      "Plain": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct"
      }
    }
  },
  {
    "model_id": "qwen3-4b",
    "model": {
      "Plain": {
        "model_id": "Qwen/Qwen3-4B"
      }
    },
    "in_situ_quant": "4"
  }
]
```

### Configuration Fields

- **model_id**: Unique identifier for the model (used in API requests)
- **model**: The model specification (same format as CLI subcommands)
- **chat_template**: Optional custom chat template
- **jinja_explicit**: Optional JINJA template file
- **num_device_layers**: Optional device layer configuration  
- **in_situ_quant**: Optional in-situ quantization setting

**Note**: All fields within the `model` object have sensible defaults and can be omitted. Only `model_id` is required for most model types. The configuration supports the full range of CLI options but with automatic defaults.

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
[
  {
    "model_id": "llama3-3b",
    "model": {
      "Plain": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct"
      }
    }
  },
  {
    "model_id": "qwen3-4b", 
    "model": {
      "Plain": {
        "model_id": "Qwen/Qwen3-4B"
      }
    },
    "in_situ_quant": "4"
  }
]
```

### Example 2: Mixed Model Types
```json
[
  {
    "model_id": "text-model",
    "model": {
      "Plain": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct"
      }
    }
  },
  {
    "model_id": "vision-model",
    "model": {
      "VisionPlain": {
        "model_id": "google/gemma-3-4b-it"
      }
    }
  }
]
```

### Example 3: GGUF Models
```json
[
  {
    "model_id": "llama-gguf",
    "model": {
      "Gguf": {
        "tok_model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "quantized_model_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "quantized_filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
      }
    }
  }
]
```

## Notes

- Each model runs in its own engine thread
- Models can have different configurations (quantization, device layers, etc.)
- Memory usage scales with the number of loaded models
- All models share the same server configuration (port, logging, etc.)
- Interactive mode uses the default model or the first model if no default is set