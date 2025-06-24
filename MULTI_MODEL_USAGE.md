# Multi-Model Support in mistralrs-server

The mistralrs-server now supports loading and serving multiple models simultaneously, allowing you to switch between different models in the same server instance.

## Usage

### Single-Model Mode (Default)
```bash
# Traditional usage - loads one model
mistralrs-server --port 1234 plain -m microsoft/DialoGPT-medium -a gpt2
```

### Multi-Model Mode
```bash
# Load multiple models from configuration file
mistralrs-server --port 1234 --multi-model --multi-model-config example-multi-model-config.json --default-model-id llama3-8b
```

## Configuration File Format

Create a JSON file with an array of model configurations:

```json
[
  {
    "model_id": "llama3-8b",
    "model": {
      "Plain": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct"
      }
    }
  },
  {
    "model_id": "mistral-7b",
    "model": {
      "Plain": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3"
      }
    },
    "num_device_layers": ["8"],
    "in_situ_quant": "Q4K"
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
    "model": "llama3-8b",
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
      "id": "llama3-8b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    },
    {
      "id": "mistral-7b", 
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    }
  ]
}
```

## CLI Arguments

- `--multi-model`: Enable multi-model mode
- `--multi-model-config <PATH>`: Path to the JSON configuration file
- `--default-model-id <ID>`: Default model ID for requests that don't specify a model

## Examples

### Example 1: Simple Text Models
```json
[
  {
    "model_id": "gpt2-small",
    "model": {
      "Plain": {
        "model_id": "gpt2"
      }
    }
  },
  {
    "model_id": "gpt2-large", 
    "model": {
      "Plain": {
        "model_id": "gpt2-large"
      }
    }
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
        "model_id": "microsoft/DialoGPT-medium"
      }
    }
  },
  {
    "model_id": "vision-model",
    "model": {
      "VisionPlain": {
        "model_id": "microsoft/kosmos-2-patch14-224"
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
        "tok_model_id": null,
        "quantized_model_id": "TheBloke/Llama-2-7b-Chat-GGUF",
        "quantized_filename": "llama-2-7b-chat.Q4_K_M.gguf"
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