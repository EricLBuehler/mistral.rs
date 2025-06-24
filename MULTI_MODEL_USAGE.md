# Multi-Model Support

> **📖 This documentation has moved to [docs/multi_model/README.md](docs/multi_model/README.md)**
>
> For the complete and up-to-date multi-model documentation, please visit:
> - **[Main documentation](docs/multi_model/README.md)**
> - **[Example configurations](docs/multi_model/)**

## Quick Reference

```bash
# Multi-model server
mistralrs-server --port 1234 multi-model --config config.json --default-model-id llama3-3b

# Send requests to specific models
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-3b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### New Configuration Format

The configuration format has been simplified to use object keys as model names:

```json
{
  "llama3-3b": {
    "Plain": {
      "model_id": "meta-llama/Llama-3.2-3B-Instruct"
    },
    "in_situ_quant": "Q4K"
  },
  "qwen3-4b": {
    "Plain": {
      "model_id": "Qwen/Qwen3-4B"
    }
  }
}
```

See [docs/multi_model/README.md](docs/multi_model/README.md) for complete documentation.