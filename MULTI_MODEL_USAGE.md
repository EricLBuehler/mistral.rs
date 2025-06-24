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

See [docs/multi_model/README.md](docs/multi_model/README.md) for complete documentation.