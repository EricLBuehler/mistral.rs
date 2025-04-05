# Quickstart

It is easiest to get started with mistral.rs using the CLI. This can function as both either an interactive mode, or a server.

mistral.rs supports loading models directly from the Hugging Face Hub. To do this, we divide models into several categories:
- Plain: text models
- Vision plain: vision models

We also support a selection of quantized models from llama.cpp:
- GGUF
- GGML (not recommended; deprecated)

For most users, it is **recommended to primarily use the "plain" and "vision plain"** categories for ease and access the best feature set.

In the CLI, you can quickly start interactive mode:
```
./mistralrs-server -i --isq q4k plain -m meta-llama/Llama-3.1-8B-Instruct
```

Or, you can easily use a vision model:
```
./mistralrs-server -i --isq q4k vision-plain -m google/gemma-3-4b-it -a gemma3
```

You can start an OpenAI-compatible server listening on the given port by replacing `-i` by `--port <port>`.
