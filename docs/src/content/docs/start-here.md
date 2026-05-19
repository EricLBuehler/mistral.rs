---
title: Start here
description: Choose the right entry point for the task you are building.
---

Use this page to pick the first document to read. Most workflows start with auto-detection and add flags only when the model, hardware, or deployment requires them.

## Choose by task

| If you need to... | Start here | Then read |
|---|---|---|
| Chat with a model on one machine | [Your first model](/mistral.rs/tutorials/01-install-and-run/) | [Pick a quantization method](/mistral.rs/guides/perf/pick-a-quantization/) |
| Verify install, GPU support, or Hugging Face access | [Your first model](/mistral.rs/tutorials/01-install-and-run/) | [Troubleshooting](/mistral.rs/reference/troubleshooting/) |
| Expose an OpenAI-compatible endpoint | [Serve a model as an API](/mistral.rs/tutorials/02-serve-an-api/) | [Configure the HTTP server](/mistral.rs/guides/serve/http-server/) |
| Use the built-in browser UI | [Serve a model as an API](/mistral.rs/tutorials/02-serve-an-api/) | [Use the built-in web UI](/mistral.rs/guides/serve/with-web-ui/) |
| Call mistral.rs from Python in-process | [Call a model from Python](/mistral.rs/tutorials/03-python-sdk/) | [Python API reference](/mistral.rs/reference/python/) |
| Embed mistral.rs in Rust | [Call a model from Rust](/mistral.rs/tutorials/04-rust-sdk/) | [Rust API on docs.rs](https://docs.rs/mistralrs) |
| Build a local agent app with tools, code execution, web search, multimodal inputs, or session state | [Build an agent](/mistral.rs/tutorials/05-build-an-agent/) | [Agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/) |
| Fit a larger model on the same hardware | [Quantize a model](/mistral.rs/tutorials/06-quantize-a-model/) | [Auto-tune with mistralrs tune](/mistral.rs/guides/perf/auto-tune/) |
| Split a model across GPUs or machines | [Performance](/mistral.rs/guides/perf/) | [Split a model across multiple GPUs](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/) |
| Run a server for real traffic | [Run mistralrs in Docker](/mistral.rs/guides/deploy/docker/) | [Production checklist](/mistral.rs/guides/deploy/production-checklist/) |

## Choose by runtime mode

| Mode | Use when | Entry point |
|---|---|---|
| CLI | You want local interactive use, quick tests, or benchmarking. | `mistralrs run`, `mistralrs bench`, `mistralrs tune` |
| HTTP server | You want OpenAI-compatible clients, a web UI, or a process boundary around inference. | `mistralrs serve` |
| Config file | You need repeatable multi-model startup or a deployment config checked into source control. | `mistralrs from-config -f config.toml` |
| Diagnostics | You want to check hardware detection, build features, or Hugging Face connectivity. | `mistralrs doctor` |
| Python package | You want in-process access from Python without running a server. | `mistralrs.Runner` |
| Rust crate | You want inference embedded inside a Rust service. | `mistralrs` crate |

## If unsure

Start with [Your first model](/mistral.rs/tutorials/01-install-and-run/), then [Serve a model as an API](/mistral.rs/tutorials/02-serve-an-api/). Those two pages exercise the default local and server paths and make later choices easier to evaluate.
