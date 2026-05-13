---
title: Python API
description: "The mistralrs Python package."
sidebar:
  order: 6
---

The `mistralrs` Python package exposes the same engine that powers the `mistralrs` CLI.

## Install

One wheel per accelerator. All wheels expose the same `mistralrs` module.

| Accelerator | Package |
| --- | --- |
| CPU (or Intel CPU with MKL) | `pip install mistralrs` |
| NVIDIA GPU | `pip install mistralrs-cuda` |
| Apple Silicon | `pip install mistralrs-metal` |
| Intel MKL (pinned) | `pip install mistralrs-mkl` |
| macOS Accelerate | `pip install mistralrs-accelerate` |

## Pages

| Page | Covers |
| --- | --- |
| [Runner](/mistral.rs/reference/python/runner/) | The main entry point. Load a model and send requests. |
| [Which](/mistral.rs/reference/python/which/) | Variants that select which kind of model to load. |
| [Requests](/mistral.rs/reference/python/requests/) | Request dataclasses passed to Runner methods. |
| [Responses](/mistral.rs/reference/python/responses/) | Response and streaming types returned by the engine. |
| [Enums](/mistral.rs/reference/python/enums/) | Architecture, dtype, and option enums. |
| [Search](/mistral.rs/reference/python/search/) | Types for web-search tool configuration. |
| [AnyMoE](/mistral.rs/reference/python/anymoe/) | AnyMoE expert and config types. |
| [Code execution](/mistral.rs/reference/python/code-execution/) | Configuration for the built-in Python code executor. |
| [Files](/mistral.rs/reference/python/files/) | First-class output files surfaced from agentic runs. |
| [MCP](/mistral.rs/reference/python/mcp/) | MCP client configuration types. |
| [Auto-mapping](/mistral.rs/reference/python/automap/) | Hints for automatic device mapping. |

See [Tutorial 3](/mistral.rs/tutorials/03-python-sdk/) for a walkthrough and the [Python guides](/mistral.rs/guides/python/) for task-oriented recipes.

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
