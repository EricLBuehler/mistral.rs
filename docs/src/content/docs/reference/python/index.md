---
title: Python API
description: "The mistralrs Python package."
sidebar:
  order: 6
---

The `mistralrs` Python package exposes the same engine that powers the `mistralrs` CLI.

## Install

`pip install mistralrs` covers CPU (Linux, Windows) and Metal (macOS arm64). CUDA wheels are GitHub release assets with `+cudaNNN.smNN` versions. See [Python SDK getting started](/mistral.rs/guides/python/getting-started/#installing) for install commands and [hardware support](/mistral.rs/reference/hardware-support/) for compute capabilities.

```bash
pip install mistralrs                                   # CPU / Metal (PyPI)
pip install "mistralrs==0.8.21+cuda128.sm89" \          # NVIDIA (replace version, CUDA level, and SM)
  --find-links https://github.com/EricLBuehler/mistral.rs/releases/expanded_assets/v0.8.21
```

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
| [Code and shell execution](/mistral.rs/reference/python/code-execution/) | Configuration for the built-in Python and shell executors. |
| [Agent approvals](/mistral.rs/reference/python/agent-approvals/) | Request and decision types for agent action approval callbacks. |
| [Files](/mistral.rs/reference/python/files/) | Input files and first-class output files surfaced from agentic runs. |
| [MCP](/mistral.rs/reference/python/mcp/) | MCP client configuration types. |
| [Auto-mapping](/mistral.rs/reference/python/automap/) | Hints for automatic device mapping. |

See [Python getting started](/mistral.rs/guides/python/getting-started/) for a walkthrough and the [Python guides](/mistral.rs/guides/python/) for task-oriented recipes.

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
