---
title: "CLI reference"
description: "Subcommands and flags of the mistralrs binary."
sidebar:
  order: 1
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

## Subcommands

| Subcommand | Purpose |
|---|---|
| [`mistralrs serve`](/reference/cli/serve/) | Start an OpenAI-compatible HTTP server, with a web UI at /ui |
| [`mistralrs run`](/reference/cli/run/) | Chat with a model interactively, or send a single prompt with `-i` |
| [`mistralrs completions`](/reference/cli/completions/) | Generate shell completions |
| [`mistralrs quantize`](/reference/cli/quantize/) | Quantize a model and save it as UQFF files |
| [`mistralrs uqff`](/reference/cli/uqff/) | Inspect, report, or verify UQFF artifacts |
| [`mistralrs doctor`](/reference/cli/doctor/) | Run system diagnostics and environment checks |
| [`mistralrs tune`](/reference/cli/tune/) | Recommend a quantization level and device mapping that fit your hardware |
| [`mistralrs download`](/reference/cli/download/) | Download the files needed to run a model without loading it |
| [`mistralrs cache`](/reference/cli/cache/) | Manage the Hugging Face model cache |
| [`mistralrs bench`](/reference/cli/bench/) | Run performance benchmarks for base or LoRA model generation |
| [`mistralrs from-config`](/reference/cli/from-config/) | Run from a full TOML configuration file |
| [`mistralrs update`](/reference/cli/update/) | Update or migrate an install using the installer |
| [`mistralrs uninstall`](/reference/cli/uninstall/) | Remove an installer-managed install |

## Global options

| Option | Default | Description |
|---|---|---|
| `--seed <SEED>` |  | Random seed for reproducibility |
| `-l, --log <LOG>` |  | Log all requests and responses to this file |
| `--token-source <TOKEN_SOURCE>` | `cache` | Token source for Hugging Face authentication. Formats: `literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none` |
| `-v, --verbose` | `0` | Increase logging verbosity. Use -v for debug and -vv for trace-level internals |
