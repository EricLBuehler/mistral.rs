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
| [`mistralrs serve`](/reference/cli/serve/) | Start HTTP/MCP server and (optionally) the UI at /ui |
| [`mistralrs run`](/reference/cli/run/) | Run model in interactive mode, or one-shot mode with `-i` |
| [`mistralrs completions`](/reference/cli/completions/) | Generate shell completions |
| [`mistralrs quantize`](/reference/cli/quantize/) | Generate UQFF quantized model file |
| [`mistralrs uqff`](/reference/cli/uqff/) | Inspect, report, or verify UQFF artifacts |
| [`mistralrs doctor`](/reference/cli/doctor/) | Run system diagnostics and environment checks |
| [`mistralrs tune`](/reference/cli/tune/) | Recommend quantization + device mapping for a model. Rejects `--quant auto`; pass `--quant <level>` or `--isq <level>` to bias the recommendation toward a specific quantization target. Adapter options are rejected because adapter memory is not included in the estimate |
| [`mistralrs login`](/reference/cli/login/) | Authenticate with Hugging Face Hub |
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

