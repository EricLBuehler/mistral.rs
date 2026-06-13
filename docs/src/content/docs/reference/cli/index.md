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
| [`mistralrs serve`](/mistral.rs/reference/cli/serve/) | Start HTTP/MCP server and (optionally) the UI at /ui |
| [`mistralrs run`](/mistral.rs/reference/cli/run/) | Run model in interactive mode, or one-shot mode with `-i` |
| [`mistralrs completions`](/mistral.rs/reference/cli/completions/) | Generate shell completions |
| [`mistralrs quantize`](/mistral.rs/reference/cli/quantize/) | Generate UQFF quantized model file |
| [`mistralrs doctor`](/mistral.rs/reference/cli/doctor/) | Run system diagnostics and environment checks |
| [`mistralrs tune`](/mistral.rs/reference/cli/tune/) | Recommend quantization + device mapping for a model. Rejects `--quant auto`; pass `--quant <level>` or `--isq <level>` to bias the recommendation toward a specific quantization target |
| [`mistralrs login`](/mistral.rs/reference/cli/login/) | Authenticate with HuggingFace Hub |
| [`mistralrs cache`](/mistral.rs/reference/cli/cache/) | Manage the HuggingFace model cache |
| [`mistralrs bench`](/mistral.rs/reference/cli/bench/) | Run performance benchmarks for plain model generation |
| [`mistralrs from-config`](/mistral.rs/reference/cli/from-config/) | Run from a full TOML configuration file |
| [`mistralrs update`](/mistral.rs/reference/cli/update/) | Update a prebuilt install to the latest release |
| [`mistralrs uninstall`](/mistral.rs/reference/cli/uninstall/) | Remove a prebuilt install |

## Global options

| Option | Default | Description |
|---|---|---|
| `--seed <SEED>` |  | Random seed for reproducibility |
| `-l, --log <LOG>` |  | Log all requests and responses to this file |
| `--token-source <TOKEN_SOURCE>` | `cache` | Token source for HuggingFace authentication. Formats: `literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none` |
| `-v, --verbose` | `0` | Increase logging verbosity. Use -v for debug and -vv for trace-level internals |

