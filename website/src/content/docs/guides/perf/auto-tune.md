---
title: Auto-tune with mistralrs tune
description: Let the CLI benchmark your hardware and tell you which quantization and batch settings to use.
sidebar:
  order: 2
---

`mistralrs tune` loads a model at several quantization levels, benchmarks each, and prints a results table.

## Basic usage

```bash
mistralrs tune -m google/gemma-4-E4B-it
```

The command:

1. Downloads the model weights.
2. Loads the model at multiple ISQ levels (2, 3, 4, 5, 6, 8, plus full precision).
3. Runs a short prompt-processing and generation benchmark at each level.

Each configuration takes a minute or two. A full run is 10–15 minutes on a 4B-class model; larger models scale up.

The output table has one row per configuration. Columns: VRAM at load, VRAM with KV cache headroom, prompt tokens/sec, generation tokens/sec, and a perplexity-based quality proxy.

## Profiles

`--profile` shapes the recommendation:

```bash
mistralrs tune --profile quality -m google/gemma-4-E4B-it
```

- `quality` — leans toward higher bit widths within the memory budget.
- `balanced` — default. Targets the memory/quality middle.
- `fast` — leans toward lower bit widths for higher tokens/sec.

The benchmark itself is identical across profiles; only the recommendation differs.

## Saving the recommendation

`--emit-config` writes the recommended settings to TOML:

```bash
mistralrs tune -m google/gemma-4-E4B-it --emit-config gemma.toml
```

Start the server with those settings:

```bash
mistralrs from-config -f gemma.toml
```

## Machine-readable output

`--json` switches the table to JSON:

```bash
mistralrs tune -m google/gemma-4-E4B-it --json > results.json
```

The JSON includes every measured configuration, not just the recommendation. Use it for automated deploy pipelines or tradeoff curve plotting.

## Limitations

The benchmark uses fixed prompt and generation lengths. Real workloads vary, and relative performance can shift at very long contexts or large batch sizes. Re-measure with workload-specific data when needed.

The quality proxy is perplexity on a short reference text. Reasonable as a signal but not a benchmark of output utility. Differences of fractions of a perplexity point are usually imperceptible.

## When not to use it

Skip tune when:

- The format and bit width are already known. `--isq 4` is one fewer step.
- The model is pre-quantized (GGUF or community AWQ/GPTQ). Tune assumes a full-precision checkpoint.
- A small model on a large GPU is clearly not memory-bound. Any higher bit width works.

For everything else — particularly first use of a new GPU — tune is a solid first command.
