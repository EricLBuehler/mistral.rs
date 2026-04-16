---
title: Auto-tune with mistralrs tune
description: Let the CLI benchmark your hardware and tell you which quantization and batch settings to use.
sidebar:
  order: 2
---

`mistralrs tune` is a utility that loads a model at several quantization levels, runs a short benchmark on each, and prints a table of the results. If you do not feel like picking a quantization format yourself, this is the fastest way to get a sensible answer.

## Basic usage

```bash
mistralrs tune -m google/gemma-4-E4B-it
```

The command does three things:

1. Downloads the model weights (same as any other mistralrs command).
2. Loads the model repeatedly at different ISQ levels (2, 3, 4, 5, 6, 8, plus full precision).
3. Runs a short prompt-processing and generation benchmark at each level.

Each configuration takes a minute or two to measure. The full run takes 10 to 15 minutes for a 4B-class model; larger models take proportionally longer.

The output is a table with one row per configuration. Columns include VRAM used at load, VRAM used including a reasonable KV cache headroom, prompt tokens per second, generation tokens per second, and a quality proxy based on perplexity on a fixed reference text.

## Profiles

The `--profile` flag shapes the benchmark toward a specific goal:

```bash
mistralrs tune --profile quality -m google/gemma-4-E4B-it
```

Three profiles are available:

- `quality`: Leans toward higher bit widths. Recommends whatever preserves output quality best within your memory budget.
- `balanced`: The default. Targets the sweet spot where memory and quality are both reasonable.
- `fast`: Leans toward lower bit widths. Recommends what gives you the most tokens per second, even if it costs some quality.

The underlying benchmark is the same for all three; the profile only changes how the final recommendation is chosen.

## Saving the recommendation

Add `--emit-config` to write the recommended settings to a TOML file:

```bash
mistralrs tune -m google/gemma-4-E4B-it --emit-config gemma.toml
```

You can then start the server with those exact settings:

```bash
mistralrs from-config -f gemma.toml
```

This is the cleanest way to move from "I am experimenting" to "I have a known-good configuration for this hardware" without writing any TOML by hand.

## Machine-readable output

`--json` flips the output from a human-readable table to JSON:

```bash
mistralrs tune -m google/gemma-4-E4B-it --json > results.json
```

The JSON includes every configuration measured, not just the winner. Useful when you want to integrate tune results into an automated deploy pipeline, or when you want to graph the tradeoff curve yourself.

## Limitations

The benchmark uses a fixed-length prompt and a fixed-length generation. Real workloads have variable lengths, and the relative performance of different quantization levels can shift at the margins with very long contexts or very large batch sizes. For most workloads the defaults are representative; for unusual ones you may want to re-measure with your own data.

The quality proxy is a perplexity score on a short reference text. It is a reasonable signal but not a benchmark of output utility. If the difference between two configurations comes down to fractions of a perplexity point, both will feel identical for most things you would use the model for.

## When not to use it

Skip the tune step when:

- You know exactly which format and bit width you want. Passing `--isq 4` is one fewer decision the engine has to make.
- You are using a pre-quantized model (GGUF or a community AWQ/GPTQ). The tune step assumes it is starting from a full-precision checkpoint.
- You are running a small model on a large GPU and memory is clearly not the bottleneck. Any of the higher bit widths will be fine; the tune step is overkill.

For everything else, especially the "I just got a new GPU and I want to know what I can do with it" case, it is a solid first command to run.
