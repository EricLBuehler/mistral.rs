---
title: Auto-tune with mistralrs tune
description: Recommend a quantization and device-mapping configuration for the host.
sidebar:
  order: 2
---

`mistralrs tune` recommends a quantization and device-mapping configuration for the host.

## Basic usage

```bash
mistralrs tune -m google/gemma-4-E4B-it
```

Output is a table with columns: `Quant | Est. Size | VRAM % | Context Room | Quality | Status`. The status column marks one row as `🚀 Recommended`; other rows are marked `✅ Fits`, `⚠️ Hybrid`, or `❌ Too Large`.

Quality tiers: `Baseline`, `Near-lossless`, `Good`, `Acceptable`, `Degraded`.

A recommended command line is printed below the table.

## Profiles

```bash
mistralrs tune --profile quality -m google/gemma-4-E4B-it
```

`--profile` accepts `quality`, `balanced` (default), or `fast`.

## Saving the recommendation

```bash
mistralrs tune -m google/gemma-4-E4B-it --emit-config gemma.toml
```

Run with the recommended settings:

```bash
mistralrs from-config -f gemma.toml
```

## Machine-readable output

```bash
mistralrs tune -m google/gemma-4-E4B-it --json > results.json
```
