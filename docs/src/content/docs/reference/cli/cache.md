---
title: "mistralrs cache"
description: "Manage the HuggingFace model cache"
sidebar:
  order: 9
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Manage the HuggingFace model cache

```
mistralrs cache [OPTIONS] <COMMAND>
```

## mistralrs cache list

List all cached models

```
mistralrs cache list [OPTIONS]
```

## mistralrs cache delete

Delete a specific model from cache

```
mistralrs cache delete [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Model ID (e.g., "Qwen/Qwen3-4B") |

