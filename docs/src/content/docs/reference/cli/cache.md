---
title: "mistralrs cache"
description: "Manage the Hugging Face model cache"
sidebar:
  order: 10
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Manage the Hugging Face model cache

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

