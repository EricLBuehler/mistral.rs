---
title: "mistralrs uqff"
description: "Inspect, report, or verify UQFF artifacts"
sidebar:
  order: 6
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Inspect, report, or verify UQFF artifacts

```
mistralrs uqff [OPTIONS] <COMMAND>
```

## mistralrs uqff report

Print or write a UQFF report

```
mistralrs uqff report [OPTIONS] <PATH>
```

| Option | Default | Description |
|---|---|---|
| `<PATH>` | required | UQFF directory or shard path |
| `--write` | `false` | Write uqff_report.json beside the artifacts |
| `--json` | `false` | Print JSON instead of human-readable text |
| `--verbose` | `false` | Include per-layer fallback details |
| `--base-model <BASE_MODEL>` |  | Base model ID to include in a written report |
| `--repo-id <REPO_ID>` |  | Hugging Face repo ID to include in a written report |

## mistralrs uqff verify

Validate UQFF artifact structure

```
mistralrs uqff verify [OPTIONS] <PATH>
```

| Option | Default | Description |
|---|---|---|
| `<PATH>` | required | UQFF directory or shard path |
| `--json` | `false` | Print JSON instead of human-readable text |
| `--strict` | `false` | Fail on missing report/producer metadata or fallback layers |
| `--allow-newer-minor` | `false` | Allow same-major UQFF files with a newer minor version |

## mistralrs uqff inspect

Open a UQFF-aware tensor explorer

```
mistralrs uqff inspect [OPTIONS] <PATH>
```

| Option | Default | Description |
|---|---|---|
| `<PATH>` | required | UQFF directory or shard path |

