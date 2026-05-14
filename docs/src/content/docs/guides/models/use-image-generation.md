---
title: Generate images with diffusion models
description: Running FLUX and similar image-generation models through the mistralrs OpenAI-compatible API.
sidebar:
  order: 2
---

mistral.rs serves diffusion models through `POST /v1/images/generations`. The main supported model is FLUX; see the [supported models reference](/mistral.rs/reference/supported-models/).

## Running FLUX

```bash
mistralrs serve -m black-forest-labs/FLUX.1-schnell
```

`FLUX.1-schnell` is permissively licensed. `FLUX.1-dev` requires Hugging Face license acceptance, same flow as [the Gemma setup](/mistral.rs/tutorials/02-serve-an-api/#accepting-the-gemma-license).

Generating an image:

```bash
curl http://localhost:1234/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "A photograph of a golden retriever wearing a scarf in autumn leaves.",
    "n": 1,
    "height": 1024,
    "width": 1024
  }'
```

The response is JSON with a `data` array. Each entry has `url` (server-side filename where the PNG was written) or `b64_json` (a `data:image/png;base64,...` data URL string) depending on the `response_format` field. The default is `Url`.

## Request fields

| Field | Default | Notes |
|---|---|---|
| `prompt` | required | Text prompt. |
| `n` | 1 | Number of images. |
| `height` | 720 | Output height in pixels. |
| `width` | 1280 | Output width in pixels. |
| `response_format` | `"Url"` | `"Url"` (response carries a server-side filename in `url`) or `"B64Json"` (response carries a `data:image/png;base64,...` string in `b64_json`). |

`size` (the OpenAI string form) is not supported. Use `height` and `width`.

## Memory notes

FLUX is memory-hungry at native precision. Diffusion models do not support `--isq`; load them at native precision.

## Output handling

With `Url` (the default), the server writes the PNG to disk and returns its filename in `url`:

```python
import shutil
saved = response["data"][0]["url"]
shutil.copy(saved, "out.png")
```

With `B64Json`, `b64_json` is a `data:image/png;base64,...` string. Strip the prefix before decoding:

```python
import base64, re

data_url = response["data"][0]["b64_json"]
payload = re.sub(r"^data:image/\w+;base64,", "", data_url)
with open("out.png", "wb") as f:
    f.write(base64.b64decode(payload))
```

The same endpoint is callable from the Rust SDK via `Model::generate_image`.
