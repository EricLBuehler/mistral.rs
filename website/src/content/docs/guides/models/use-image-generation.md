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

The response is JSON with a `data` array. Each entry has `url` (data URL) or `b64_json` (base64) depending on the `response_format` field. The default is `url`.

## Request fields

| Field | Default | Notes |
|---|---|---|
| `prompt` | required | Text prompt. |
| `n` | 1 | Number of images. |
| `height` | 720 | Output height in pixels. |
| `width` | 1280 | Output width in pixels. |
| `response_format` | `"url"` | `"url"` (data URL) or `"b64_json"` (base64). |

`size` (the OpenAI string form) is not supported. Use `height` and `width`.

## Memory notes

FLUX is memory-hungry at native precision. Diffusion models do not support `--isq`; load them at native precision.

## Output handling

Save a data URL response:

```python
import base64, re

data_url = response["data"][0]["url"]
payload = re.sub(r"^data:image/\w+;base64,", "", data_url)
with open("out.png", "wb") as f:
    f.write(base64.b64decode(payload))
```

With the OpenAI Python client (request `response_format="b64_json"` for direct base64):

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:1234/v1", api_key="unused")
response = client.images.generate(
    model="default",
    prompt="A golden retriever in autumn leaves.",
    n=1,
    response_format="b64_json",
    extra_body={"height": 1024, "width": 1024},
)

with open("out.png", "wb") as f:
    f.write(base64.b64decode(response.data[0].b64_json))
```

The same endpoint is callable from the Rust SDK via `Model::generate_image`.
