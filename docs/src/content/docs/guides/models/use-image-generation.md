---
title: Generate images with diffusion models
description: Running FLUX and similar image-generation models through the mistralrs OpenAI-compatible API.
sidebar:
  order: 3
---

mistral.rs serves diffusion models through `POST /v1/images/generations`. The main supported model is FLUX; see the [supported models reference](/mistral.rs/reference/supported-models/).

## Running FLUX

```bash
mistralrs serve -m black-forest-labs/FLUX.1-schnell
```

`FLUX.1-schnell` is permissively licensed. `FLUX.1-dev` requires Hugging Face license acceptance, same flow as [the Gemma setup](/mistral.rs/tutorials/02-serve-an-api/#accepting-the-gemma-license).

For low-memory hosts, use the offloaded architecture. It keeps far less on the GPU at the cost of much slower generation:

| Loader | GPU memory target | Notes |
|---|---|---|
| `Flux` | about 33 GB | Fully loaded path. Fastest. |
| `FluxOffloaded` | about 4 GB | CPU offload path. Useful when the full model does not fit. |

The model is roughly 12B parameters, and the T5 XXL text encoder adds a large memory footprint. Diffusion models do not support ISQ.

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

## Python SDK

```python
from mistralrs import (
    DiffusionArchitecture,
    ImageGenerationResponseFormat,
    Runner,
    Which,
)

runner = Runner(
    which=Which.DiffusionPlain(
        model_id="black-forest-labs/FLUX.1-schnell",
        arch=DiffusionArchitecture.FluxOffloaded,
    )
)

response = runner.generate_image(
    "A vibrant sunset in the mountains, high quality.",
    ImageGenerationResponseFormat.Url,
)
print(response.data[0].url)
```

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
