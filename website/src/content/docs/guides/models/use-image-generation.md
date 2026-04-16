---
title: Generate images with diffusion models
description: Running FLUX and similar image-generation models through the mistralrs OpenAI-compatible API.
sidebar:
  order: 2
---

mistral.rs handles diffusion models through the same engine that runs language models. The API is the OpenAI image-generation shape (`POST /v1/images/generations`), so any client that knows how to generate images against OpenAI will work unchanged.

Today the main supported model is FLUX. Other diffusion architectures land over time; see the [supported models reference](/mistral.rs/reference/supported-models/) for the current list.

## Running FLUX

```bash
mistralrs serve -m black-forest-labs/FLUX.1-schnell
```

FLUX.1-schnell is the faster, more permissively licensed variant. `FLUX.1-dev` works the same way but has a more restrictive license (accept the Hugging Face license first, same flow as [the Gemma setup](/mistral.rs/tutorials/02-serve-an-api/#accepting-the-gemma-license)).

Generating an image:

```bash
curl http://localhost:1234/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "A photograph of a golden retriever wearing a scarf in autumn leaves.",
    "n": 1,
    "size": "1024x1024"
  }'
```

The response is an OpenAI-shaped JSON object with a `data` array. Each entry has either a `b64_json` field (the image encoded as base64) or a `url` field (a data URL). The default is base64.

## Size options

FLUX accepts a range of resolutions. Common choices:

- `512x512` for quick previews.
- `1024x1024` for the default quality.
- `1024x1536` or `1536x1024` for portrait or landscape aspect ratios.

Sizes that are not multiples of 64 along each axis get rounded to the nearest multiple during generation. The model's output quality degrades at very large sizes (2048+), so stick to the ranges above unless you have a specific reason.

## Sampling steps and guidance

FLUX.1-schnell is designed to work in 4 steps. FLUX.1-dev benefits from more (20-50). The default is a reasonable per-model value.

To override:

```json
{
  "model": "default",
  "prompt": "...",
  "steps": 20,
  "guidance_scale": 3.5
}
```

`steps` is the number of denoising steps. More steps mean better quality and longer wall-clock time. `guidance_scale` controls how strongly the model adheres to the prompt; lower values are more creative, higher values are more literal.

These are mistralrs extensions to the OpenAI shape. They are accepted but ignored by OpenAI's own endpoint, so the same request body is portable across both.

## Memory notes

Diffusion models are memory-hungry. FLUX.1-schnell in BF16 needs around 30 GB of VRAM; at `--isq 4` quantization it fits in 12 GB, which is in reach for most consumer cards.

```bash
mistralrs serve --isq 4 -m black-forest-labs/FLUX.1-schnell
```

Quantization does degrade output quality somewhat for diffusion models (more than for language models, as a rule). If you have the memory, run unquantized. If you do not, 4-bit is usable.

## What to do with the output

The base64 payload is what you typically want to decode and save:

```python
import base64

with open("out.png", "wb") as f:
    f.write(base64.b64decode(result["data"][0]["b64_json"]))
```

Or, with the OpenAI Python client:

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:1234/v1", api_key="unused")
response = client.images.generate(
    model="default",
    prompt="A golden retriever in autumn leaves.",
    n=1,
    size="1024x1024",
)

with open("out.png", "wb") as f:
    f.write(base64.b64decode(response.data[0].b64_json))
```

The same response shape works from the Rust SDK through `Model::generate_image`.
