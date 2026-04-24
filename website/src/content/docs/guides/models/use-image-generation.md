---
title: Generate images with diffusion models
description: Running FLUX and similar image-generation models through the mistralrs OpenAI-compatible API.
sidebar:
  order: 2
---

mistral.rs handles diffusion models through the same engine as language models. The API is the OpenAI image-generation shape (`POST /v1/images/generations`), so any OpenAI-compatible image client works unchanged.

The main supported model is FLUX. Other diffusion architectures land over time; see the [supported models reference](/mistral.rs/reference/supported-models/).

## Running FLUX

```bash
mistralrs serve -m black-forest-labs/FLUX.1-schnell
```

`FLUX.1-schnell` is faster and more permissively licensed. `FLUX.1-dev` works the same way but requires Hugging Face license acceptance, same flow as [the Gemma setup](/mistral.rs/tutorials/02-serve-an-api/#accepting-the-gemma-license).

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

The response is an OpenAI-shaped JSON object with a `data` array. Each entry has either `b64_json` (base64-encoded image) or `url` (data URL). Default is base64.

## Size options

FLUX accepts a range of resolutions:

- `512x512`: quick previews.
- `1024x1024`: default quality.
- `1024x1536` or `1536x1024`: portrait or landscape.

Sizes not divisible by 64 round to the nearest multiple.

## Sampling steps and guidance

FLUX.1-schnell is trained for 4 steps. FLUX.1-dev accepts 20 to 50.

Override:

```json
{
  "model": "default",
  "prompt": "...",
  "steps": 20,
  "guidance_scale": 3.5
}
```

`steps` controls denoising iterations, more steps means better quality and longer wall-clock. `guidance_scale` controls prompt adherence, lower values are more creative, higher values more literal.

These are mistral.rs extensions to the OpenAI shape. OpenAI accepts and ignores them, so the request body is portable.

## Memory notes

FLUX is memory-hungry at native precision. `--isq` reduces the footprint:

```bash
mistralrs serve --isq 4 -m black-forest-labs/FLUX.1-schnell
```

Diffusion output is more sensitive to quantization than language model output.

## Output handling

Decode and save the base64 payload:

```python
import base64

with open("out.png", "wb") as f:
    f.write(base64.b64decode(result["data"][0]["b64_json"]))
```

With the OpenAI Python client:

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

The same response shape works from the Rust SDK via `Model::generate_image`.
