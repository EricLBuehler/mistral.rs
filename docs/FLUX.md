# FLUX.1: [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

The FLUX model is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions.

We support both the `-schnell` and `-dev` versions of the model.

## Quick Start

```bash
mistralrs run diffusion -m black-forest-labs/FLUX.1-schnell -a flux
```

## Memory Usage

The FLUX model itself is 12 billion parameters (~24GB), and the T5 XXL encoder model it uses requires ~9GB. We support loading the models fully onto the GPU, which allows much faster inference. If you do not have enough memory, try the offloaded (`-offloaded` or `-Offloaded`) model types. These will load the model on the CPU but perform computations on the GPU.

|Type|Memory requirement|Generation Time (s), A100|
| -- | -- | -- |
|Normal| ~33GB | 9.4 |
|Offloaded| ~4GB | 92.7 |

## HTTP API

Start the server:

```
mistralrs serve diffusion -p 1234 -m black-forest-labs/FLUX.1-schnell -a flux
```

Send a request:

```py
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

result = client.images.generate(
    model="default",
    prompt="A vibrant sunset in the mountains, 4k, high quality.",
    n=1,
)
print(result.data[0].url)
```

## Rust SDK

```rust
use std::time::Instant;
use anyhow::Result;
use mistralrs::{DiffusionGenerationParams, DiffusionLoaderType, DiffusionModelBuilder, ImageGenerationResponseFormat};

#[tokio::main]
async fn main() -> Result<()> {
    let model = DiffusionModelBuilder::new(
        "black-forest-labs/FLUX.1-schnell",
        DiffusionLoaderType::FluxOffloaded,
    )
    .with_logging()
    .build()
    .await?;

    let start = Instant::now();

    let response = model
        .generate_image(
            "A vibrant sunset in the mountains, 4k, high quality.".to_string(),
            ImageGenerationResponseFormat::Url,
            DiffusionGenerationParams::default(),
        )
        .await?;

    let finished = Instant::now();

    println!(
        "Done! Took {} s. Image saved at: {}",
        finished.duration_since(start).as_secs_f32(),
        response.data[0].url.as_ref().unwrap()
    );

    Ok(())
}
```

## Python SDK

```py
from mistralrs import (
    Runner,
    Which,
    DiffusionArchitecture,
    ImageGenerationResponseFormat,
)

runner = Runner(
    which=Which.DiffusionPlain(
        model_id="black-forest-labs/FLUX.1-schnell",
        arch=DiffusionArchitecture.FluxOffloaded,
    ),
)

res = runner.generate_image(
    "A vibrant sunset in the mountains, 4k, high quality.",
    ImageGenerationResponseFormat.Url,
)
print(res.choices[0].url)
```
