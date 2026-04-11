# Gemma 3n Model: [`google/gemma-3n-E4B-it`](https://huggingface.co/google/gemma-3n-E4B-it)

Gemma 3n models are designed for efficient execution on low-resource devices. They are capable of multimodal input, handling text, image, video, and audio input, and generating text outputs. These models support over 140 spoken languages.

Prequantized UQFF models:
- [Gemma 3n E4B](https://huggingface.co/EricB/gemma-3n-E4B-it-UQFF)
- [Gemma 3n E2B](https://huggingface.co/EricB/gemma-3n-E2B-it-UQFF)

## Quick Start

```bash
mistralrs run -m google/gemma-3n-E4B-it --isq 4 --image photo.jpg -i "Describe this image"
```

## Input Formats

The Python and HTTP APIs support sending inputs as:
- **Images**: URL, path to a local file, or base64 encoded string (via `image_url`)
- **Audio**: URL or path to a local file (via `audio_url`)

The Rust SDK takes images from the [image](https://docs.rs/image/latest/image/index.html) crate and audio from `AudioInput`.

## MatFormer dynamic model resizing

Gemma 3n implements the MatFormer architecture, which allows one model to be resized dynamically and tune performance on resource-constrained systems.

You can access it using the `matformer_config_path` ([example config](https://github.com/EricLBuehler/mistral.rs/blob/master/matformer_configs/gemma3n.csv)) and `matformer_slice_name` arguments throughout the APIs. You can read more about MatFormer in mistral.rs [here](MATFORMER.md).

### Available Slices

The default configuration file ([`matformer_configs/gemma3n.csv`](https://github.com/EricLBuehler/mistral.rs/blob/master/matformer_configs/gemma3n.csv)) includes:
- **Main model** (3.98B params, 35 layers) - Full model with best performance
- **Config for official E2B Model** (1.91B params, 30 layers) - Balanced performance/efficiency
- Various intermediate configurations from E1.96B to E3.79B with different layer and FFN configurations

### Choosing the Right Slice

- **Resource-constrained environments**: Use "Config for official E2B Model" (1.91B params)
- **Balanced performance**: Try E2.49B to E2.98B configurations (block-level configs offer better balance)
- **Maximum quality**: Use "Main model" (3.98B params) or omit MatFormer configuration entirely

### Command Line Example

```bash
mistralrs run -m google/gemma-3n-E4B-it \
  --matformer-config-path matformer_configs/gemma3n.csv \
  --matformer-slice-name "Config for E2.49B (block-level)"
```

## HTTP API

1) Start the server

```bash
mistralrs serve -m google/gemma-3n-E4B-it --isq 4 -p 1234
```

Or with MatFormer:

```bash
mistralrs serve -m google/gemma-3n-E4B-it --isq 4 -p 1234 \
  --matformer-config-path matformer_configs/gemma3n.csv \
  --matformer-slice-name "Config for E2.49B (block-level)"
```

2) Send a request

```py
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "What is this?",
                },
            ],
        },
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

**Audio + image example:**

```json
{
  "role": "user",
  "content": [
    {
      "type": "audio_url",
      "audio_url": { "url": "https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg" }
    },
    {
      "type": "image_url",
      "image_url": { "url": "https://www.allaboutbirds.org/guide/assets/og/528129121-1200px.jpg" }
    },
    {
      "type": "text",
      "text": "Describe what is happening in this clip in as much detail as possible."
    }
  ]
}
```

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_local_img.py).

## Python SDK

You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gemma3n.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="google/gemma-3n-E4B-it",
        arch=MultimodalArchitecture.Gemma3n,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is this?",
                    },
                ],
            }
        ],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

**With MatFormer:**

```python
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="google/gemma-3n-E4B-it",
        arch=MultimodalArchitecture.Gemma3n,
        matformer_config_path="matformer_configs/gemma3n.csv",
        matformer_slice_name="Config for E2.49B (block-level)",
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What do you see in this image?",
                    },
                ],
            }
        ],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_local_img.py).

## Rust SDK

You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/multimodal_models/main.rs).

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, MultimodalMessages, MultimodalModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        MultimodalModelBuilder::new("google/gemma-3n-E4B-it")
            .with_isq(IsqType::Q4K)
            .with_logging()
            .build()
            .await?;

    let bytes = match reqwest::blocking::get(
        "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    let messages = MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        "Please describe the image in detail.",
        vec![image],
    );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
```

**With MatFormer and audio:**

```rust
use anyhow::Result;
use mistralrs::{AudioInput, IsqType, TextMessageRole, MultimodalMessages, MultimodalModelBuilder};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let model = MultimodalModelBuilder::new("google/gemma-3n-E4B-it")
        .with_isq(IsqType::Q4K)
        .with_matformer_config_path(PathBuf::from("matformer_configs/gemma3n.csv"))
        .with_matformer_slice_name("Config for E2.49B (block-level)".to_string())
        .with_logging()
        .build()
        .await?;

    let audio_bytes = reqwest::blocking::get(
        "https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg",
    )?
    .bytes()?
    .to_vec();
    let audio = AudioInput::from_bytes(&audio_bytes)?;

    let image_bytes = reqwest::blocking::get(
        "https://www.allaboutbirds.org/guide/assets/og/528129121-1200px.jpg",
    )?
    .bytes()?
    .to_vec();
    let image = image::load_from_memory(&image_bytes)?;

    let messages = MultimodalMessages::new()
        .add_multimodal_message(
            TextMessageRole::User,
            "Describe in detail what is happening.",
            vec![image],
            vec![audio],
            vec![],
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
```
