# Gemma 3n Model: [`google/gemma-3n-E4B-it`](https://huggingface.co/google/gemma-3n-E4B-it)

Gemma 3n models are designed for efficient execution on low-resource devices. They are capable of multimodal input, handling text, image, video, and audio input, and generating text outputs. These models support over 140 spoken languages.

The Gemma 3n Model has support in the Rust, Python, and HTTP APIs. Additionally, the Gemma 3n Model supports ISQ for increased performance.

- **Full multimodal support**: mistral.rs supports text, audio, and vision inputs to Gemma 3n!

- **🪆 mistral.rs supports dynamically resizing the Gemma 3n model with that MatFormer architecture!**

    Gemma 3n implements the MatFormer architecture, which allows one model to be resized dynamically and tune performance on resource-constrained systems.

    Mistral.rs supports this feature!
    
    You can access it using the `matformer_config_path` ([example config](https://github.com/EricLBuehler/mistral.rs/blob/master/matformer_configs/gemma3n.csv)) and `matformer_slice_name` arguments throughout the APIs.
    
- **Prequantized UQFF models:**
  - [Gemma 3n E4B](https://huggingface.co/EricB/gemma-3n-E4B-it-UQFF)
  - [Gemma 3n E2B](https://huggingface.co/EricB/gemma-3n-E2B-it-UQFF)

## Using MatFormer with Gemma 3n

MatFormer allows you to dynamically adjust the model size based on your resource constraints. The Gemma 3n model comes with several pre-configured slices that offer different performance/resource trade-offs.

You can read more about MatFormer in mistral.rs [here](MATFORMER.md).

### Available Slices

The default configuration file ([`matformer_configs/gemma3n.csv`](https://github.com/EricLBuehler/mistral.rs/blob/master/matformer_configs/gemma3n.csv)) includes:
- **Main model** (3.98B params, 35 layers) - Full model with best performance
- **Config for official E2B Model** (1.91B params, 30 layers) - Balanced performance/efficiency  
- Various intermediate configurations from E1.96B to E3.79B with different layer and FFN configurations

### Command Line Example

```bash
# Run with the E2.49B slice for balanced performance/efficiency
mistralrs run vision -m google/gemma-3n-E4B-it \
  --matformer-config-path matformer_configs/gemma3n.csv \
  --matformer-slice-name "Config for E2.49B (block-level)"
```

### Python SDK Example

```python
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

# Use the E2.49B slice for balanced performance/efficiency
runner = Runner(
    which=Which.VisionPlain(
        model_id="google/gemma-3n-E4B-it",
        arch=VisionArchitecture.Gemma3n,
        matformer_config_path="matformer_configs/gemma3n.csv",
        matformer_slice_name="Config for E2.49B (block-level)",
    ),
)

# The model will use 35 layers with mixed FFN dimensions (4096 for early layers, 8192 for middle)
# This results in ~37% parameter reduction while maintaining better performance than E2B
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="ignore",
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
        max_tokens=100,
    )
)
print(res.choices[0].message.content)
```

### Rust SDK Example

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Build model with MatFormer E2.49B configuration
    let model = VisionModelBuilder::new("google/gemma-3n-E4B-it")
        .with_isq(IsqType::Q4K)
        .with_matformer_config_path(PathBuf::from("matformer_configs/gemma3n.csv"))
        .with_matformer_slice_name("Config for E2.49B (block-level)".to_string())
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

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "Describe this image briefly.",
        vec![image],
    );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    println!("Using E2.49B slice: 35 layers, 2.49B effective params");
    
    Ok(())
}
```

### Choosing the Right Slice

- **Resource-constrained environments**: Use "Config for official E2B Model" (1.91B params)
- **Balanced performance**: Try E2.49B to E2.98B configurations (block-level configs offer better balance)
- **Maximum quality**: Use "Main model" (3.98B params) or omit MatFormer configuration entirely

The slice selection allows you to:
- Reduce memory usage proportionally to the parameter count
- Speed up inference roughly linearly with the number of layers
- Maintain acceptable quality for many use cases with smaller slices

## HTTP server
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/gemma3n.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg" alt="Mount Washington" width = "1000" height = "666">
<h6><a href = "https://www.nhmagazine.com/mount-washington/">Credit</a></h6>

**Prompt:**
```
Please describe this image in detail.
```

**Output:**
```
The image captures a breathtaking, wide-angle view of a majestic mountain covered in a blanket of snow. The mountain dominates the frame, its peak reaching towards a partly cloudy sky. The snow cover is uneven, with patches of exposed dark rock and textured snow formations creating a visually interesting surface. 

A winding, snow-covered path or road snakes its way up the mountainside, appearing as a bright white line against the darker slopes. This path draws the eye upwards towards the summit, where a few structures, possibly communication towers or observation points, are visible. 

The lower slopes of the mountain are covered in a dense forest of evergreen trees, their dark green hues contrasting beautifully with the white snow. The forest extends down into a valley, hinting at a wider landscape beyond the frame. 

The sky above is a mix of pale blue and soft grey clouds, with some darker, more dramatic cloud formations near the top of the mountain. The lighting suggests it might be early morning or late afternoon, casting subtle shadows across the mountain's surface and highlighting its contours. 

The overall impression is one of grandeur, tranquility, and the raw beauty of a winter landscape. The scale of the mountain is impressive, and the winding path invites a sense of exploration and adventure.
```

---

1) Start the server

```
mistralrs serve vision -p 1234 -m google/gemma-3n-E4B-it

# Or with MatFormer for balanced performance:
mistralrs serve vision -p 1234 -m google/gemma-3n-E4B-it \
  --matformer-config-path matformer_configs/gemma3n.csv \
  --matformer-slice-name "Config for E2.49B (block-level)"
```

2) Send a request

```py
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="ignore",
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
                    "text": "Please describe this image in detail.",
                },
            ],
        },
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
)
resp = completion.choices[0].message.content
print(resp)

```

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_local_img.py).

---

## Rust
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/vision_models/main.rs).

This is a minimal example of running the Gemma 3n model with a dummy image.

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new("google/gemma-3n-E4B-it")
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

    let messages = VisionMessages::new().add_image_message(
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

## Python
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gemma3n.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="google/gemma-3n-E4B-it",
        arch=VisionArchitecture.Gemma3n,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="ignore",
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
                        "text": "Please describe this image in detail.",
                    },
                ],
            }
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```


### OpenAI HTTP API

Audio is delivered with the `audio_url` content-type that mirrors OpenAIʼs official specification:

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

### Rust SDK

```rust
use anyhow::Result;
use mistralrs::{AudioInput, IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("google/gemma-3n-E4B-it")
        .with_isq(IsqType::Q4K)
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

    let messages = VisionMessages::new()
        .add_multimodal_message(
            TextMessageRole::User,
            "Describe in detail what is happening.",
            vec![image],
            vec![audio],
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
```

With this, you now have a single-call pipeline that fuses *sound*, *vision*, and *text* – all running locally through `mistral.rs`! 🔥

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_local_img.py).