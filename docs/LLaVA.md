# LLaVA and LLaVANext Model: `llava-hf model family`

The [LLaVA](https://arxiv.org/abs/2310.03744) and [LLaVANext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) are great multimodal models that can handle both text and vision inputs.

This implementation supports both LLaVA and LLaVANext(which adds multi resolution image processing) and two types of LLM base model: llama and mistral. Currently it is tested on:
* llava-hf/llava-v1.6-mistral-7b-hf
* llava-hf/llava-v1.6-vicuna-7b-hf
* llava-hf/llava-1.5-7b-hf


The LLaVA and LLaVANext Model has support in the Rust, Python, and HTTP APIs. The LLaVA and LLaVANext Model also supports ISQ for increased performance. 

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust API takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

## HTTP server
You can find this example [here](../examples/server/llava_next.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg" alt="Mount Washington" width = "1000" height = "666">
<h6><a href = "https://www.nhmagazine.com/mount-washington/">Credit</a></h6>

**Prompt:**
```
<image>What is shown in this image? 
```

**Output:**
```
Text: The image shows a steep, snow-covered hillside with a pine tree on the right side, close to the top. The landscape appears to be a mountainous area with winter conditions. There are no visible humans or permanent structures in the immediate vicinity that suggest this is a summer or recreational location. It's likely a cold, snowy day or season, and the slopes might be part of a mountainous region.
```

---

1) Start the server
```
cargo run --release --features ... -- --port 1234 --isq Q4K vision-plain -m llava-hf/llava-v1.6-mistral-7b-hf -a llava_next
//or 
cargo run  --features cuda -- --port 1234  --isq Q4K --chat-template ./chat_templates/vicuna.json vision-plain -m /root/autodl-tmp/llava-v1.6-vicuna-7b-hf -a llava_next // if use vicuna as backend llm, then we need to specific the chat-template
```

2) Send a request

```py
import openai

completion = client.chat.completions.create(
    model="llava_next",
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
                    "text": "<image>What is shown in this image?",
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

- You can find an example of encoding the [image via base64 here](../examples/server/phi3v_base64.py).
- You can find an example of loading an [image locally here](../examples/server/phi3v_local_img.py).

---

## Rust
You can find this example [here](../mistralrs/examples/llava_next/main.rs).

This is a minimal example of running the LLaVA and LLaVANext model with a dummy image.

```rust
use either::Either;
use image::{ColorType, DynamicImage};
use indexmap::IndexMap;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    Constraint, DefaultSchedulerMethod, Device, DeviceMapMetadata, MistralRs, MistralRsBuilder,
    ModelDType, NormalRequest, Request, RequestMessage, Response, SamplingParams, SchedulerConfig,
    TokenSource, VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig,
};

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    let loader = VisionLoaderBuilder::new(
        VisionSpecificConfig {
            use_flash_attn: false,
        },
        None,
        None,
        Some("llava-hf/llava-v1.6-mistral-7b-hf".to_string()),
    )
    .build(VisionLoaderType::LLaVANext);
    // Load, into a Pipeline

    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &Device::cuda_if_available(0)?,
        false,
        DeviceMapMetadata::dummy(),
        None,
        None, // No PagedAttention.
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(
        pipeline,
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(5.try_into().unwrap()),
        },
    )
    .build())
}

fn main() -> anyhow::Result<()> {
    let mistralrs = setup()?;

    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest {
        messages: RequestMessage::VisionChat {
            images: vec![DynamicImage::new(1280, 720, ColorType::Rgb8)],
            messages: vec![IndexMap::from([
                ("role".to_string(), Either::Left("user".to_string())),
                (
                    "content".to_string(),
                    Either::Left("<image>What is shown in this image?".to_string()),
                ),
            ])],
        },
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::None,
        suffix: None,
        adapters: None,
        tools: None,
        tool_choice: None,
    });
    mistralrs.get_sender()?.blocking_send(request)?;

    let response = rx.blocking_recv().unwrap();
    match response {
        Response::Done(c) => println!("Text: {}", c.choices[0].message.content),
        _ => unreachable!(),
    }
    Ok(())
}
```

## Python
You can find this example [here](../examples/python/llava_next.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        arch=VisionArchitecture.LLaVANext,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="llava_next",
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
                        "text": "<image>What is shown in this image?",
                    },
                ],
            },
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

- You can find an example of encoding the [image via base64 here](../examples/python/phi3v_base64.py).
- You can find an example of loading an [image locally here](../examples/python/phi3v_local_img.py).