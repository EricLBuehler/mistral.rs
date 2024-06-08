# Phi 3 Vision Support: `microsoft/Phi-3-vision-128k-instruct`

The Phi 3 Vision Model has support in the Rust, Python, and HTTP APIs. The Phi 3 Vision Model supports ISQ for increased performance.

> Note: The Phi 3 Vision model works best with one image although it is supported to send multiple images.

> Note: when sending multiple images, they will be resized to the minimum dimension by which all will fit without cropping.
> Aspect ratio is not preserved.

## HTTP server
You can find this example [here](../examples/server/phi3v.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://upload.wikimedia.org/wikipedia/commons/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg" alt="Mount Everest" width = "1000" height = "666">

**Prompt:**
```
<|image_1|>\nWhat is shown in this image?
```

**Output:**
```
The image shows a large, snow-covered mountain with a clear blue sky. There are no visible clouds or precipitation, and the mountain appears to be quite steep with visible crevices and ridges. The surrounding landscape includes rocky terrain at the base of the mountain.
```

---

1) Start the server
```
cargo run --release --features cuda -- --port 1234 vision-plain -m microsoft/Phi-3-vision-128k-instruct -a phi3v
```

2) Send a request

```py
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:1234/v1/"

completion = openai.chat.completions.create(
    model="phi3v",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "<|image_1|>\nWhat is shown in this image?",
                },
            ],
        },
    ],
    max_tokens=32,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
)
resp = completion.choices[0].message.content
print(resp)

```

---

## Rust
You can find this example [here](../mistralrs/examples/phi3v/main.rs).

This is a minimal example of running the Phi 3 Vision model with a dummy image.

```rust
use either::Either;
use image::{ColorType, DynamicImage};
use indexmap::IndexMap;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    Constraint, Device, DeviceMapMetadata, MistralRs, MistralRsBuilder, NormalRequest, Request,
    RequestMessage, Response, SamplingParams, SchedulerMethod, TokenSource, VisionLoaderBuilder,
    VisionLoaderType, VisionSpecificConfig,
};

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    let loader = VisionLoaderBuilder::new(
        VisionSpecificConfig {
            use_flash_attn: false,
            repeat_last_n: 64,
        },
        None,
        None,
        Some("microsoft/Phi-3-vision-128k-instruct".to_string()),
    )
    .build(VisionLoaderType::Phi3V);
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        None,
        &Device::cuda_if_available(0)?,
        false,
        DeviceMapMetadata::dummy(),
        None,
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build())
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
                    Either::Left("<|image_1|>\nWhat is shown in this image?".to_string()),
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
    });
    mistralrs.get_sender().blocking_send(request)?;

    let response = rx.blocking_recv().unwrap();
    match response {
        Response::Done(c) => println!("Text: {}", c.choices[0].message.content),
        _ => unreachable!(),
    }
    Ok(())
}
```

## Python
You can find this example [here](../examples/python/phi3v.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="microsoft/Phi-3-vision-128k-instruct",
        tokenizer_json=None,
        repeat_last_n=64,
        arch=VisionArchitecture.Phi3V,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="phi3v",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/e/e7/ Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "<|image_1|>\nWhat is shown in this image?",
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