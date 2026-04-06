# Gemma 4 Model Family: [`Collection`](https://huggingface.co/collections/google/gemma-4)

Gemma 4 is a multimodal model that supports text, vision (image), video, and audio input with text output. It builds on the Gemma family with full multimodal capabilities across all four input modalities.

Pre-quantized UQFF models are available in the [mistralrs-community Gemma 4 collection](https://huggingface.co/collections/mistralrs-community/gemma-4).

> **Video support**: Non-GIF video formats (mp4, avi, mov, etc.) require FFmpeg to be installed. See [VIDEO.md](VIDEO.md) for installation instructions and details.

## Quick Start

```bash
mistralrs run -m google/gemma-4-E4B-it --isq 8 --image photo.jpg -i "Describe this image"
```

With a video:

```bash
mistralrs run -m google/gemma-4-E4B-it --isq 8 --video video.mp4 -i "Describe this video in detail."
```

With audio:

```bash
mistralrs run -m google/gemma-4-E4B-it --isq 8 --audio audio.mp3 -i "Transcribe this fully."
```

## Input Formats

The Python and HTTP APIs support sending inputs as:
- **Images**: URL, path to a local file, or base64 encoded string (via `image_url`)
- **Videos**: URL or path to a local file (via `video_url`)
- **Audio**: URL or path to a local file (via `audio_url`)

The Rust SDK takes images from the [image](https://docs.rs/image/latest/image/index.html) crate, audio from `AudioInput`, and video from `VideoInput`.

## HTTP API

1) Start the server

```bash
mistralrs serve -m google/gemma-4-E4B-it --isq 8 -p 1234
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

**Video example:**

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
                    "type": "video_url",
                    "video_url": {
                        "url": "path/to/video.mp4"
                    },
                },
                {
                    "type": "text",
                    "text": "What happens in this video?",
                },
            ],
        }
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

**Combined image + video + audio example:**

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
                    "type": "video_url",
                    "video_url": {
                        "url": "path/to/video.mp4"
                    },
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "path/to/audio.wav"
                    },
                },
                {
                    "type": "text",
                    "text": "Describe what you see in the image and video, and what you hear in the audio.",
                },
            ],
        },
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_local_img.py).

## Python SDK

You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gemma4.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="google/gemma-4-E4B-it",
        arch=MultimodalArchitecture.Gemma4,
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

## Rust SDK

You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/multimodal_models/main.rs).

This is a minimal example of running the Gemma 4 model with a video input. Video decoding uses the `parse_video_url` helper from `mistralrs-server-core`, which handles FFmpeg decoding and frame sampling automatically.

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, MultimodalMessages, MultimodalModelBuilder};
use mistralrs_server_core::video::parse_video_url;

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        MultimodalModelBuilder::new("google/gemma-4-E4B-it")
            .with_isq(IsqType::Q4K)
            .with_logging()
            .build()
            .await?;

    let video = parse_video_url("path/to/video.mp4", None).await?;

    let messages = MultimodalMessages::new().add_video_message(
        TextMessageRole::User,
        "What happens in this video?",
        vec![video],
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
