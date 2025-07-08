# Phi 4 Multimodal Model: [`microsoft/Phi-4-multimodal-instruct`](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

The Phi 4 Multimodal Model has support in the Rust, Python, and HTTP APIs. The Phi 4 Multimodal Model supports ISQ for increased performance.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust API takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

> Note: The Phi 4 Multimodal model works best with one image although it is supported to send multiple images.

> Note: when sending multiple images, they will be resized to the minimum dimension by which all will fit without cropping.
> Aspect ratio is not preserved in that case.

> [!NOTE]
> The Phi 4 Multimodal model does not automatically add the image tokens!
> They should be added to messages manually, and are of the format `<|image_{N}|>` where N starts from 1.

[**Phi 4 multimodal supports audio inputs!**](#audio-input).

## HTTP server
You can find this example [here](../examples/server/phi3v.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg" alt="Mount Washington" width = "1000" height = "666">
<h6><a href = "https://www.nhmagazine.com/mount-washington/">Credit</a></h6>

**Prompt:**
```
<|image_1|>\nWhat is shown in this image? Write a detailed response analyzing the scene.
```

**Output:**
```
A mountain with snow on it.
```

---

1) Start the server

> [!NOTE]
> You should replace `--features ...` with one of the features specified [here](../README.md#supported-accelerators), or remove it for pure CPU inference.

```
cargo run --release --features ... -- --port 1234 vision-plain -m microsoft/Phi-4-multimodal-instruct
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
                    "text": "<|image_1|>\nWhat is shown in this image? Write a detailed response analyzing the scene.",
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
You can find this example [here](../mistralrs/examples/phi3v/main.rs).

This is a minimal example of running the Phi 4 Multimodal model with a dummy image.

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new("microsoft/Phi-4-multimodal-instruct")
            .with_isq(IsqType::Q4K)
            .with_logging()
            .build()
            .await?;

    let bytes = match reqwest::blocking::get(
        "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "What is depicted here? Please describe the scene in detail.",
        image,
        &model,
    )?;

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
You can find this example [here](../examples/python/phi3v.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="microsoft/Phi-4-multimodal-instruct",
        arch=VisionArchitecture.Phi4MM,
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
                            "url": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "<|image_1|>\nWhat is shown in this image? Write a detailed response analyzing the scene.",
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

## Audio input

Alongside vision, Phi 4 Multimodal in `mistral.rs` can accept **audio** as an additional modality.  This unlocks fully-local pipelines such as **text + speech + vision → text** where the model can reason jointly over what it *hears* and what it *sees*.

`mistral.rs` automatically decodes the supplied audio (WAV/MP3/FLAC/OGG/… – anything [Symphonia](https://github.com/pdeljanov/Symphonia) can handle) into 16-bit PCM.

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
      "text": "<|audio_1|><|image_1|>\nDescribe what is happening in this clip in as much detail as possible."
    }
  ]
}
```

### Rust API

```rust
use anyhow::Result;
use mistralrs::{AudioInput, IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("microsoft/Phi-4-multimodal-instruct")
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
            &model,
        )?;

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
```

With this, you now have a single-call pipeline that fuses *sound*, *vision*, and *text* – all running locally through `mistral.rs`! 🔥

- You can find an example of encoding the [image via base64 here](../examples/python/phi3v_base64.py).
- You can find an example of loading an [image locally here](../examples/python/phi3v_local_img.py).