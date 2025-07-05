# Gemma 3n Model: [`google/gemma-3n-E4B-it`](https://huggingface.co/google/gemma-3n-E4B-it)

Gemma 3n models are designed for efficient execution on low-resource devices. They are capable of multimodal input, handling text, image, video, and audio input, and generating text outputs. These models support over 140 spoken languages.

The Gemma 3n Model has support in the Rust, Python, and HTTP APIs. The Gemma 3n Model supports ISQ for increased performance.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust API takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

## Audio input

Alongside vision, Gemma 3n in `mistral.rs` can accept **audio** as an additional modality.  This unlocks fully-local pipelines such as **text + speech + vision → text** where the model can reason jointly over what it *hears* and what it *sees*.

`mistral.rs` automatically decodes the supplied audio (WAV/MP3/FLAC/OGG/… – anything [Symphonia](https://github.com/pdeljanov/Symphonia) can handle) into 16-bit PCM.

## HTTP server
You can find this example [here](../examples/server/gemma3n.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg" alt="Zinnia Flower" width = "1000" height = "666">
<h6><a href = "https://www.wikimedia.org/">Credit</a></h6>

**Prompt:**
```
Please describe this image in detail.
```

**Output:**
```
The image is a close-up shot of a vibrant pink and yellow flower, likely a rose, set against a blurred background of green foliage. The flower is the clear focal point, sharply in focus and brightly illuminated. 

Here's a detailed breakdown:

**The Flower:**

*   **Color:** The petals are a striking, almost neon pink, with hints of deeper magenta in the folds. The center of the flower transitions to a warm, golden yellow.
*   **Form:** The flower appears to be in full bloom, with numerous layers of petals unfurling outwards. The petals have a soft, velvety texture suggested by the way light catches them. They are arranged in a classic rose shape, with some petals curled and others gently spreading.
*   **Details:**  The edges of the petals are slightly ruffled and uneven, adding to the naturalistic feel.  There's a subtle gradient of color within each petal, creating depth and dimension.  The yellow center is densely packed with what appear to be stamens or pistils.
*   **Lighting:** The flower is strongly lit from above and slightly to the side, creating highlights and shadows that emphasize its three-dimensional form. This lighting also enhances the vibrancy of the colors.

**The Background:**

*   **Color:** The background is a soft, out-of-focus green, suggesting leaves and other foliage. 
*   **Blur:** The background is heavily blurred (bokeh), which helps to isolate the flower and make it stand out. The blur creates a sense of depth and draws the viewer's eye to the sharp details of the flower.
*   **Texture:** While blurred, you can discern the texture of leaves – some appear smooth, others slightly textured.

**Overall Impression:**

The image is visually striking due to the contrast between the bright pink and yellow of the flower and the soft green background. The sharp focus on the flower combined with the blurred background creates a sense of intimacy and emphasizes the beauty of the bloom. It's a classic example of a macro photograph used to highlight the intricate details and vibrant colors of nature. The overall feeling is one of freshness, beauty, and vibrancy.
```

---

1) Start the server

> [!NOTE]
> You should replace `--features ...` with one of the features specified [here](../README.md#supported-accelerators), or remove it for pure CPU inference.

```
cargo run --release --features ... -- --port 1234 run -m google/gemma-3n-E4B-it
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
                        "url": "https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg"
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

- You can find an example of encoding the [image via base64 here](../examples/server/phi3v_base64.py).
- You can find an example of loading an [image locally here](../examples/server/phi3v_local_img.py).

---

## Rust
You can find this example [here](../mistralrs/examples/gemma3n/main.rs).

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
        "https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "Please describe the image in detail.",
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
You can find this example [here](../examples/python/gemma3n.py).

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
                            "url": "https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg"
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

### Rust API

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