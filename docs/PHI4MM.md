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
cargo run --release --features ... -- --port 1234 vision-plain -m microsoft/Phi-4-multimodal-instruct -a phi4mm
```

2) Send a request

```py
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="phi4mm",
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
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new("microsoft/Phi-4-multimodal-instruct", VisionLoaderType::Phi4MM)
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
        model="phi4mm",
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

- You can find an example of encoding the [image via base64 here](../examples/python/phi3v_base64.py).
- You can find an example of loading an [image locally here](../examples/python/phi3v_local_img.py).