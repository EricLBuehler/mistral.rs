# Idefics 2 Model: [`HuggingFaceM4/idefics2-8b-chatty`](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty)

The Idefics 2 Model has support in the Rust, Python, and HTTP APIs. The Idefics 2 Model also supports ISQ for increased performance. 

> Note: Some of examples use our [Cephalo model series](https://huggingface.co/collections/lamm-mit/cephalo-664f3342267c4890d2f46b33) but could be used with any model ID.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust API takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

## Interactive mode

> [!NOTE]
> In interactive mode, the Idefics 2 vision model does not automatically add the image token!
> It should be added to messages manually, and is of the format `<image>`.

## HTTP server
You can find this example [here](../examples/server/idefics2.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg" width = "1000" height = "666">

**Prompt:**
```
What is shown in this image?
```

**Output:**
```
The image depicts a group of orange ants climbing over a black pole. The ants are moving in the same direction, forming a line as they ascend the pole.
```

---

1) Start the server

> [!NOTE]
> You should replace `--features ...` with one of the features specified [here](../README.md#supported-accelerators), or remove it for pure CPU inference.

```
cargo run --release --features ... -- --port 1234 --isq Q4K vision-plain -m HuggingFaceM4/idefics2-8b-chatty -a idefics2
```

2) Send a request

```py
import openai

completion = client.chat.completions.create(
    model="idefics2",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "What is shown in this image?",
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
You can find this example [here](../mistralrs/examples/idefics2/main.rs).

This is a minimal example of running the Idefics 2 model with a dummy image.

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new(
        "HuggingFaceM4/idefics2-8b-chatty",
        VisionLoaderType::Idefics2,
    )
    .with_isq(IsqType::Q4K)
    .with_logging()
    .build()
    .await?;

    let bytes = match reqwest::blocking::get(
        "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new().add_idefics_image_message(
        TextMessageRole::User,
        "What is depicted here? Please describe the scene in detail.",
        image,
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
You can find this example [here](../examples/python/phi3v.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="lamm-mit/Cephalo-Idefics-2-vision-8b-beta",
        arch=VisionArchitecture.Idefics2,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="idefics2",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is shown in this image?",
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