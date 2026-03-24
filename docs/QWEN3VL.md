# Qwen 3 Vision Model: [`Qwen3 VL Collection`](https://huggingface.co/collections/Qwen/qwen3-vl)

The Qwen 3 VL models are the successors to the Qwen 2.5 VL models, featuring a diverse lineup of increased performance, flexible sizes, and reasoning-capable models.

Mistral.rs supports the Qwen 3 VL vision model family (including MoE variants), with examples in the Rust, Python, and HTTP APIs. ISQ quantization is supported to allow running the model with less memory requirements. MoE variants also support [MoQE](ISQ.md) via the `--organization moqe` flag.

UQFF quantizations are also available.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust SDK takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

> Note: When using device mapping or model topology, only the text model and its layers will be managed. This is because it contains most of the model parameters.

## ToC
- [Qwen 3 Vision Model: `Qwen3 VL Collection`](#qwen-3-vision-model-qwen3-vl-collection)
  - [ToC](#toc)
  - [Interactive mode](#interactive-mode)
  - [HTTP server](#http-server)
  - [Rust](#rust)
  - [Python](#python)

## Interactive mode

Mistral.rs supports interactive mode for vision models! It is an easy way to interact with the model.

Start up interactive mode with the Qwen3 VL model:

```
mistralrs run vision -m Qwen/Qwen3-VL-4B-Instruct
```

## HTTP server
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen3_vl.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

1) Start the server

```
mistralrs serve vision -p 1234 -m Qwen/Qwen3-VL-4B-Instruct
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
                        "url": "https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "What type of flower is this? Give some fun facts.",
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

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("Qwen/Qwen3-VL-4B-Instruct")
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
        "What is this?",
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

---

## Python
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3_vl.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

MODEL_ID = "Qwen/Qwen3-VL-4B-Thinking"

runner = Runner(
    which=Which.VisionPlain(
        model_id=MODEL_ID,
        arch=VisionArchitecture.Qwen3VL,
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
                            "url": "https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What type of flower is this? Give some fun facts.",
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

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_local_img.py).
