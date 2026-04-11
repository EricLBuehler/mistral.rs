# LLaVA and LLaVANext Model: [`llava-hf model family`](https://huggingface.co/llava-hf)

The [LLaVA](https://arxiv.org/abs/2310.03744) and [LLaVANext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) are multimodal models that can handle both text and vision inputs.

This implementation supports both LLaVA and LLaVANext (which adds multi resolution image processing) and two types of LLM base model: llama and mistral. Currently it is tested on:
* llava-hf/llava-v1.6-mistral-7b-hf
* llava-hf/llava-v1.6-vicuna-7b-hf
* llava-hf/llava-1.5-7b-hf

ISQ is supported for increased performance.

## Quick Start

```bash
mistralrs run -m llava-hf/llava-v1.6-mistral-7b-hf --isq 4 --image photo.jpg -i "Describe this image"
```

For vicuna backend, specify the chat template:

```bash
mistralrs run -m llava-hf/llava-v1.6-vicuna-7b-hf --isq 4 -c ./chat_templates/vicuna.json --image photo.jpg -i "Describe this image"
```

## Input Formats

The Python and HTTP APIs support sending inputs as:
- **Images**: URL, path to a local file, or base64 encoded string (via `image_url`)

The Rust SDK takes images from the [image](https://docs.rs/image/latest/image/index.html) crate.

## HTTP API

1) Start the server

```bash
mistralrs serve -m llava-hf/llava-v1.6-mistral-7b-hf --isq 4 -p 1234
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

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_local_img.py).

## Python SDK

You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/llava_next.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        arch=MultimodalArchitecture.LLaVANext,
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
            },
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
    let model = MultimodalModelBuilder::new(
        "llava-hf/llava-v1.6-mistral-7b-hf",
    )
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

    let messages = MultimodalMessages::new().add_llava_image_message(
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
