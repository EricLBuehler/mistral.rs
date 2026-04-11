# Llama 3.2 Vision Model: [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

Mistral.rs supports the Llama 3.2 multimodal model. ISQ quantization is supported to allow running the model with less memory requirements.

UQFF quantizations are also available.

> Note: Some examples use the [Cephalo Llama 3.2 model](lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k), a member of the [Cephalo](https://huggingface.co/collections/lamm-mit/cephalo-664f3342267c4890d2f46b33) model collection. This model is finetune of Llama 3.2 with enhanced capabilities in scientific images. To use the base Llama 3.2 Vision model, simply use the [associated model ID](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).

> Note: When using device mapping or model topology, only the text model and its layers will be managed. This is because it contains most of the model parameters. *The text model has 40 layers*.

## Quick Start

```bash
mistralrs run -m meta-llama/Llama-3.2-11B-Vision-Instruct --isq 4 --image photo.jpg -i "Describe this image"
```

## Input Formats

The Python and HTTP APIs support sending inputs as:
- **Images**: URL, path to a local file, or base64 encoded string (via `image_url`)

The Rust SDK takes images from the [image](https://docs.rs/image/latest/image/index.html) crate.

## UQFF models

[UQFF](UQFF.md) is a quantized file format similar to GGUF based on ISQ. It removes the memory and compute requirements that come with ISQ by providing ready-made quantizations. The key advantage over GGUF is the flexibility to store multiple quantizations in one file.

We provide UQFF files ([EricB/Llama-3.2-11B-Vision-Instruct-UQFF](https://huggingface.co/EricB/Llama-3.2-11B-Vision-Instruct-UQFF)) for this Llama 3.2 Vision model.

For example:
```bash
mistralrs run -m meta-llama/Llama-3.2-11B-Vision-Instruct --from-uqff EricB/Llama-3.2-11B-Vision-Instruct-UQFF/llama-3.2-11b-vision-q4k.uqff
```

## HTTP API

1) Start the server

```bash
mistralrs serve -m meta-llama/Llama-3.2-11B-Vision-Instruct --isq 4 -p 1234
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

You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/llama_vision.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

MODEL_ID = "lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k"

runner = Runner(
    which=Which.MultimodalPlain(
        model_id=MODEL_ID,
        arch=MultimodalArchitecture.VLlama,
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

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_local_img.py).

## Rust SDK

You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/multimodal_models/main.rs).

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, MultimodalMessages, MultimodalModelBuilder};

const MODEL_ID: &str = "lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k";

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        MultimodalModelBuilder::new(MODEL_ID)
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

    let messages = MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        "What is depicted here? Please describe the scene in detail.",
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
