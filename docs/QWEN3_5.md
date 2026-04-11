# Qwen 3.5 Multimodal Models: [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B)

The Qwen 3.5 models are vision-language models using a hybrid Gated Delta Network (GDN) + full attention architecture. Both dense and MoE (Mixture of Experts) variants are supported:

- **Dense**: `Qwen/Qwen3.5-27B`
- **MoE**: `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-397B-A17B`

ISQ quantization is supported to allow running the model with less memory requirements. MoE variants also support [MoQE](ISQ.md) via the `--organization moqe` flag.

UQFF quantizations are also available.

> Note: When using device mapping or model topology, only the text model and its layers will be managed. This is because it contains most of the model parameters.

## Quick Start

```bash
mistralrs run -m Qwen/Qwen3.5-27B --isq 4 --image photo.jpg -i "Describe this image"
```

Or with the MoE variant:

```bash
mistralrs run -m Qwen/Qwen3.5-35B-A3B --isq 4 --image photo.jpg -i "Describe this image"
```

## Input Formats

The Python and HTTP APIs support sending inputs as:
- **Images**: URL, path to a local file, or base64 encoded string (via `image_url`)

The Rust SDK takes images from the [image](https://docs.rs/image/latest/image/index.html) crate.

## HTTP API

1) Start the server

```bash
mistralrs serve -m Qwen/Qwen3.5-27B --isq 4 -p 1234
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

You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3_5.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

# Dense variant
MODEL_ID = "Qwen/Qwen3.5-27B"

runner = Runner(
    which=Which.MultimodalPlain(
        model_id=MODEL_ID,
        arch=MultimodalArchitecture.Qwen3_5,
    ),
)

# For MoE variant, use:
# MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
# runner = Runner(
#     which=Which.MultimodalPlain(
#         model_id=MODEL_ID,
#         arch=MultimodalArchitecture.Qwen3_5Moe,
#     ),
# )

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

#[tokio::main]
async fn main() -> Result<()> {
    let model = MultimodalModelBuilder::new("Qwen/Qwen3.5-27B")
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

    let messages = MultimodalMessages::new().add_image_message(
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
