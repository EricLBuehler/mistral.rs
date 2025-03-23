# Mistral Small 3.1 Model: [`mistralai/Mistral-Small-3.1-24B-Instruct-2503`](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)

The Mistral Small 3.1 model is a strong multimodal (text+vision) model with 128k context length, function calling, and strong visual understanding.

We support the Mistral 3 Model in the Rust, Python, and HTTP APIs, including ISQ for increased performance.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust API takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

## Tool calling with Mistral Small 3.1

The Mistral Small 3.1 model itself does not come with the correct JINJA chat template to enable tool calling. We provide a chat template for
tool calling with Mistral Small 3.1, and you can use it by specifying the `jinja_explicit` parameter in the various APIs. For example:

```bash
./mistralrs-server --port 1234 --isq q4k --jinja-explicit chat_templates/mistral_small_tool_call.jinja vision-plain -m mistralai/Mistral-Small-3.1-24B-Instruct-2503 -a mistral3  
```


## HTTP server
You can find this example [here](../examples/server/mistral3.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**

<img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg">
<h6><a href = "https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg">Credit</a></h6>

**Prompt:**
```
What is this?
```

**Output:**
```
The image shows a close-up of a vibrant flower with pink petals and a central cluster of yellowish-brown stamens. This flower appears to be from the genus *Gazania*, commonly known as treasure flowers or gazanias. These flowers are known for their daisy-like appearance and bright colors.

Gazania flowers typically have ray florets (the petal-like structures) that can change color based on light conditions—often appearing more vibrant in direct sunlight. They are popular in gardens for their hardiness and ability to thrive in sunny locations with well-drained soil.

If there's anything specific about this flower or its care that interests you further, feel free to ask!
```

---

1) Start the server

> [!NOTE]
> You should replace `--features ...` with one of the features specified [here](../README.md#supported-accelerators), or remove it for pure CPU inference.

```
cargo run --release --features ... -- --port 1234 vision-plain -m mistralai/Mistral-Small-3.1-24B-Instruct-2503 -a mistral3
```

2) Send a request

```py
from openai import OpenAI
import httpx
import textwrap
import json


client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")


completion = client.chat.completions.create(
    model="mistral3",
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
You can find this example [here](../mistralrs/examples/mistral3/main.rs).

This is a minimal example of running the Mistral 3 model with a dummy image.

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new("mistralai/Mistral-Small-3.1-24B-Instruct-2503", VisionLoaderType::Mistral3)
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
You can find this example [here](../examples/python/mistral3.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        arch=VisionArchitecture.Mistral3,
    ),
    in_situ_quant="Q4K"
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral3",
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