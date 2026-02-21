# Llama 4 Series: [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)

**🚧 We are preparing a collection of UQFF quantized models! 🚧**

---

The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. 

**Architecture:**
- Efficient inference: 17B activated parameters
- Very sparse: 1 activated expert for both Scout (of 16), and Maverick (of 128)
- RoPE enhancement: iRoPE enables high context-length functionality

**Integration in mistral.rs:**
- Tool calling + [Automatic web search](WEB_SEARCH.md)
- ISQ
- Rust, Python and HTTP APIs

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust SDK takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

## HTTP server
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/llama4.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**

<img src="https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg">
<h6><a href = "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg">Credit</a></h6>

**Prompt:**
```
Please describe this image in detail.
```

**Output:**
```
The image presents a breathtaking mountain landscape, with a snow-capped peak dominating the scene. The mountain's rugged terrain is characterized by numerous ridges and valleys, while its summit is adorned with several structures that appear to be communication towers or antennas.

**Key Features:**

* **Mountain:** The mountain is the central focus of the image, showcasing a mix of snow-covered and bare areas.
* **Sky:** The sky above the mountain features a dramatic display of clouds, with dark grey clouds at the top gradually giving way to lighter blue skies towards the bottom.
* **Valley:** In the foreground, a valley stretches out, covered in trees that are mostly bare, suggesting a winter setting.
* **Lighting:** The lighting in the image is striking, with the sun casting a warm glow on the mountain's snow-covered slopes while leaving the surrounding areas in shadow.

**Overall Impression:**

The image exudes a sense of serenity and majesty, capturing the beauty of nature in a dramatic and awe-inspiring way. The contrast between the snow-covered mountain and the bare trees in the valley creates a visually appealing scene that invites the viewer to appreciate the natural world.
```

---

1) Start the server

```
mistralrs serve vision -p 1234 --isq 4 -m meta-llama/Llama-4-Scout-17B-16E-Instruct
```

2) Send a request

```py
from openai import OpenAI
import httpx
import textwrap
import json


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

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_local_img.py).

---

## Rust
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/vision_models/main.rs).

This is a minimal example of running the Llama 4 model with a dummy image.

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    )
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

## Python
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/llama4.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        arch=VisionArchitecture.Llama4,
    ),
    in_situ_quant="4",
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