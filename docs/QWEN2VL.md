# Qwen 2 Vision Model: [`Qwen2-VL Collection`](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)

Mistral.rs supports the Qwen2-VL vision model family, with examples in the Rust, Python, and HTTP APIs. ISQ quantization is supported to allow running the model with less memory requirements.

UQFF quantizations are also available.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust SDK takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

> Note: When using device mapping or model topology, only the text model and its layers will be managed. This is because it contains most of the model parameters. *The text model has 28 layers*.

## ToC
- [Qwen 2 Vision Model: `Qwen2-VL Collection`](#qwen-2-vision-model-qwen2-vl-collection)
  - [ToC](#toc)
  - [Interactive mode](#interactive-mode)
  - [HTTP server](#http-server)
  - [Rust](#rust)
  - [Python](#python)

## Interactive mode

Mistral.rs supports interactive mode for vision models! It is an easy way to interact with the model.

1) Start up interactive mode with the Qwen2-VL model

```
mistralrs run vision -m Qwen/Qwen2-VL-2B-Instruct
```

2) Say hello!
```
> Hello!
Hello! How can I assist you today?
```

3) Pass the model an image and ask a question.

```
> Hello!
Hello! How can I assist you today?
> \image https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg What type of flower is this? Give some fun facts.
flowers are a type of flowering plant that produce flowers that are typically used for decoration, pollination, and reproduction. there are many different types of flowers, each with its own unique characteristics and uses. here are some fun facts about camellias:

  * camellias are native to china and have been cultivated for over 2,000 years.
  * camellias are known for their long blooming season, with some varieties blooming continuously for months.
  * camellias come in a wide variety of colors, including red, pink, white, and yellow.
  * camellias are also known for their fragrant blooms, which can be enjoyed by both humans and animals.
  * camellias are often used in gardens and parks as a decorative element, and are also popular in landscaping and horticulture.

camellias are also known for their resilience and ability to thrive in a variety of conditions, making them a popular choice for gardeners and landscapers. they require well-draining soil and full sun or partial shade, and can be grown in containers or in the ground. overall, camellias are a beautiful and versatile flower that can add beauty and interest to any garden or landscape.
```

## HTTP server
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen2vl.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg" alt="Mount Washington">

**Prompt:**
```
What type of flower is this? Give some fun facts.
```

**Output:**
```
flowers are a beautiful addition to any garden or outdoor space. They come in many different colors and shapes, and can be used for both decorative purposes and as sources of pollination for bees and other insects.

One fun fact about camellias is that they are native to Japan, but were introduced to Europe in the 17th century by Portuguese sailors who brought them back from their voyages around the world. Camellias have been popular as ornamental plants since then, with many varieties available for cultivation.

Camellias also have interesting cultural significance in Japan, where they are often associated with good fortune and prosperity. In Chinese culture, camellias symbolize longevity and immortality.
In conclusion, camellias are beautiful flowers that add color and interest to gardens or outdoor spaces. They come in many different colors and shapes, making them a popular choice for gardeners everywhere!
```

---

1) Start the server

```
mistralrs serve vision -p 1234 -m Qwen/Qwen2-VL-2B-Instruct
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

const MODEL_ID: &str = "Qwen/Qwen2-VL-2B-Instruct";

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new(MODEL_ID)
            .with_isq(IsqType::Q4K)
            .with_logging()
            .build()
            .await?;

    let bytes = match reqwest::blocking::get(
        "https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "What type of flower is this? Give some fun facts.",
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
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen2vl.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

runner = Runner(
    which=Which.VisionPlain(
        model_id=MODEL_ID,
        arch=VisionArchitecture.Qwen2VL,
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
