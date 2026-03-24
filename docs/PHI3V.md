# Phi 3 Vision Model: [`microsoft/Phi-3.5-vision-instruct`](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)

The Phi 3 Vision Model has support in the Rust, Python, and HTTP APIs. The Phi 3 Vision Model supports ISQ for increased performance.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust SDK takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

> Note: The Phi 3 Vision model works best with one image although it is supported to send multiple images.

> Note: when sending multiple images, they will be resized to the minimum dimension by which all will fit without cropping.
> Aspect ratio is not preserved in that case.

## HTTP server
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg" alt="Mount Washington" width = "1000" height = "666">
<h6><a href = "https://www.nhmagazine.com/mount-washington/">Credit</a></h6>

**Prompt:**
```
What is shown in this image? Write a detailed response analyzing the scene.
```

**Output:**
```
The image captures a breathtaking view of a mountain peak, bathed in the soft glow of sunlight. The peak, dusted with a layer of snow, stands tall against the backdrop of a clear blue sky. A trail, etched into the mountain's side by countless hikers before it, winds its way up to the summit. The trail's white color contrasts sharply with the surrounding landscape, drawing attention to its path and inviting exploration.

The perspective from which this photo is taken offers an expansive view of the mountain and its surroundings. It seems as if one could look down from this vantage point and see miles upon miles of untouched wilderness stretching out into the distance. The colors in the image are predominantly blue and white, reflecting both sky and snow-covered mountains respectively. However, there are also hints of green from trees dotting lower parts of mountainsides or valleys below them - adding another layer to this picturesque scene. This serene landscape evokes feelings of tranquility and adventure at once - an invitation to explore nature's grandeur while respecting its majesty at all times!
```

---

1) Start the server

```
mistralrs serve vision -p 1234 -m microsoft/Phi-3.5-vision-instruct
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
                    "text": "What is shown in this image? Write a detailed response analyzing the scene.",
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

This is a minimal example of running the Phi 3 Vision model with a dummy image.

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new("microsoft/Phi-3.5-vision-instruct")
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

    let messages = VisionMessages::new().add_phiv_image_message(
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
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="microsoft/Phi-3.5-vision-instruct",
        arch=VisionArchitecture.Phi3V,
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
                            "url": "https://upload.wikimedia.org/wikipedia/commons/e/e7/ Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is shown in this image? Write a detailed response analyzing the scene.",
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