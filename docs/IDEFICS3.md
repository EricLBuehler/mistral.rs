# Idefics 3 Vision: [`HuggingFaceM4/Idefics3-8B-Llama3`](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)

Mistral.rs supports the Idefics 3 vision model, with examples in the Rust, Python, and HTTP APIs. ISQ quantization is supported to allow running the model with less memory requirements.

UQFF quantizations are also available.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust SDK takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

> Note: When using device mapping or model topology, only the text model and its layers will be managed. This is because it contains most of the model parameters. Check the Hugging Face text model config for more information or raise an issue.

## ToC
- [Idefics 3 Vision: `HuggingFaceM4/Idefics3-8B-Llama3`](#idefics-3-vision-huggingfacem4idefics3-8b-llama3)
  - [ToC](#toc)
  - [Using the 🤗 Smol VLM models](#using-the--smol-vlm-models)
  - [Interactive mode](#interactive-mode)
  - [HTTP server](#http-server)
  - [Rust](#rust)
  - [Python](#python)
  - [UQFF models](#uqff-models)

## Using the [🤗 Smol VLM](HuggingFaceTB/SmolVLM-Instruct) models

Simply substitute the Idefics 3 model ID (`HuggingFaceM4/Idefics3-8B-Llama3`) with the Smol VLM one (`HuggingFaceTB/SmolVLM-Instruct`)!

## Interactive mode

Mistral.rs supports interactive mode for vision models! It is an easy way to interact with the model.

1) Start up interactive mode with the Idefics 3 model

```
mistralrs run vision --isq 4 -m HuggingFaceM4/Idefics3-8B-Llama3
```

2) Ask a question
```
> \image https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Rosa_Precious_platinum.jpg/220px-Rosa_Precious_platinum.jpg What is this image?
The image depicts a single, large, red rose in full bloom. The rose is positioned against a blurred background that suggests a natural setting, possibly outdoors. The petals of the rose are vividly red with a slight sheen, indicating that they are wet, likely from recent rainfall or dew. The petals are tightly packed and have a velvety texture, which is characteristic of roses. The edges of the petals are slightly curled and appear to be glistening with water droplets, enhancing the overall freshness and beauty of the flower.

The stem of the rose is visible and appears to be green, with a few small thorns scattered along its length. The stem is slender and supports the weight of the large, showy head of the rose. The leaves that accompany the stem are not fully visible in the image but are implied by the presence of the stem.

The background is out of focus, which helps to emphasize the rose as the main subject of the image. The blurred background suggests a natural environment, possibly a garden or a field, with hints of greenery and possibly other flowers or plants. The lighting in the image is natural, likely from sunlight, which casts soft shadows on the petals and adds depth to the scene.

The overall composition of the image focuses on the rose, making it the central point of interest. The wetness of the petals adds a dynamic element to the stillness of the flower, giving it a sense of life and vitality. This could symbolize themes of beauty, nature, and perhaps even passion or love.

In summary, this image captures a single red rose in full bloom with wet petals against a blurred natural background. The rose is the focal point, with its vibrant red color and glistening petals drawing attention. The natural lighting and out-of-focus background enhance the beauty and freshness of the flower.
```

4) Continue the chat by passing another image.
```
> \image https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Rosa_Precious_platinum.jpg/220px-Rosa_Precious_platinum.jpg What is this image?
The image depicts a single, large, red rose in full bloom. The rose is positioned against a blurred background that suggests a natural setting, possibly outdoors. The petals of the rose are vividly red with a slight sheen, indicating that they are wet, likely from recent rainfall or dew. The petals are tightly packed and have a velvety texture, which is characteristic of roses. The edges of the petals are slightly curled and appear to be glistening with water droplets, enhancing the overall freshness and beauty of the flower.

The stem of the rose is visible and appears to be green, with a few small thorns scattered along its length. The stem is slender and supports the weight of the large, showy head of the rose. The leaves that accompany the stem are not fully visible in the image but are implied by the presence of the stem.

The background is out of focus, which helps to emphasize the rose as the main subject of the image. The blurred background suggests a natural environment, possibly a garden or a field, with hints of greenery and possibly other flowers or plants. The lighting in the image is natural, likely from sunlight, which casts soft shadows on the petals and adds depth to the scene.

The overall composition of the image focuses on the rose, making it the central point of interest. The wetness of the petals adds a dynamic element to the stillness of the flower, giving it a sense of life and vitality. This could symbolize themes of beauty, nature, and perhaps even passion or love.

In summary, this image captures a single red rose in full bloom with wet petals against a blurred natural background. The rose is the focal point, with its vibrant red color and glistening petals drawing attention. The natural lighting and out-of-focus background enhance the beauty and freshness of the flower.
> \image https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg What mountain is this?
The mountain is Mount Washington.
```

## HTTP server
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/idefics3.py).

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
The image depicts a majestic mountain landscape under a partly cloudy sky, characterized by its rugged and snow-covered peaks. The mountain is prominently featured in the center of the image, showcasing its expansive and undulating terrain. The summit of the mountain is capped with snow, indicating that it might be winter or early springtime.

The slopes of the mountain are steep and uneven, covered with patches of snow that appear to have been recently fallen or freshly groomed for skiing or other winter activities. There are visible ski trails descending from the summit down towards what seems to be a valley below, suggesting that this location could be a popular ski resort area.

In addition to the main peak, there are smaller hills and ridges surrounding it on both sides. These secondary peaks also have varying degrees of snow cover but appear less prominent than the central peak.

The sky above is mostly overcast with clouds covering most parts but allowing some sunlight to peek through in certain areas, casting soft shadows on parts of the mountainside. This lighting suggests that it might not be midday yet as there isn't an intense brightness typical for noon hours.

On closer inspection near one side of this grandeur scene stands tall trees without leaves; their bare branches starkly contrasting against both white snow and blue sky create an interesting... (cut off)
```

---

1) Start the server

```
mistralrs serve vision -p 1234 --isq 4 -m HuggingFaceM4/Idefics3-8B-Llama3
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

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

const MODEL_ID: &str = "HuggingFaceM4/Idefics3-8B-Llama3";

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new(MODEL_ID)
        .with_isq(IsqType::Q8_0)
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
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/idefics3.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="HuggingFaceM4/Idefics3-8B-Llama3",
        arch=VisionArchitecture.Idefics3,
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
                            "url": "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg"
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

- You can find an example of encoding the [image via base64 here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_base64.py).
- You can find an example of loading an [image locally here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_local_img.py).

## UQFF models
Coming soon!
