# MiniCPM-O 2.6 Model: [`openbmb/MiniCPM-o-2_6`](https://huggingface.co/openbmb/MiniCPM-o-2_6)

Mistral.rs supports the MiniCPM-O 2.6 model, with examples in the Rust, Python, and HTTP APIs. ISQ quantization is supported to allow running the model with less memory requirements.

UQFF quantizations are coming soon.

> [!NOTE]
> Only the vision portion of this model has been implemented. No audio features are supported yet.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust SDK takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

## ToC
- [MiniCPM-O 2.6 Model: `openbmb/MiniCPM-o-2_6`](#minicpm-o-26-model-openbmbminicpm-o-2_6)
  - [ToC](#toc)
  - [Interactive mode](#interactive-mode)
  - [HTTP server](#http-server)
  - [Rust](#rust)
  - [Python](#python)

## Interactive mode

Mistral.rs supports interactive mode for vision models! It is an easy way to interact with the model.

1) Start up interactive mode with the MiniCPM-O 2.6 Model model

```
mistralrs run vision --isq 4 -m openbmb/MiniCPM-o-2_6
```

2) Say hello!
```
> Hello!
How can I assist you today?
```

3) Pass the model an image and ask a question.

```
> Hello!
How can I assist you today?
> \image https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Rosa_Precious_platinum.jpg/220px-Rosa_Precious_platinum.jpg What is this image?
The image shows a close-up view of a rose flower with dew drops on its petals. The rose is in full bloom, with its petals unfolding and displaying vibrant pink coloration. The dew drops on the petals create a delicate, glistening effect, adding to the overall visual appeal of the flower. The background is blurred, focusing attention on the intricate details of the rose.
```


## HTTP server
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/minicpmo_2_6.py).

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
The image shows Mount Washington, the highest peak in the Northeastern United States, located in the White Mountains of New Hampshire. The scene captures the mountain's rugged terrain and varied landscape features. 

In the foreground, there are dense forests of coniferous trees, primarily spruce and fir, which are typical of the region's boreal forest ecosystem. The trees are densely packed, indicating a high level of vegetation cover and biodiversity.

Moving upwards, the image reveals rocky outcroppings and boulders scattered across the slope, indicating the mountain's geological history of glacial activity. The presence of these rocks suggests that the area was once covered by ice sheets during the last ice age, which carved out the landscape and left behind a mix of boulders and talus slopes.

In the mid-ground, the image shows a series of ridges and valleys, which are characteristic of the mountain's glacially sculpted terrain. These features were formed by the movement of ice sheets that carved out U-shaped valleys and left behind a series of rounded hills and ridges.

At the summit, there is a prominent observation tower or weather station, which is likely used for scientific research and weather monitoring. The structure is situated at an elevation of approximately 6,288 feet (1,917 meters) above sea level, making it one of the highest points in the region.

The image also captures the atmospheric conditions on Mount Washington, with clouds and mist visible in the background. The mountain's unique location in a region where cold Arctic air meets warm moist air from the Gulf Stream creates a unique microclimate known as the "Home Rule," where extreme weather conditions can occur.

Overall, the image showcases the diverse geological and ecological features of Mount Washington, highlighting its role as a significant natural landmark in the Northeastern United States.
```

---

1) Start the server

```
mistralrs serve vision -p 1234 --isq 4 -m openbmb/MiniCPM-o-2_6
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

const MODEL_ID: &str = "openbmb/MiniCPM-o-2_6";

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new(MODEL_ID)
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
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/minicpmo_2_6.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

MODEL_ID = "openbmb/MiniCPM-o-2_6"

runner = Runner(
    which=Which.VisionPlain(
        model_id=MODEL_ID,
        arch=VisionArchitecture.MiniCpmO,
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
