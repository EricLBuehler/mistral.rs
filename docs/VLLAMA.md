# Llama 3.2 Vision Model: [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

Mistral.rs supports the Llama 3.2 vision model, with examples in the Rust, Python, and HTTP APIs. ISQ quantization is supported to allow running the model with less memory requirements.

UQFF quantizations will be released shortly.

The Python and HTTP APIs support sending images as:
- URL
- Path to a local image
- [Base64](https://en.wikipedia.org/wiki/Base64) encoded string

The Rust API takes an image from the [image](https://docs.rs/image/latest/image/index.html) crate.

> Note: Some examples use the [Cephalo Llama 3.2 model](lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k), a member of the [Cephalo](https://huggingface.co/collections/lamm-mit/cephalo-664f3342267c4890d2f46b33) model collection. This model is finetune of Llama 3.2 with enhanced capabilities in scientific images. To use the base Llama 3.2 Vision model, simply use the [associated model ID](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).

## ToC
- [Interactive mode](#interactive-mode)
- [HTTP server](#http-server)
- [Rust API](#rust)
- [Python API](#python)

## Interactive mode

Mistral.rs supports interactive mode for vision models! It is an easy way to interact with the model.

https://github.com/user-attachments/assets/4d11c35c-9ea2-42b8-8cab-5f7e8e2ee9ff

1) Start up interactive mode with the Llama 3.2 model
```
cargo run --features ... --release -- -i --isq Q4K vision-plain -m lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k -a vllama
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
> \image https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg <|image|>What type of flower is this?        
The flower shown in the image appears to be a Gaillardia x grandiflora, commonly known as the blanket flower. It is characterized by its daisy-like appearance with a prominent yellow center and petals that range in color from white to pink to red.
```

4) Continue the chat by passing another image.
```
> Hello!
How can I assist you today?
> \image https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg <|image|>What type of flower is this?        
The flower shown in the image appears to be a Gaillardia x grandiflora, commonly known as the blanket flower. It is characterized by its daisy-like appearance with a prominent yellow center and petals that range in color from white to pink to red.
> \image https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg <|image|>What mountain is this?
The image appears to be of Mount Washington, the highest peak in the Northeastern United States. It is located in the White Mountains of New Hampshire and is known for its extreme weather conditions, including high winds and low temperatures.
```

## HTTP server
You can find this example [here](../examples/server/llama_vision.py).

We support an OpenAI compatible HTTP API for vision models. This example demonstrates sending a chat completion request with an image.

> Note: The image_url may be either a path, URL, or a base64 encoded string.

---

**Image:**
<img src="https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg" alt="Mount Washington" width = "1000" height = "666">
<h6><a href = "https://www.nhmagazine.com/mount-washington/">Credit</a></h6>

**Prompt:**
```
<|image|>What is shown in this image? Write a detailed response analyzing the scene.
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
cargo run --release --features ... -- --port 1234 --isq Q4K vision-plain -m lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k -a vllama
```

2) Send a request

```py
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:1234/v1/"

completion = client.chat.completions.create(
    model="llama-vision",
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
                    "text": "<|image|>What is shown in this image? Write a detailed response analyzing the scene.",
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
You can find this example [here](../mistralrs/examples/llama_vision/main.rs).

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};

const MODEL_ID: &str = "lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k";

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new(MODEL_ID, VisionLoaderType::VLlama)
            .with_isq(IsqType::Q4K)
            .with_logging()
            .build()
            .await?;

    let bytes = match reqwest::blocking::get(
        "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new().add_vllama_image_message(
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

---

## Python
You can find this example [here](../examples/python/llama_vision.py).

This example demonstrates loading and sending a chat completion request with an image.

> Note: the image_url may be either a path, URL, or a base64 encoded string.

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

MODEL_ID = "lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k"

runner = Runner(
    which=Which.VisionPlain(
        model_id="MODEL_ID",
        arch=VisionArchitecture.VLlama,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="llama-vision",
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
                        "text": "<|image|>What is shown in this image? Write a detailed response analyzing the scene.",
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