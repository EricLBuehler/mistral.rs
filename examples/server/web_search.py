from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

messages = [
    {
        "role": "user",
        "content": "Can you show me some code using mistral.rs for running Llama 3.2 Vision?",
    }
]

"""
Here are a few code examples using Mistral.rs to run Llama 3.2 Vision:

1. The Python binding of Mistral.rs can be installed using the following command:
```
pip install mistralrs-metal
```
Then you can use the following code to run the Llama 3.2 Vision model:
```python
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="llama-v3p2-90b-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://niche-museums.imgix.net/pioneer-history.jpeg?w=1600&h=800&fit=crop&auto=compress"
                        },
                    },
                    {
                        "type": "text",
                        "text": "<|image_1|> What is shown in this image? Write a detailed response analyzing the scene.",
                    },
                ],
            }
        ],
        max_tokens=1024,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
        stream=True,
    )
)
for item in res:
    print(item.choices[0].delta.content, end='')
```
This code runs the Llama 3.2 Vision model and provides a detailed response analyzing the scene in the given image. The code first downloads the model and then sends a chat completion request with the specified image and prompt.

2. Another code example to run the Llama 3.2 Vision model using `mistral.rs` in Rust:
```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};

const MODEL_ID: &str = "lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k";
const IMAGE_URL: &str = "https://niche-museums.imgix.net/pioneer-history.jpeg?w=1600&h=800&fit=crop&auto=compress";

fn main() -> Result<()> {
    let model = VisionModelBuilder::new(MODEL_ID, VisionLoaderType::VLlama)
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;
    
    let bytes = reqwest::blocking::get(IMAGE_URL)?.bytes()?;
    let image = image::load_from_memory(&bytes)?.to_luma();    
    
    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User, 
        "What is in this image? Please analyze this image.", 
        image, 
        &model, 
    );
    
    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
```
The code initializes the Llama 3.2 Vision model and sends an image message along with a text prompt to analyze the image. Finally, it prints the model's response.

These code examples demonstrate how to use Mistral.rs to run the Llama 3.2 Vision model and analyze images using Rust and Python.
"""

completion = client.chat.completions.create(
    model="llama-3.1", messages=messages, tool_choice="auto", max_tokens=1024
)

# print(completion.usage)
print(completion.choices[0].message.content)

tool_called = completion.choices[0].message.tool_calls[0].function
print(tool_called)
