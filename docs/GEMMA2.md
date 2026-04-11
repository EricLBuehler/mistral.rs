# Gemma 2: [`collection`](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)

The Gemma 2 models are a family of text-to-text decoder-only LLMs.

## Quick Start

```bash
mistralrs run --isq 4 -m google/gemma-2-9b-it
```

## HTTP API

```bash
mistralrs serve --isq 4 -p 1234 -m google/gemma-2-9b-it
```

```py
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Tell me a story about the Rust type system."}
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

## Python SDK

```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="google/gemma-2-9b-it",
        arch=Architecture.Gemma2,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
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

## Rust SDK

```rust
use anyhow::Result;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("google/gemma-2-9b-it")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Hello!");

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
```
