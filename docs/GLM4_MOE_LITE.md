# GLM-4.7-Flash (MoE): [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash)

GLM-4.7-Flash is a mixture of experts (MoE) model from the GLM family with MLA (Multi-head Latent Attention) architecture.

## Quick Start

```bash
mistralrs run --isq 4 -m zai-org/GLM-4.7-Flash
```

## HTTP API

```bash
mistralrs serve --isq 4 -p 1234 -m zai-org/GLM-4.7-Flash
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
        model_id="zai-org/GLM-4.7-Flash",
        arch=Architecture.GLM4MoeLite,
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
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/text_models/main.rs).

```rust
use anyhow::Result;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("zai-org/GLM-4.7-Flash")
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
