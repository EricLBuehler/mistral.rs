# DeepSeek V2: [`deepseek-ai/DeepSeek-V2-Lite`](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)

The DeepSeek V2 is a mixture of expert (MoE) model featuring ["Multi-head Latent Attention"](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite#5-model-architecture).

- Context length of **32k tokens** (Lite model), **128k tokens** (full model)
- 64 routed experts (Lite model), 160 routed experts (full model)

## Quick Start

```bash
mistralrs run --isq 4 -m deepseek-ai/DeepSeek-V2-Lite
```

> Note: This model supports MoQE which can be activated in the ISQ organization parameter within the various APIs, as demonstrated below:

```bash
mistralrs run --isq 4 -m deepseek-ai/DeepSeek-V2-Lite --isq-organization moqe
```

## HTTP API

```bash
mistralrs serve --isq 4 -p 1234 -m deepseek-ai/DeepSeek-V2-Lite
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
        model_id="deepseek-ai/DeepSeek-V2-Lite",
        arch=Architecture.DeepseekV2,
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
    let model = TextModelBuilder::new("deepseek-ai/DeepSeek-V2-Lite")
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
