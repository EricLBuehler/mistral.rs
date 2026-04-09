# Phi 3.5 MoE Model: [`microsoft/Phi-3.5-MoE-instruct`](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)

The Phi 3.5 MoE model is a 16x3.8B parameter decoder-only text-to-text mixture of expert LLM.

- Context length of **128k tokens**
- Trained on **4.9T tokens**
- 16 experts (16x3.8B parameters) with **6.6B active parameters**
- Expect inference performance of a 7B model

> [!NOTE]
> This model supports MoQE which can be activated in the ISQ organization parameter within the various APIs, as demonstrated below.

## Quick Start

```bash
mistralrs run -m microsoft/Phi-3.5-MoE-instruct --isq 4 -i "Tell me a story about the Rust type system."
```

With MoQE:

```bash
mistralrs run -m microsoft/Phi-3.5-MoE-instruct --isq 4 --isq-organization moqe -i "Tell me a story about the Rust type system."
```

## HTTP API

1) Start the server

```bash
mistralrs serve -m microsoft/Phi-3.5-MoE-instruct --isq 4 -p 1234
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
            "content": "Tell me a story about the Rust type system.",
        },
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
        model_id="microsoft/Phi-3.5-MoE-instruct",
        arch=Architecture.Phi3_5MoE,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
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
    let model = TextModelBuilder::new("microsoft/Phi-3.5-MoE-instruct")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
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
