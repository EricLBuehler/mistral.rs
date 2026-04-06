# SmolLM3: [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

SmolLM3 is a 3B parameter long-context hybrid reasoning language model. It supports 6 languages, advanced reasoning and long context. SmolLM3 is a fully open model that offers strong performance at the 3B-4B scale.

## Quick Start

```bash
mistralrs run --isq 8 -m HuggingFaceTB/SmolLM3-3B
```

**UQFF prequantized:**
```bash
mistralrs run -m EricB/SmolLM3-3B-UQFF --from-uqff smollm33b-q4k-0.uqff
```

> Note: tool calling support is fully implemented for the SmolLM3 models, including agentic web search.

> Check out prequantized UQFF SmolLM3 here: https://huggingface.co/EricB/SmolLM3-3B-UQFF

## Enabling thinking
The SmolLM3 models are hybrid reasoning models which can be controlled at inference-time. **By default, reasoning is enabled for these models.** To dynamically control this, it is recommended to either add `/no_think` or `/think` to your prompt. Alternatively, you can specify the `enable_thinking` flag as detailed by the API-specific examples.

## HTTP API
You can find a more detailed example demonstrating enabling/disabling thinking [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/smollm3.py).

```bash
mistralrs serve --isq 8 -p 1234 -m HuggingFaceTB/SmolLM3-3B
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
You can find a more detailed example demonstrating enabling/disabling thinking [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/smollm3.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="HuggingFaceTB/SmolLM3-3B",
        arch=Architecture.SmolLm3,
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
        # enable_thinking=False,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

## Rust SDK
You can find a more detailed example demonstrating enabling/disabling thinking [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/text_models/main.rs).

```rust
use anyhow::Result;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("HuggingFaceTB/SmolLM3-3B")
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let messages = TextMessages::new()
        // .enable_thinking(false)
        .add_message(TextMessageRole::User, "Hello!");

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
```
