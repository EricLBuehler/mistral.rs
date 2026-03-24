# Qwen 3: [`collection`](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)

The Qwen 3 family is a collection of hybrid reasoning MoE and non-MoE models ranging from 0.6b to 235b parameters.

```bash
mistralrs run --isq 4 -m Qwen/Qwen3-8B
mistralrs run --isq 4 -m Qwen/Qwen3-30B-A3B
```

> Note: mistral.rs can load all [FP8 pre-quantized versions](https://huggingface.co/Qwen/Qwen3-14B-FP8) natively! Simply replace the model ID.

> Note: tool calling support is fully implemented for the Qwen 3 models, including agentic web search.

## Enabling thinking
The Qwen 3 models are hybrid reasoning models which can be controlled at inference-time. **By default, reasoning is enabled for these models.** To dynamically control this, it is recommended to either add `/no_think` or `/think` to your prompt. Alternatively, you can specify the `enable_thinking` flag as detailed by the API-specific examples.

## HTTP API
You can find a more detailed example demonstrating enabling/disabling thinking [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen3.py).

```bash
mistralrs serve --isq 4 -p 1234 -m Qwen/Qwen3-8B
```

```py
import openai

messages = []
prompt = input("Enter system prompt >>> ")
if len(prompt) > 0:
    messages.append({"role": "system", "content": prompt})


while True:
    prompt = input(">>> ")
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=256,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
        # enable_thinking=False,
    )
    resp = completion.choices[0].message.content
    print(resp)
    messages.append({"role": "assistant", "content": resp})
```

## Python SDK
You can find a more detailed example demonstrating enabling/disabling thinking [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="Qwen/Qwen3-8B",
        arch=Architecture.Qwen3,
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
    let model = TextModelBuilder::new("Qwen/Qwen3-8B")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let messages = TextMessages::new()
        // .enable_thinking(false)
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
