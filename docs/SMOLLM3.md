# SmolLM3: [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

SmolLM3 is a 3B parameter long-context hybrid reasoning language model. It supports 6 languages, advanced reasoning and long context. SmolLM3 is a fully open model that offers strong performance at the 3B–4B scale.

**Default, easiest:**
```bash
./mistralrs-server -i --isq 8 run -m HuggingFaceTB/SmolLM3-3B
```

**UQFF prequantized:**
```bash
./mistralrs-server -i run -m EricB/SmolLM3-3B-UQFF -f smollm33b-q4k-0.uqff
```

> Note: tool calling support is fully implemented for the SmolLM3 models, including agentic web search.

> Check out prequantized UQFF SmolLM3 here: https://huggingface.co/EricB/SmolLM3-3B-UQFF

## Enabling thinking
The SmolLM3 models are hybrid reasoning models which can be controlled at inference-time. **By default, reasoning is enabled for these models.** To dynamically control this, it is recommended to either add `/no_think` or `/think` to your prompt. Alternatively, you can specify the `enable_thinking` flag as detailed by the API-specific examples.

## HTTP API
You can find a more detailed example demonstrating enabling/disabling thinking [here](../examples/server/smollm3.py).

```
./mistralrs-server --isq 8 --port 1234 plain -m HuggingFaceTB/SmolLM3-3B
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

## Python API
You can find a more detailed example demonstrating enabling/disabling thinking [here](../examples/python/smollm3.py).

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

## Rust API
You can find a more detailed example demonstrating enabling/disabling thinking [here](../mistralrs/examples/smollm3/main.rs).

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
