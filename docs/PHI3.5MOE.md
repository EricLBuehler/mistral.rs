# Phi 3.5 Model: [`microsoft/Phi-3.5-MoE-instruct`](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)

The Phi 3.5 MoE model is a 16x3.8B parameter decoder-only text-to-text mixture of expert LLM.

- Context length of **128k tokens**
- Trained on **4.9T tokens**
- 16 experts (16x3.8B parameters) with **6.6B active parameters**
- Expect inference performance of a 7B model

## About the MoE mechanism
1) Compute router gating logits
2) From the router gating logits, select the top-2 selected experts and the associated weights
3) The hidden states for each token in the sequence is computed by (if selected) applying the expert output to that token, and then weighting it. 
    - If multiple experts are selected for the token, then this becomes a weighted sum
    - The design is flexible: 2 or 1 experts can be selected, enabling dense or sparse gating

```bash
mistralrs run --isq 4 -m microsoft/Phi-3.5-MoE-instruct
```

> [!NOTE]
> This models supports MoQE which can be activated in the ISQ organization parameter within the various APIs, as demonstrated below:

```bash
mistralrs run --isq 4 -m microsoft/Phi-3.5-MoE-instruct --isq-organization moqe
```

## HTTP API

```bash
mistralrs serve --isq 4 -p 1234 -m microsoft/Phi-3.5-MoE-instruct
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
    )
    resp = completion.choices[0].message.content
    print(resp)
    messages.append({"role": "assistant", "content": resp})
```

## Python SDK
```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-MoE-instruct",
        arch=Architecture.Phi3_5MoE ,
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
