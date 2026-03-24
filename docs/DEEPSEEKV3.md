# DeepSeek V3: [`deepseek-ai/DeepSeek-V3`](https://huggingface.co/deepseek-ai/DeepSeek-V3), [`deepseek-ai/DeepSeek-R1`](https://huggingface.co/deepseek-ai/DeepSeek-R1)

The DeepSeek V3 is a mixture of expert (MoE) model.

```bash
mistralrs run --isq 4 -m deepseek-ai/DeepSeek-R1
```

> [!NOTE]
> The non-distill versions of the DeepSeek R1 models share the DeepSeek V3 architecture.

> [!NOTE]
> This model supports MoQE which can be activated in the ISQ organization parameter within the various APIs, as demonstrated below:

```bash
mistralrs run --isq 4 -m deepseek-ai/DeepSeek-R1 --isq-organization moqe
```

## Running the distill models

The various [distillation](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) models can be run out of the box.
```bash
mistralrs run --isq 4 -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B
mistralrs run --isq 4 -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
mistralrs run --isq 4 -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

## HTTP API

```bash
mistralrs serve --isq 4 -p 1234 -m deepseek-ai/DeepSeek-R1
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
        model_id="deepseek-ai/DeepSeek-R1",
        arch=Architecture.DeepseekV3,
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
    let model = TextModelBuilder::new("deepseek-ai/DeepSeek-R1")
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
