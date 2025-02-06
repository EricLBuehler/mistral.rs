# DeepSeek V3: [`deepseek-ai/DeepSeek-V3`](https://huggingface.co/deepseek-ai/DeepSeek-V3), [`deepseek-ai/DeepSeek-R1`](https://huggingface.co/deepseek-ai/DeepSeek-R1)

The DeepSeek V3 is a mixture of expert (MoE) model.

```
./mistralrs-server --isq Q4K -i plain -m deepseek-ai/DeepSeek-R1
```

> [!NOTE]
> The non-distill versions of the DeepSeek R1 models share the DeepSeek V3 architecture.

> [!NOTE]
> This models supports MoQE which can be activated in the ISQ organization parameter within the various APIs, as demonstrated below:

```
./mistralrs-server --isq Q4K -i plain -m deepseek-ai/DeepSeek-R1 --organization moqe
```

## Running the distill models

The various [distillation](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) models can be run out of the box.
```
./mistralrs-server -i --isq Q4K plain -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B
./mistralrs-server -i --isq Q4K plain -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
./mistralrs-server -i --isq Q4K plain -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

## HTTP API

```
./mistralrs-server --isq Q4K --port 1234 plain -m deepseek-ai/DeepSeek-R1
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
        model="deepseekv2",
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

## Python API
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
        model="mistral",
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

## Rust API
You can find this example [here](../mistralrs/examples/deepseekr1/main.rs).

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
