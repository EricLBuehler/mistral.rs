# Qwen 3 Next: [`collection`](https://huggingface.co/collections/Qwen/qwen3-coder-next)

Qwen3-Coder-Next is a coding-focused language model using a hybrid Gated Delta Network (GDN) + full attention architecture with Mixture of Experts. With only 3B activated parameters (80B total), it achieves performance comparable to models with 10-20x more active parameters. It supports a 256K context window.

> Note: mistral.rs can load the [FP8 pre-quantized version](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8) natively! Simply replace the model ID.

```bash
mistralrs run --isq 4 -m Qwen/Qwen3-Coder-Next
```

GGUF quantized models are also supported:

```bash
mistralrs run --format gguf -m Qwen/Qwen3-Coder-Next-GGUF -f <filename>
```

## HTTP API
You can find a more detailed example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen3_next.py).

```bash
mistralrs serve --isq 4 -p 1234 -m Qwen/Qwen3-Coder-Next
```

```py
import openai

client = openai.OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

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
You can find a more detailed example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3_next.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="Qwen/Qwen3-Coder-Next",
        arch=Architecture.Qwen3Next,
    ),
    in_situ_quant="Q4K",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Write a Python function to compute fibonacci numbers."}
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
You can find a more detailed example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/text_models/main.rs).

```rust
use anyhow::Result;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("Qwen/Qwen3-Coder-Next")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::User,
            "Write a Python function to compute fibonacci numbers.",
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
