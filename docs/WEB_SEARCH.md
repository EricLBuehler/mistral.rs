# Web search tool in mistral.rs

mistral.rs is compatible with OpenAI's `web_search_options` parameter! Once enabled, this allows web searching for models.

This works with all models that support [tool calling](TOOL_CALLING.md). However, your mileage may vary depending on the specific model. The following models work during testing and are recommended for usage:
- Hermes 3 3b/8b
- Mistral 3 24b
- Llama 4 Scout/Maverick

Besides tool calling and parsing of web content, we also use an embedding model to select the most relevant search results.

You can use the web search tool in all the APIs: Python, Rust, and server.

## Specifying a custom embedding model

Internally, we use a BERT model (Snowflake/snowflake-arctic-embed-l-v2.0)[https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0] (0.6b parameters = ~2.3GB) to select the most relevant search results. You can specify a custom BERT model by providing a Hugging Face model ID in the various APIs:

- Rust: `with_search` in the builder
- Python: `search_bert_model` in the Runner
- Server: `search-bert-model` before the model type selector (`plain`/`vision-plain`)

## HTTP server
**Be sure to add `--enable-search`!**

Here are some examples using various models:
```
./mistralrs-server --enable-search --port 1234 --isq q4k --jinja-explicit chat_templates/mistral_small_tool_call.jinja vision-plain -m mistralai/Mistral-Small-3.1-24B-Instruct-2503 -a mistral3
```

```
./mistralrs-server --enable-search --port 1234 --isq q4k plain -m NousResearch/Hermes-3-Llama-3.1-8B
```

```py
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

messages = [
    {
        "role": "user",
        "content": "Can you show me some code using mistral.rs for running Llama 3.2 Vision?",
    }
]

completion = client.chat.completions.create(
    model="llama-3.1",
    messages=messages,
    tool_choice="auto",
    max_tokens=1024,
    web_search_options={},
)

# print(completion.usage)
print(completion.choices[0].message.content)

if completion.choices[0].message.tool_calls is not None:
    # Should never happen.
    tool_called = completion.choices[0].message.tool_calls[0].function
    print(tool_called)
```


## Python API
```py
from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    WebSearchOptions,
)

runner = Runner(
    which=Which.Plain(
        model_id="NousResearch/Hermes-3-Llama-3.1-8B",
        arch=Architecture.Llama,
    ),
    enable_search=True,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": "Can you show me some code using mistral.rs for running Llama 3.2 Vision?",
            }
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
        web_search_options=WebSearchOptions(
            search_context_size=None, user_location=None
        ),
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

## Rust API
```rust
use anyhow::Result;
use mistralrs::{
    BertEmbeddingModel, IsqType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
    WebSearchOptions,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("NousResearch/Hermes-3-Llama-3.1-8B")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_search(BertEmbeddingModel::default())
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "What is the weather forecast for Boston?",
    );
    let messages =
        RequestBuilder::from(messages).with_web_search_options(WebSearchOptions::default());

    let response = model.send_chat_request(messages).await?;

    println!("What is the weather forecast for Boston?\n\n");
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
```
