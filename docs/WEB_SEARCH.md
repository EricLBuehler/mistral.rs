# Web search tool in mistral.rs

mistral.rs is compatible with OpenAI's `web_search_options` parameter! Once enabled, this allows web searching for models.

This works with all models that support [tool calling](TOOL_CALLING.md). However, your mileage may vary depending on the specific model. The following models work during testing and are recommended for usage:

- Hermes 3 3b/8b
- Mistral 3 24b
- Llama 4 Scout/Maverick
- Qwen 3 (⭐ Recommended!)

Web search is supported both in streaming and completion responses! This makes it easy to integrate and test out in interactive mode!

Besides tool calling and parsing of web content, we also use an embedding model to select the most relevant search results.

You can use the web search tool in all the APIs: Python, Rust, and server.

## Selecting a search embedding model

Internally, we now use [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) to embed documents for ranking. You can pick from the built-in reranker variants (currently just `embedding_gemma`) in every API:

- Rust: `with_search(SearchEmbeddingModel::EmbeddingGemma300M)` in the builder
- Python: `search_embedding_model="embedding_gemma"` in the Runner
- Server: `--search-embedding-model embedding_gemma` flag

## Specifying a custom search callback

By default, mistral.rs uses a DuckDuckGo-based search callback. To override this, you can provide your own search function:

- Rust: use `.with_search_callback(...)` on the model builder with an `Arc<dyn Fn(&SearchFunctionParameters) -> anyhow::Result<Vec<SearchResult>> + Send + Sync>`.
- Python: pass the `search_callback` keyword argument to `Runner`, which should be a function `def search_callback(query: str) -> List[Dict[str, str]]` returning a list of results with keys `"title"`, `"description"`, `"url"`, and `"content"`.

Example in Python:
```py
def search_callback(query: str) -> list[dict[str, str]]:
    # Implement your custom search logic here, returning a list of result dicts
    return [
        {
            "title": "Example Result",
            "description": "An example description",
            "url": "https://example.com",
            "content": "Full text content of the page",
        },
        # more results...
    ]

from mistralrs import Runner, Which, Architecture
runner = Runner(
    which=Which.Plain(model_id="YourModel/ID", arch=Architecture.Mistral),
    enable_search=True,
    search_callback=search_callback,
)
```

## HTTP server
**Be sure to add `--enable-search`!**

Here are some examples using various models. Note that this works for both streaming and completion requests, so interactive mode is featured here!

```bash
mistralrs run --enable-search --isq 4 -m Qwen/Qwen3-4B
```

```bash
mistralrs serve --enable-search -p 1234 --isq 4 --jinja-explicit chat_templates/mistral_small_tool_call.jinja -m mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

```bash
mistralrs run --enable-search --isq 4 -m NousResearch/Hermes-3-Llama-3.1-8B
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
    model="default",
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


## Python SDK
```py
from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    WebSearchOptions,
)

# Define a custom search callback if desired
def my_search_callback(query: str) -> list[dict[str, str]]:
    # Fetch or compute search results here
    return [
        {
            "title": "Mistral.rs GitHub",
            "description": "Official mistral.rs repository",
            "url": "https://github.com/EricLBuehler/mistral.rs",
            "content": "mistral.rs is a Rust binding for Mistral models...",
        },
    ]

runner = Runner(
    which=Which.Plain(
        model_id="NousResearch/Hermes-3-Llama-3.1-8B",
        arch=Architecture.Llama,
    ),
    enable_search=True,
    search_callback=my_search_callback,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
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

## Rust SDK
```rust
use anyhow::Result;
use mistralrs::{
    SearchEmbeddingModel, IsqType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
    WebSearchOptions,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("NousResearch/Hermes-3-Llama-3.1-8B")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_search(SearchEmbeddingModel::default())
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
