# HTTP server

Mistral.rs provides a lightweight OpenAI API compatible HTTP server based on [axum](https://github.com/tokio-rs/axum). The request and response formats are supersets of the OpenAI API, and more details can be found [here](https://ericlbuehler.github.io/mistral.rs/mistralrs_server/openai/struct.ChatCompletionRequest.html) for requests and [here](https://ericlbuehler.github.io/mistral.rs/mistralrs_core/struct.ChatCompletionResponse.html) for responses.

The API consists of the following endpoints.

## `POST`: `/v1/chat/completions`
Process an OpenAI compatible request, returning an OpenAI compatible response when finished. Please find the official OpenAI API documentation [here](https://platform.openai.com/docs/api-reference/chat).

To send a request with the Python `openai` library:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "EMPTY"
)

completion = client.chat.completions.create(
model="",
messages=[
    {"role": "system", "content": "You are Mistral.rs, an AI assistant."},
    {"role": "user", "content": "Write a story about Rust error handling."}
]
)

print(completion.choices[0].message)
```

Or with `curl`:
```bash
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPTY" \
-d '{
"model": "",
"messages": [
{
    "role": "system",
    "content": "You are Mistral.rs, an AI assistant."
},
{
    "role": "user",
    "content": "Write a story about Rust error handling."
}
]
}'
```

A streaming request can also be created by setting `"stream": true` in the request JSON. Please see [this](https://cookbook.openai.com/examples/how_to_stream_completions) guide.

## Request
### `ChatCompletionRequest`
OpenAI compatible request.
```rust
pub struct ChatCompletionRequest {
    pub messages: Vec<Message>,
    pub model: String,
    pub logit_bias: Option<HashMap<u32, f32>>,
    // Default false
    pub logprobs: bool,
    pub top_logprobs: Option<usize>,
    pub max_tokens: Option<usize>,
    // Default 1
    pub n: usize,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub stop: Option<StopTokens>,
    // Default 1
    pub temperature: Option<f64>,
    // Default 1
    pub top_p: Option<f64>,
    // Default 32
    pub top_k: Option<usize>,
    pub stream: bool,
}
```

### `Message`
Message with role of either `user`, `system` or `assistant`.
```rust
pub struct Message {
    pub content: String,
    pub role: String,
    pub name: Option<String>,
}
```

### `StopTokens`
Stop tokens. Each item in a `Multi` variant should represent one token.
```rust
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
    MultiId(Vec<u32>),
    SingleId(u32),
}
```

## Response

### `ChatCompletionResponse`
The OpenAI compatible chat completion response.
```rust
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u128,
    pub model: &'static str,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: ChatCompletionUsage,
}
```


### `Choice`
An individual choice, containing a `ResponseMessage` and maybe `Logprobs`.
```rust
pub struct Choice {
    pub finish_reason: String,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}
```

### `ResponseMessage`
```rust
pub struct ResponseMessage {
    pub content: String,
    pub role: String,
}
```

### `Logprobs`
Logprobs and top logprobs for each token.
```rust
pub struct Logprobs {
    pub content: Option<Vec<ResponseLogprob>>,
}
```

### `ResponseLogprob`
Logprobs and top logprobs for each token, with corresponding bytes. Top logprobs are ordered in descending probability.
```rust
pub struct ResponseLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}
```

### `TopLogprob`
```rust
pub struct TopLogprob {
    pub token: u32,
    pub logprob: f32,
    pub bytes: String,
}
```

### `ChatCompletionUsage`
```rust
pub struct ChatCompletionUsage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub avg_tok_per_sec: f32,
    pub avg_prompt_tok_per_sec: f32,
    pub avg_compl_tok_per_sec: f32,
}
```