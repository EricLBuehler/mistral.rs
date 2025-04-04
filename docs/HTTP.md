# HTTP server

Mistral.rs provides a lightweight OpenAI API compatible HTTP server based on [axum](https://github.com/tokio-rs/axum). The request and response formats are supersets of the OpenAI API.

The API consists of the following endpoints. They can be viewed in your browser interactively by going to `http://localhost:<port>/docs`.

## Additional object keys

To support additional features, we have extended the completion and chat completion request objects. Both have the same keys added:

- `top_k`: `int` | `null`. If non null, it is only relevant if positive.
- `grammar`: `{"type" : "regex" | "lark" | "json_schema" | "llguidance", "value": string}` or `null`. Grammar to use. This is mutually exclusive to the OpenAI-compatible `response_format`.
- `min_p`: `float` | `null`. If non null, it is only relevant if 1 >= min_p >= 0.


## `POST`: `/v1/chat/completions`
Process an OpenAI compatible request, returning an OpenAI compatible response when finished. Please find the official OpenAI API documentation [here](https://platform.openai.com/docs/api-reference/chat). To control the interval keep-alive messages are sent, set the `KEEP_ALIVE_INTERVAL` environment variable to the desired time in ms.

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

## `GET`: `/v1/models`
Returns the running models. 

Example with `curl`:
```bash
curl http://localhost:<port>/v1/models
```

## `GET`: `/` or `/health`
Returns the server health.

Example with `curl`:
```bash
curl http://localhost:<port>/health
```

## `GET`: `/docs`
Returns OpenAPI API docs via SwaggerUI.

Example with `curl`:
```bash
curl http://localhost:<port>/docs
```

## `POST`: `/v1/completions`
Process an OpenAI compatible completions request, returning an OpenAI compatible response when finished. Please find the official OpenAI API documentation [here](https://platform.openai.com/docs/api-reference/completions). 

To send a request with the Python `openai` library:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "EMPTY"
)

completion = client.completions.create(
    model="mistral",
    prompt="What is Rust?",
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
)

print(completion.choices[0].message)
```

Or with `curl`:
```bash
curl http://localhost:8080/v1/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPTY" \
-d '{
"model": "",
"prompt": "What is Rust?"
}'
```


## `POST`: `/re_isq`
Reapply ISQ to the model if possible. Pass the names as a JSON object with the key `ggml_type` to a string (the quantization level).

Example with `curl`:
```bash
curl http://localhost:<port>/re_isq -H "Content-Type: application/json" -H "Authorization: Bearer EMPTY" -d '{"ggml_type":"Q4K"}'
```
