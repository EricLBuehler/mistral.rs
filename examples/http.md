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