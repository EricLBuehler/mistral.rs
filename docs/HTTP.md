# HTTP server

Mistral.rs provides a lightweight OpenAI API compatible HTTP server based on [axum](https://github.com/tokio-rs/axum). The request and response formats are supersets of the OpenAI API.

The API consists of the following endpoints. They can be viewed in your browser interactively by going to `http://localhost:<port>/docs`.

> ℹ️  Besides the HTTP endpoints described below, `mistralrs serve` can also expose the same functionality via the **MCP protocol**.
> Enable it with `--mcp-port <port>` and see [MCP/server.md](MCP/server.md) for details.

## Additional object keys

To support additional features, we have extended the completion and chat completion request objects. Both have the same keys added:

- `top_k`: `int` | `null`. If non null, it is only relevant if positive.
- `grammar`: `{"type" : "regex" | "lark" | "json_schema" | "llguidance", "value": string}` or `null`. Grammar to use. This is mutually exclusive to the OpenAI-compatible `response_format`.
- `min_p`: `float` | `null`. If non null, it is only relevant if 1 >= min_p >= 0.
- `enable_thinking`: `bool`, default to `false`. Enable thinking for models that support it.
- `truncate_sequence`: `bool` | `null`. When `true`, requests that exceed the model context length will be truncated instead of rejected; otherwise the server returns a validation error. Embedding requests truncate tokens at the end of the prompt, while chat/completion requests truncate tokens at the start of the prompt.
- `repetition_penalty`: `float` | `null`. Penalty for repeating tokens. This is distinct from `frequency_penalty` and `presence_penalty` - it applies a direct multiplicative penalty to repeated token logits.
- `web_search_options`: `object` | `null`. Enable web search integration (see [WEB_SEARCH.md](WEB_SEARCH.md)). Contains optional fields: `search_context_size` ("low", "medium", "high"), `user_location` (object with location info), `search_description` (override search tool description), `extract_description` (override extraction tool description).
- `reasoning_effort`: `string` | `null`. For Harmony-format models (like GPT-OSS), controls the depth of reasoning: `"low"`, `"medium"`, or `"high"`.
- `dry_multiplier`: `float` | `null`. DRY (Don't Repeat Yourself) sampling multiplier. Controls the strength of the anti-repetition penalty.
- `dry_base`: `float` | `null`. DRY sampling base value.
- `dry_allowed_length`: `int` | `null`. DRY sampling allowed length before penalty applies.
- `dry_sequence_breakers`: `array of strings` | `null`. Tokens that reset the DRY penalty sequence.

## Response Extensions

The response objects include additional fields beyond the standard OpenAI API:

### Harmony Mode Responses

For models using Harmony format (like GPT-OSS), responses may include additional reasoning content:

- `reasoning_content`: `string` | `null`. Chain-of-thought reasoning from Harmony-format models. This field contains the model's internal analysis and commentary that led to the final response. It is separate from the main `content` field.

When streaming, `reasoning_content` appears in the `delta` object alongside `content`.

**Example response:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 42.",
      "reasoning_content": "Let me analyze this step by step..."
    }
  }]
}
```

## Model Parameter Validation

Mistral.rs validates that the `model` parameter in API requests matches the model that was actually loaded by the server. This ensures requests are processed by the correct model and prevents confusion.

**Behavior:**
- If the `model` parameter matches the loaded model name, the request proceeds normally
- If the `model` parameter doesn't match, the request fails with an error message indicating the mismatch
- The special model name `"default"` can be used to bypass this validation entirely

**Examples:**
- ✅ Request with `"model": "meta-llama/Llama-3.2-3B-Instruct"` when `meta-llama/Llama-3.2-3B-Instruct` is loaded -> **succeeds**
- ❌ Request with `"model": "gpt-4"` when `mistral-7b-instruct` is loaded -> **fails**
- ✅ Request with `"model": "default"` regardless of loaded model -> **always succeeds**

**Usage:** Use `"default"` in the model field when you need to satisfy API clients that require a model parameter but don't need to specify a particular model. This is demonstrated in all the examples below.

## `POST`: `/v1/chat/completions`
Process an OpenAI compatible request, returning an OpenAI compatible response when finished. Please find the official OpenAI API documentation [here](https://platform.openai.com/docs/api-reference/chat). To control the interval keep-alive messages are sent, set the `KEEP_ALIVE_INTERVAL` environment variable to the desired time in ms.

To send a request with the Python `openai` library:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1", # "http://<Your api-server IP>:port"
    api_key = "EMPTY"
)

completion = client.chat.completions.create(
model="default",
messages=[
    {"role": "system", "content": "You are Mistral.rs, an AI assistant."},
    {"role": "user", "content": "Write a story about Rust error handling."}
]
)

print(completion.choices[0].message)
```

Or with `curl`:
```bash
curl http://localhost:1234/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPTY" \
-d '{
"model": "default",
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

> ℹ️ Requests whose prompt exceeds the model's maximum context length now fail unless you opt in to truncation. Set `"truncate_sequence": true` to drop the oldest prompt tokens while reserving room (equal to `max_tokens` when provided, otherwise one token) for generation. Specifically, tokens from the front of the prompt are dropped.

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

### Completions-specific parameters

In addition to the common parameters listed above, the completions endpoint supports:

- `best_of`: `int` | `null`. Generate `best_of` completions server-side and return the best one (the one with the highest log probability per token). When used with `n`, `best_of` must be greater than `n`.
- `echo`: `bool`, default `false`. Echo back the prompt in addition to the completion.

To send a request with the Python `openai` library:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1", # "http://<Your api-server IP>:port"
    api_key = "EMPTY"
)

completion = client.completions.create(
    model="default",
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
curl http://localhost:1234/v1/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPTY" \
-d '{
"model": "default",
"prompt": "What is Rust?"
}'
```

> ℹ️ The `truncate_sequence` flag behaves the same way for the completions endpoint: keep it `false` (default) to receive a validation error, or set it to `true` to trim the prompt automatically.

## `POST`: `/v1/embeddings`
Serve an embedding model (for example, EmbeddingGemma) to enable this endpoint:

```bash
mistralrs serve -m google/embeddinggemma-300m
```

In multi-model mode, include an `Embedding` entry in your selector config to expose it alongside chat models.

Create vector embeddings via the OpenAI-compatible endpoint. Supported request fields:

- `input`: a single string, an array of strings, an array of token IDs (`[123, 456]`), or a batch of token arrays (`[[...], [...]]`).
- `encoding_format`: `"float"` (default) returns arrays of `f32`; `"base64"` returns Base64 strings.
- `dimensions`: currently unsupported; providing it yields a validation error.
- `truncate_sequence`: `bool`, default `false`. Set to `true` to clip over-length prompts instead of receiving a validation error.

> ℹ️ Requests whose prompt exceeds the model's maximum context length now fail unless you opt in to truncation. Embedding requests truncate tokens from the end of the prompt.

Example (Python `openai` client):

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="EMPTY",
)

result = client.embeddings.create(
    model="default",
    input=[
        "Embeddings capture semantic relationships between texts.",
        "What is graphene?",
    ],
    truncate_sequence=True,
)

for item in result.data:
    print(item.index, len(item.embedding))
```

Example with `curl`:

```bash
curl http://localhost:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "default",
    "input": ["graphene conductivity", "superconductor basics"],
    "encoding_format": "base64",
    "truncate_sequence": false
  }'
```

Responses follow the OpenAI schema: `object: "list"`, `data[*].embedding` containing either float arrays or Base64 strings depending on `encoding_format`, and a `usage` block (`prompt_tokens`, `total_tokens`). At present those counters report `0` because token accounting for embeddings is not yet implemented.

## `POST`: `/v1/images/generations`
Generate images using diffusion models (like FLUX). First, serve a diffusion model:

```bash
mistralrs serve -m black-forest-labs/FLUX.1-schnell
```

Supported request fields:
- `model`: Model identifier (use `"default"` to bypass validation)
- `prompt`: Text description of the image to generate
- `n`: Number of images to generate (default: 1)
- `response_format`: `"url"` or `"b64_json"` (default: `"url"`)
- `height`: Image height in pixels (default: 720)
- `width`: Image width in pixels (default: 1280)

Example with Python:

```python
import openai
import base64

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="EMPTY",
)

response = client.images.generate(
    model="default",
    prompt="A majestic snow-covered mountain at sunset",
    n=1,
    response_format="b64_json",
    size="1280x720",  # width x height
)

# Save the generated image
image_data = base64.b64decode(response.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_data)
```

Example with `curl`:

```bash
curl http://localhost:1234/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "default",
    "prompt": "A majestic snow-covered mountain at sunset",
    "n": 1,
    "response_format": "b64_json",
    "height": 720,
    "width": 1280
  }'
```

## `POST`: `/v1/audio/speech`
Generate speech from text using speech models (like Dia). First, serve a speech model:

```bash
mistralrs serve -m nari-labs/Dia-1.6B
```

Supported request fields:
- `model`: Model identifier (use `"default"` to bypass validation)
- `input`: Text to convert to speech. For Dia models, use speaker tags like `[S1]` and `[S2]` to control multiple voices
- `response_format`: `"wav"` or `"pcm"` (only these formats are supported)

> Note: The `voice` and `instructions` fields from the OpenAI API are currently ignored.

Example with Python:

```python
import requests

response = requests.post(
    "http://localhost:1234/v1/audio/speech",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    },
    json={
        "model": "default",
        "input": "[S1] Hello, how are you today? [S2] I'm doing great, thanks for asking!",
        "response_format": "wav",
    },
)

# Save the audio file
with open("output.wav", "wb") as f:
    f.write(response.content)
```

Example with `curl`:

```bash
curl http://localhost:1234/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "default",
    "input": "[S1] Dia is an open weights text to dialogue model. [S2] Try it now!",
    "response_format": "wav"
  }' \
  --output output.wav
```

The response is raw audio data with the appropriate `Content-Type` header (`audio/wav` for WAV format, `audio/pcm` for PCM format).

## `POST`: `/v1/responses`
Create a response using the OpenAI-compatible Responses API. Please find the official OpenAI API documentation [here](https://platform.openai.com/docs/api-reference/responses). 

To send a request with the Python `openai` library:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key = "EMPTY"
)

# First turn
resp1 = client.responses.create(
    model="default",
    input="Apples are delicious!"
)
print(resp1.output_text)

# Follow-up - no need to resend the first message
resp2 = client.responses.create(
    model="default",
    previous_response_id=resp1.id,
    input="Can you eat them?"
)
print(resp2.output_text)
```

Or with `curl`:
```bash
curl http://localhost:1234/v1/responses \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPTY" \
-d '{
"model": "default",
"input": "Tell me about Rust programming"
}'

# Follow-up using previous_response_id
curl http://localhost:1234/v1/responses \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPTY" \
-d '{
"model": "default",
"previous_response_id": "resp_12345-uuid-here",
"input": "What makes it memory safe?"
}'
```

The API also supports multimodal inputs (images, audio) and streaming responses by setting `"stream": true` in the request JSON.

> ℹ️ The Responses API forwards `truncate_sequence` to underlying chat completions. Enable it if you want over-length conversations to be truncated rather than rejected.

## `GET`: `/v1/responses/{response_id}`
Retrieve a previously created response by its ID.

Example with `curl`:
```bash
curl http://localhost:1234/v1/responses/resp_12345-uuid-here \
-H "Authorization: Bearer EMPTY"
```

## `DELETE`: `/v1/responses/{response_id}`
Delete a stored response and its associated conversation history.

Example with `curl`:
```bash
curl -X DELETE http://localhost:1234/v1/responses/resp_12345-uuid-here \
-H "Authorization: Bearer EMPTY"
```

## `POST`: `/re_isq`
Reapply ISQ to the model if possible. Pass the names as a JSON object with the key `ggml_type` to a string (the quantization level).

Example with `curl`:
```bash
curl http://localhost:<port>/re_isq -H "Content-Type: application/json" -H "Authorization: Bearer EMPTY" -d '{"ggml_type":"4"}'
```

## Model Management Endpoints

These endpoints allow dynamic management of loaded models, enabling you to free memory by unloading models and reload them on demand.

### `POST`: `/v1/models/unload`

Unload a model from memory while preserving its configuration for later reload. The model can be reloaded manually or will auto-reload when a request is sent to it.

**Request body:**
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct"
}
```

**Response:**
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "status": "unloaded"
}
```

Example with `curl`:
```bash
curl -X POST http://localhost:1234/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.2-3B-Instruct"}'
```

### `POST`: `/v1/models/reload`

Manually reload a previously unloaded model. This is also triggered automatically when a request is sent to an unloaded model.

**Request body:**
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct"
}
```

**Response:**
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "status": "loaded"
}
```

Example with `curl`:
```bash
curl -X POST http://localhost:1234/v1/models/reload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.2-3B-Instruct"}'
```

### `POST`: `/v1/models/status`

Get the current status of a specific model.

**Request body:**
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct"
}
```

**Response:**
```json
{
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "status": "loaded"
}
```

Example with `curl`:
```bash
curl -X POST http://localhost:1234/v1/models/status \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.2-3B-Instruct"}'
```

### Status Values

The `status` field in responses can be one of:

| Status | Description |
|--------|-------------|
| `loaded` | Model is loaded and ready to serve requests |
| `unloaded` | Model is unloaded but can be reloaded |
| `reloading` | Model is currently being reloaded |
| `not_found` | Model ID not recognized |
| `no_loader_config` | Model cannot be reloaded (missing loader configuration) |
| `internal_error` | An internal error occurred (check `error` field for details) |

When an error occurs, the response may include an `error` field with additional details:
```json
{
  "model_id": "unknown-model",
  "status": "not_found",
  "error": null
}
```

### Auto-Reload Behavior

When a request (e.g., chat completion) is sent to an unloaded model, the model will automatically reload before processing the request. This enables a "lazy loading" pattern where models are only loaded when needed, helping manage GPU memory efficiently.

### Models List with Status

The `/v1/models` endpoint includes a `status` field for each model:

```bash
curl http://localhost:1234/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    },
    {
      "id": "meta-llama/Llama-3.2-3B-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local",
      "status": "loaded"
    }
  ]
}
```
