# `mistralrs` API

## `Runner`

### `send_chat_completion_request(self, request: ChatCompletionRequest) -> str | ChatCompletionStreamer`
Send an OpenAI compatible request, returning OpenAI compatible object or a streamer which returns OpenAI compatible object chunks.

### `send_completion_request(self, request: CompletionRequest) -> str`
Send an OpenAI compatible request, returning an OpenAI compatible object.

## `ChatCompletionRequest`
Request is a class with a constructor which accepts the following arguments. It is used to create a chat completion request to pass to `send_chat_completion_request`.

- `messages: list[dict[str, str]]`
- `model: str`
- `logit_bias: dict[int, float]`
- `logprobs: bool`
- `top_logprobs: usize | None`
- `max_tokens: usize | None`
- `n_choices: usize`
- `presence_penalty: float | None`
- `frequency_penalty: float | None`
- `stop_token_ids: list[int] | None`
- `temperature: float | None`
- `top_p: float | None`
- `top_k: usize | None`
- `stream: bool = False`

`ChatCompletionRequest(messages, model, logprobs = false, n_choices = 1, logit_bias = None, top_logprobs = None, max_tokens = None, presence_penalty = None, frequency_penalty = None, stop_token_ids = None, temperature = None, top_p = None, top_k = None, stream = False)`

## `CompletionRequest`
Request is a class with a constructor which accepts the following arguments. It is used to create a chat completion request to pass to `send_completion_request`.

- `prompt: str`
- `model: str`
- `best_of: int`
- `echo_prompt: bool = False`
- `logit_bias: dict[int, float] | None = None`
- `max_tokens: int | None = None`
- `n_choices: int = 1`
- `best_of: int = 1`
- `presence_penalty: float | None = None`
- `frequency_penalty: float | None = None`
- `stop_seqs: list[str] | None = None`
- `temperature: float | None = None`
- `top_p: float | None = None`
- `top_k: int | None = None`
- `suffix: str | None = None`
- `grammar: str | None = None`
- `grammar_type: str | None = None`

`CompletionRequest(prompt, model, best_of, echo_prompt = False, logit_bias = None, max_tokens: None, n_choices = 1, best_of = 1, presence_penalty = None, frequency_penalty = None, stop_seqs = None, temperature = None, top_p = None, top_k = None, suffix = None, grammar = None, grammar_type = None)`


## Example
```python
from mistralrs import Runner, Which, ChatCompletionRequest, Message, Role

runner = Runner(
    which=Which.MistralGGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
    )
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[Message(Role.User, "Tell me a story about the Rust type system.")],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```