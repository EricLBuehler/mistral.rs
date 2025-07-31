from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.GGUF(
        tok_model_id="Qwen/Qwen3-0.6B",
        quantized_model_id="unsloth/Qwen3-0.6B-GGUF",
        quantized_filename="Qwen3-0.6B-Q4_K_M.gguf",
    )
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
        stream=True,
    )
)
for chunk in res:
    print(chunk.choices[0].delta.content, end="", flush=True)
