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
        stream=True,
    )
)
for chunk in res:
    print(chunk)
