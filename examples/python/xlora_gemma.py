from mistralrs import GemmaLoader, XLoraLoader, ChatCompletionRequest

loader = XLoraLoader(
    GemmaLoader,
    model_id="google/gemma-7b-it",
    no_kv_cache=False,
    repeat_last_n=64,
    xlora_model_id="lamm-mit/x-lora-gemma-7b",
    order_file="gemma-ordering.json",
)
runner = loader.load()
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        repetition_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res)
