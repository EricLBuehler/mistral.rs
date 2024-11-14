from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.XLoraGGUF(
        tok_model_id=None,  # Automatically determine from ordering file
        quantized_model_id="TheBloke/zephyr-7B-beta-GGUF",
        quantized_filename="zephyr-7b-beta.Q4_0.gguf",
        xlora_model_id="lamm-mit/x-lora",
        order="orderings/xlora-paper-ordering.json",
        tgt_non_granular_index=None,
    ),
    which_draft=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    ),
    speculative_gamma=32,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
