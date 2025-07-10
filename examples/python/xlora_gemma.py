from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.XLora(
        model_id=None,  # Automatically determine from ordering file
        xlora_model_id="lamm-mit/x-lora-gemma-7b",
        order="orderings/xlora-gemma-paper-ordering.json",
        tgt_non_granular_index=None,
        arch=Architecture.Mistral,
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
        temperature=0.5,
    )
)
print(res.choices[0].message.content)
print(res.usage)
