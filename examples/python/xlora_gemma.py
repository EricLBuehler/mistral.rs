from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.XLoraGemma(
        model_id="google/gemma-7b-it",
        tokenizer_json=None,
        repeat_last_n=64,
        xlora_model_id="lamm-mit/x-lora-gemma-7b",
        order="orderings/xlora-gemma-paper-ordering.json",
        tgt_non_granular_index=None,
    )
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[{"role": "user", "content": "What is graphene?"}],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.5,
    )
)
print(res.choices[0].message.content)
print(res.usage)
