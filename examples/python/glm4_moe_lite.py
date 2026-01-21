from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="zai-org/GLM-4.7-Flash",
        arch=Architecture.GLM4MoeLite,
    ),
    in_situ_quant="Q4K",
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
    )
)
print(res.choices[0].message.content)
print(res.usage)
