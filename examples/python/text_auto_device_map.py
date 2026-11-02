from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    memory_goal=0.9,
    which=Which.Plain(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
    ),
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
