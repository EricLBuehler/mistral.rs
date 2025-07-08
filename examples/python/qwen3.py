from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

# Non-MoE model
runner = Runner(
    which=Which.VisionPlain(
        model_id="https://huggingface.co/Qwen/Qwen3-4B",
        arch=VisionArchitecture.Qwen3,
    ),
    in_situ_quant="Q4K",
)

# MoE model
# runner = Runner(
#     which=Which.VisionPlain(
#         model_id="https://huggingface.co/Qwen/Qwen3-30B-A3B",
#         arch=VisionArchitecture.Qwen3Moe,
#     ),
#     in_situ_quant="Q4K",
# )

messages = [
    {
        "role": "user",
        "content": "Hello! How many rs in strawberry?",
    },
]

# ------------------------------------------------------------------
# First question, thinking mode is enabled by default
# ------------------------------------------------------------------
completion = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=messages,
        max_tokens=1024,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )
)
resp = completion.choices[0].message.content
print(resp)

messages.append({"role": "assistant", "content": completion.choices[0].message.content})

messages = [
    {
        "role": "user",
        "content": "How many rs in blueberry? /no_think",
    },
]

# ------------------------------------------------------------------
# Second question, disable thinking mode with explicit or /no_think
# ------------------------------------------------------------------
completion = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=messages,
        max_tokens=1024,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
        # enable_thinking=False
    )
)
resp = completion.choices[0].message.content
print(resp)


messages.append({"role": "assistant", "content": completion.choices[0].message.content})

messages = [
    {
        "role": "user",
        "content": "Are you sure? /think",
    },
]

# ------------------------------------------------------------------
# Third question, reenable thinking mode with explicit or /think
# ------------------------------------------------------------------
completion = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=messages,
        max_tokens=1024,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
        # enable_thinking=False
    )
)
resp = completion.choices[0].message.content
print(resp)
