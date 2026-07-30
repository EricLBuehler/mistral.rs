"""
LiquidAI LFM2.5 MoE text generation with the Python SDK.
"""

from mistralrs import Architecture, ChatCompletionRequest, Runner, Which

runner = Runner(
    which=Which.Plain(
        model_id="LiquidAI/LFM2.5-8B-A1B",
        arch=Architecture.Lfm2Moe,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": "Explain why sparse MoE models can be efficient in two short paragraphs.",
            }
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
