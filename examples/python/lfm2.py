"""
LiquidAI LFM2.5 text generation with the Python SDK.
"""

from mistralrs import Architecture, ChatCompletionRequest, Runner, Which

runner = Runner(
    which=Which.Plain(
        model_id="LiquidAI/LFM2.5-230M",
        arch=Architecture.Lfm2,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": "Explain what graphene is in two short paragraphs.",
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
