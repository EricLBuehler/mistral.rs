"""
LiquidAI LFM3 text generation with the Python SDK.

Run with `python examples/python/lfm3.py`.
The architecture is detected automatically; Q4K quantization reduces weight memory.
"""

from mistralrs import ChatCompletionRequest, Runner, Which

runner = Runner(
    which=Which.Plain(model_id="LiquidAI/LFM3-1.7B"),
    in_situ_quant="q4k",
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
    )
)
print(res.choices[0].message.content)
print(res.usage)
