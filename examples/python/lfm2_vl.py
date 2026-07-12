"""
LiquidAI LFM2.5-VL image understanding with the Python SDK.
"""

from mistralrs import ChatCompletionRequest, MultimodalArchitecture, Runner, Which

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="LiquidAI/LFM2.5-VL-450M",
        arch=MultimodalArchitecture.Lfm2Vl,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image and identify the main subject.",
                    },
                ],
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
